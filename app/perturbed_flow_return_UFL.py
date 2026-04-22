import os
os.environ["OMP_NUM_THREADS"] = "1"

import math

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, taylor_test
from pyadjoint import Block, AdjFloat, get_working_tape, ReducedFunctional, Control, Tape, set_working_tape, continue_annotation, taylor_to_dict
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock, solve_init_params
import ufl
from firedrake.adjoint_utils.blocks.assembly import AssembleBlock
from firedrake.adjoint_utils.blocks.block_utils import isconstant
from ufl.domain import as_domain

from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable, build_xi_diff
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole

# ---------------------------------------------------------------------------
#  Monkey-patch: fix AssembleBlock.evaluate_hessian_component
# ---------------------------------------------------------------------------
#  Firedrake's assembly.py initialises the cross-term accumulator as
#      ddform = 0.          # Python float
#  then does
#      ddform += firedrake.derivative(dform, ...)
#  The result is a UFL ``Sum(0., Form)`` which does NOT have an
#  ``.arguments()`` method, so the subsequent ``firedrake.adjoint(ddform)``
#  call crashes with
#      AttributeError: 'Sum' object has no attribute 'arguments'
#  The fix: initialise ``ddform = None``, accumulate with ``if/else``,
#  and guard the final ``compute_action_adjoint`` call with ``is not None``.
#
#  This patch is applied module-wide so every AssembleBlock on the tape
#  benefits.  It is a no-op for the forward and reverse passes (only the
#  Hessian path is affected).
# ---------------------------------------------------------------------------
def _patched_evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):

    form = prepared
    hessian_input = hessian_inputs[0]
    adj_input = adj_inputs[0]

    arity_form = len(form.arguments())

    c1 = block_variable.output
    c1_rep = block_variable.saved_output

    if isconstant(c1):
        mesh = as_domain(form)
        space = c1._ad_function_space(mesh)
    elif isinstance(c1, (Function, Cofunction)):
        space = c1.function_space()
    elif isinstance(c1, MeshGeometry):
        c1_rep = SpatialCoordinate(c1)
        space = c1._ad_function_space()
    else:
        return None

    hessian_outputs, dform = self.compute_action_adjoint(
        hessian_input, arity_form, form, c1_rep, space
    )

    # --- FIX: use None instead of 0. to avoid Sum(float, Form) ---
    ddform = None
    for other_idx, bv in relevant_dependencies:
        c2_rep = bv.saved_output
        tlm_input = bv.tlm_value

        if tlm_input is None:
            continue

        if isinstance(c2_rep, MeshGeometry):
            X = SpatialCoordinate(c2_rep)
            term = derivative(dform, X, tlm_input)
        else:
            term = derivative(dform, c2_rep, tlm_input)

        ddform = term if ddform is None else ddform + term

    if ddform is not None and adj_input is not None:
        ddform = ufl.algorithms.expand_derivatives(ddform)
        if not (isinstance(ddform, ufl.ZeroBaseForm)
                or (isinstance(ddform, ufl.Form) and ddform.empty())):
            if hasattr(ddform, 'arguments'):
                # Standard symbolic path.
                hessian_outputs += self.compute_action_adjoint(
                    adj_input, arity_form, dform=ddform
                )[0]
            else:
                # Numerical fallback for Interpolate forms.
                #
                # When the original form is a ``firedrake.Interpolate``
                # (arity 1), ``dform`` is a
                # ``BaseFormOperatorDerivative`` and the second
                # derivative ``derivative(dform, c2, tlm)`` also returns
                # a ``BaseFormOperatorDerivative``.  Summing two of
                # these gives a plain UFL ``Sum`` that does NOT support
                # ``.arguments()`` — so the symbolic path via
                # ``firedrake.adjoint(ddform)`` crashes.
                #
                # The workaround mirrors what firedrake already does for
                # SpatialCoordinate derivatives: assemble ``ddform``
                # into a concrete tensor and do the action numerically.
                assembled_ddform = firedrake.assemble(ddform)
                if assembled_ddform != 0:
                    if arity_form == 0:
                        hessian_outputs += assembled_ddform._ad_mul(adj_input)
                    elif arity_form == 1:
                        adj_output = firedrake.Cofunction(space.dual())
                        mat = assembled_ddform.petscmat
                        with adj_input.dat.vec_ro as v_vec:
                            with adj_output.dat.vec as res_vec:
                                mat.multHermitian(v_vec, res_vec)
                        hessian_outputs += adj_output

    return hessian_outputs

AssembleBlock.evaluate_hessian_component = _patched_evaluate_hessian_component
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
#  Monkey-patch: fix AdjFloatExprBlock.evaluate_hessian_component
# ---------------------------------------------------------------------------
#  The cross-term ``codegen(diff=(idx, idx1))(*inputs) * adj_input * tlm_input``
#  crashes when adj_input is None (adjoint seed did not reach this block).
#  Similarly the linear term ``codegen(diff=(idx,))(*inputs) * hessian_input``
#  crashes when hessian_input is None.  Both must be guarded.
# ---------------------------------------------------------------------------
from pyadjoint.adjfloat import AdjFloatExprBlock

def _patched_adjfloat_expr_hessian(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):
    hessian_input, = hessian_inputs
    adj_input, = adj_inputs

    if hessian_input is not None:
        val = self._operator.codegen(diff=(idx,))(*inputs) * hessian_input
    else:
        val = 0.0

    if adj_input is not None:
        for idx1, dep in relevant_dependencies:
            tlm_input = dep.tlm_value
            if tlm_input is not None:
                val += (self._operator.codegen(diff=(idx, idx1))(*inputs)
                        * adj_input * tlm_input)
    return val

AdjFloatExprBlock.evaluate_hessian_component = _patched_adjfloat_expr_hessian
# ---------------------------------------------------------------------------


class CrossProductBCBlock(Block):
    """Tape block for bc = cross(e_vec, X_ref + xi - offset).

    The operation is linear in xi (CG1 vector), output lives in V_out (CG2 vector).
    The CG1→CG2 mapping is handled by firedrake's interpolation infrastructure:
    we precompute `base = cross(e, X_ref - offset)` at CG2 dofs, and the
    xi-dependent part is `cross(e, xi)` interpolated from CG1 to CG2.

    Dependencies: idx=0 → xi (CG1 vector Function)
    """

    def __init__(self, xi_fn, out_fn, e_vec, X_ref_CG2_data, offset, V_xi, V_out, M):
        super().__init__()
        self.add_dependency(xi_fn)
        self.add_output(out_fn.create_block_variable())

        e = np.asarray(e_vec, dtype=float).ravel()
        self.e_vec = e
        # Skew-symmetric matrix S(e) such that cross(e, v) = S @ v
        self.S = np.array([[0, -e[2], e[1]],
                           [e[2], 0, -e[0]],
                           [-e[1], e[0], 0]], dtype=float)
        # base = cross(e, X_ref_CG2 - offset) precomputed at CG2 nodes
        off = np.asarray(offset, dtype=float).ravel()
        self.base_data = (X_ref_CG2_data - off[np.newaxis, :]) @ self.S.T
        self.V_xi = V_xi
        self.V_out = V_out
        self.M = M          # CG1→CG2 sparse matrix (n_CG2, n_CG1)
        self.MT = M.T.tocsr()

    def _apply_S(self, data):
        """Apply cross(e, data) = data @ S^T per row."""
        return data @ self.S.T

    def recompute_component(self, inputs, block_variable, idx, prepared):
        out = block_variable.output
        with stop_annotating():
            xi_CG2 = self.M @ inputs[0].dat.data_ro  # (n_CG2, 3)
            out.dat.data[:] = self.base_data + self._apply_S(xi_CG2)
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if adj_inputs[0] is None:
            return None
        with stop_annotating():
            adj_out = np.asarray(adj_inputs[0].dat.data_ro)  # (n_CG2, 3)
            # Adjoint of cross(e, ·) is cross(·, e) = -cross(e, ·) = -(S @ v) = v @ (-S)^T
            # i.e. adj_xi_CG2 = adj_out @ (-S)^T = -adj_out @ S^T
            adj_xi_CG2 = -self._apply_S(adj_out)
            adj_xi_CG1 = self.MT @ adj_xi_CG2
            adj = Cofunction(self.V_xi.dual())
            adj.dat.data[:] = adj_xi_CG1
        return adj

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if tlm_inputs[0] is None:
            return None
        with stop_annotating():
            d_xi_CG2 = self.M @ tlm_inputs[0].dat.data_ro
            out = Function(self.V_out)
            out.dat.data[:] = self._apply_S(d_xi_CG2)
        return out

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies, prepared=None):
        # Linear block → d² = 0 → only pass through the second-order seed.
        if hessian_inputs[0] is None:
            return None
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)


def cross_product_bc(xi_fn, e_vec, X_ref, offset, V_xi, V_out, M, name="bc"):

    e = np.asarray(e_vec, dtype=float).ravel()
    S = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]], dtype=float)
    off = np.asarray(offset, dtype=float).ravel()

    # Precompute base at CG2 nodes: cross(e, X_ref_CG2 - offset)
    X_ref_CG2 = M @ X_ref.dat.data_ro  # (n_CG2, 3)

    out = Function(V_out, name=name)
    with stop_annotating():
        xi_CG2 = M @ xi_fn.dat.data_ro
        out.dat.data[:] = (X_ref_CG2 - off[np.newaxis, :] + xi_CG2) @ S.T

    if annotate_tape():
        block = CrossProductBCBlock(xi_fn, out, e, X_ref_CG2, offset, V_xi, V_out, M)
        get_working_tape().add_block(block)

    return out


class NumpyLinSolveBlock(Block):

    def __init__(self, A_bvs, b_bvs, x_fns, n):

        super().__init__()
        self.n = n
        for bv in A_bvs:
            self.add_dependency(bv)
        for bv in b_bvs:
            self.add_dependency(bv)
        for fn in x_fns:
            self.add_output(fn.create_block_variable())


    def _unpack(self, inputs):

        n = self.n
        A = np.array([float(v) for v in inputs[: n * n]]).reshape(n, n)
        b = np.array([float(v) for v in inputs[n * n :]])
        return A, b


    def recompute_component(self, inputs, block_variable, idx, prepared):

        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)
        out = block_variable.output
        with stop_annotating():
            out.dat.data[:] = x[idx]
        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):

        n = self.n
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)
        adj_x = np.zeros(n)
        for k in range(n):
            ai = adj_inputs[k]
            if ai is not None:
                adj_x[k] = float(ai.dat.data_ro[0])
        mu = np.linalg.solve(A.T, adj_x)
        if idx < n * n:
            i, j = divmod(idx, n)
            return AdjFloat(-mu[i] * x[j])
        else:
            k = idx - n * n
            return AdjFloat(mu[k])


    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):

        n = self.n
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)
        dA = np.zeros((n, n))
        db = np.zeros(n)
        for k in range(n * n):
            if tlm_inputs[k] is not None:
                dA[k // n, k % n] = float(tlm_inputs[k])
        for k in range(n):
            if tlm_inputs[n * n + k] is not None:
                db[k] = float(tlm_inputs[n * n + k])
        dx = np.linalg.solve(A, db - dA @ x)
        out = block_variable.output
        with stop_annotating():
            result = Function(out.function_space())
            result.dat.data[:] = dx[idx]
        return result


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):

        n = self.n
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)

        # Collect the adjoint seed (adj_x) from all outputs
        adj_x = np.zeros(n)
        for k in range(n):
            ai = adj_inputs[k]
            if ai is not None:
                adj_x[k] = float(ai.dat.data_ro[0])
        mu = np.linalg.solve(A.T, adj_x)

        # Collect the second-order seed (hessian_x)
        hess_x = np.zeros(n)
        for k in range(n):
            hi = hessian_inputs[k]
            if hi is not None:
                hess_x[k] = float(hi.dat.data_ro[0])

        # Collect TLM inputs: dA and db
        deps = self.get_dependencies()
        dA = np.zeros((n, n))
        db = np.zeros(n)
        for dep_idx, bv in relevant_dependencies:
            tlm = bv.tlm_value
            if tlm is None:
                continue
            if dep_idx < n * n:
                dA[dep_idx // n, dep_idx % n] = float(tlm)
            else:
                db[dep_idx - n * n] = float(tlm)

        # TLM of x: dx = A^{-1}(db - dA x)
        dx = np.linalg.solve(A, db - dA @ x)

        # TLM of mu: d_mu = A^{-T}(hess_x - dA^T mu)
        d_mu = np.linalg.solve(A.T, hess_x - dA.T @ mu)

        # Part (1): linear adjoint of hessian seed (same formula as adj
        # but with hess_x instead of adj_x, giving d_mu instead of mu)
        if idx < n * n:
            i, j = divmod(idx, n)
            part1 = -d_mu[i] * x[j]
            # Part (2): cross-term from dx
            part2 = -mu[i] * dx[j]
            return AdjFloat(part1 + part2)
        else:
            k = idx - n * n
            return AdjFloat(d_mu[k])


class RScalarBlock(Block):
    """Tape-aware extraction of the single value of a Function on R-space.

    The idiomatic ``assemble(fn * dx) / assemble(Constant(1.0) * dx)``
    pattern puts the R-space coefficient inside a UFL form, which breaks
    pyadjoint's Hessian pass via a UFL ``replace`` failure inside
    firedrake's tsfc-interface.  Worse, the implicit mesh-coordinate
    adjoint contributions from the numerator and denominator
    AssembleBlocks do not numerically cancel under the chain rule when
    the mesh changes (e.g. via a sphere-radius scaling delta_a),
    producing wrong first derivatives.

    This block bypasses both issues by reading ``fn.dat.data[0]``
    directly in ``recompute_component`` and providing exact reverse / TLM
    / Hessian routines that carry only the R-space adjoint, with no
    spurious mesh contribution.  The map ``fn -> AdjFloat(fn[0])`` is
    linear, so its second derivative vanishes; only the linear part of
    the second-order seed propagates back through
    ``evaluate_hessian_component``.
    """

    def __init__(self, fn, out):
        super().__init__()
        self.add_dependency(fn)
        self.add_output(out.create_block_variable())

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return AdjFloat(float(inputs[0].dat.data_ro[0]))

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if adj_inputs[0] is None:
            return None
        adj_val = float(adj_inputs[0])
        R_space = inputs[0].function_space()
        with stop_annotating():
            out = Cofunction(R_space.dual())
            out.dat.data[0] = adj_val
        return out

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        if tlm_inputs[0] is None:
            return None
        return AdjFloat(float(tlm_inputs[0].dat.data_ro[0]))

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear extraction: second-order term is zero, only the linear
        # adjoint of the second-order seed flows back.
        if hessian_inputs[0] is None:
            return None
        return self.evaluate_adj_component(
            inputs, hessian_inputs, block_variable, idx, prepared)


class CylProjectBlock(Block):
    """Tape-aware azimuthal/symmetric BC projection.

    Computes either the azimuthal or symmetric Stokes BC from the
    cylindrical velocity field and the rotation angles:

        mode="azim":  bc_bg = -dot(u_bar_bc, e_theta_bc) * e_theta_bc
            Since dot(u_bar_bc, e_theta_bc) = u_theta = u_cyl[:, 2],
            and e_theta_bc = [-sin_phi, cos_phi, 0]:
            out[:, 0] =  u_cyl[:, 2] * sin_phi[:]
            out[:, 1] = -u_cyl[:, 2] * cos_phi[:]
            out[:, 2] =  0

        mode="sym":  bc_sym = -(u_r * e_r_bc + u_z * e_z)
            out[:, 0] = -u_cyl[:, 0] * cos_phi[:]
            out[:, 1] = -u_cyl[:, 0] * sin_phi[:]
            out[:, 2] = -u_cyl[:, 1]

    The map is bilinear in (cos_phi/sin_phi, u_cyl): the pure second
    derivatives vanish, only the cross-partials between the trig
    functions and u_cyl components are non-zero.  This is the same
    structure as V0aCombineBlock but for CG2 scalar × CG2 vector
    instead of R-space scalar × CG2 vector.

    This block replaces ``bc.interpolate(-u_bar_a_bc)`` (resp.
    ``-u_bar_s_bc``) which involves a product of two tape-tracked
    Functions inside a single Interpolate — exactly the case that
    crashes pyadjoint's Hessian pass with ``ZeroBaseForm has no
    ufl_shape``.

    Dependencies:
        idx 0: cos_phi  (CG2 scalar Function)
        idx 1: sin_phi  (CG2 scalar Function)
        idx 2: u_cyl_3d (CG2 vector Function, 3 components)
    """

    def __init__(self, cos_phi, sin_phi, u_cyl, out, mode):
        super().__init__()
        assert mode in ("azim", "sym")
        self.mode = mode
        self.add_dependency(cos_phi)
        self.add_dependency(sin_phi)
        self.add_dependency(u_cyl)
        self.add_output(out.create_block_variable())
        self.V_out = out.function_space()

    def _compute(self, c_data, s_data, u_data):
        """Pure numpy forward computation."""
        n = c_data.shape[0]
        out = np.zeros((n, 3))
        if self.mode == "azim":
            out[:, 0] =  u_data[:, 2] * s_data
            out[:, 1] = -u_data[:, 2] * c_data
            # out[:, 2] = 0
        else:  # sym
            out[:, 0] = -u_data[:, 0] * c_data
            out[:, 1] = -u_data[:, 0] * s_data
            out[:, 2] = -u_data[:, 1]
        return out

    def recompute_component(self, inputs, block_variable, idx, prepared):
        out = block_variable.output
        with stop_annotating():
            c = inputs[0].dat.data_ro
            s = inputs[1].dat.data_ro
            u = inputs[2].dat.data_ro
            out.dat.data[:] = self._compute(c, s, u)
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if adj_inputs[0] is None:
            return None
        a = np.asarray(adj_inputs[0].dat.data_ro)  # (n, 3)
        c = inputs[0].dat.data_ro  # (n,)
        s = inputs[1].dat.data_ro
        u = inputs[2].dat.data_ro  # (n, 3)

        with stop_annotating():
            if idx == 0:  # cos_phi
                out = Cofunction(inputs[0].function_space().dual())
                out.dat.data[:] = (-u[:, 2] * a[:, 1] if self.mode == "azim"
                                   else -u[:, 0] * a[:, 0])
                return out
            elif idx == 1:  # sin_phi
                if self.mode == "azim":
                    val_arr = u[:, 2] * a[:, 0]
                else:
                    val_arr = -u[:, 0] * a[:, 1]
                out = Cofunction(inputs[1].function_space().dual())
                out.dat.data[:] = val_arr
                return out
            elif idx == 2:  # u_cyl
                out = Cofunction(self.V_out.dual())
                if self.mode == "azim":
                    out.dat.data[:, 0] = 0
                    out.dat.data[:, 1] = 0
                    out.dat.data[:, 2] = s * a[:, 0] - c * a[:, 1]
                else:
                    out.dat.data[:, 0] = -c * a[:, 0] - s * a[:, 1]
                    out.dat.data[:, 1] = -a[:, 2]
                    out.dat.data[:, 2] = 0
                return out

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        c = inputs[0].dat.data_ro
        s = inputs[1].dat.data_ro
        u = inputs[2].dat.data_ro
        dc = tlm_inputs[0].dat.data_ro if tlm_inputs[0] is not None else None
        ds = tlm_inputs[1].dat.data_ro if tlm_inputs[1] is not None else None
        du = tlm_inputs[2].dat.data_ro if tlm_inputs[2] is not None else None

        with stop_annotating():
            out = Function(self.V_out)
            data = np.zeros_like(u)
            if self.mode == "azim":
                if du is not None:
                    data[:, 0] +=  du[:, 2] * s
                    data[:, 1] += -du[:, 2] * c
                if ds is not None:
                    data[:, 0] +=  u[:, 2] * ds
                if dc is not None:
                    data[:, 1] += -u[:, 2] * dc
            else:  # sym
                if du is not None:
                    data[:, 0] += -du[:, 0] * c
                    data[:, 1] += -du[:, 0] * s
                    data[:, 2] += -du[:, 1]
                if dc is not None:
                    data[:, 0] += -u[:, 0] * dc
                if ds is not None:
                    data[:, 1] += -u[:, 0] * ds
            out.dat.data[:] = data
        return out

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        h_in = hessian_inputs[0]
        a_in = adj_inputs[0]

        # Part 1: linear pass-through of hessian seed
        if h_in is not None:
            part1 = self.evaluate_adj_component(
                inputs, hessian_inputs, block_variable, idx, prepared)
        else:
            part1 = None

        if a_in is None:
            return part1

        # Part 2: cross-partials (bilinear structure).
        # Non-zero cross-partials exist only between the trig fields
        # (deps 0,1) and u_cyl (dep 2).  No cross between deps 0 and 1,
        # and no pure second derivatives.
        a = np.asarray(a_in.dat.data_ro)
        deps = self.get_dependencies()

        with stop_annotating():
            if idx == 0:  # cos_phi — cross with u_cyl (dep 2)
                u_tlm = deps[2].tlm_value
                if u_tlm is None:
                    return part1
                du = u_tlm.dat.data_ro
                if self.mode == "azim":
                    cross_val = -du[:, 2] * a[:, 1]
                else:
                    cross_val = -du[:, 0] * a[:, 0]
                if part1 is None:
                    cross = Cofunction(inputs[0].function_space().dual())
                    cross.dat.data[:] = cross_val
                    return cross
                part1.dat.data[:] += cross_val
                return part1

            elif idx == 1:  # sin_phi — cross with u_cyl (dep 2)
                u_tlm = deps[2].tlm_value
                if u_tlm is None:
                    return part1
                du = u_tlm.dat.data_ro
                if self.mode == "azim":
                    cross_val = du[:, 2] * a[:, 0]
                else:
                    cross_val = -du[:, 0] * a[:, 1]
                if part1 is None:
                    cross = Cofunction(inputs[1].function_space().dual())
                    cross.dat.data[:] = cross_val
                    return cross
                part1.dat.data[:] += cross_val
                return part1

            elif idx == 2:  # u_cyl — cross with cos_phi (dep 0) and sin_phi (dep 1)
                cross_data = np.zeros_like(inputs[2].dat.data_ro)
                c_tlm = deps[0].tlm_value
                s_tlm = deps[1].tlm_value
                if self.mode == "azim":
                    if s_tlm is not None:
                        cross_data[:, 2] += s_tlm.dat.data_ro * a[:, 0]
                    if c_tlm is not None:
                        cross_data[:, 2] += -c_tlm.dat.data_ro * a[:, 1]
                else:
                    if c_tlm is not None:
                        cross_data[:, 0] += -c_tlm.dat.data_ro * a[:, 0]
                    if s_tlm is not None:
                        cross_data[:, 0] += -s_tlm.dat.data_ro * a[:, 1]

                if not np.any(cross_data):
                    return part1
                if part1 is None:
                    cross = Cofunction(self.V_out.dual())
                    cross.dat.data[:] = cross_data
                    return cross
                part1.dat.data[:] += cross_data
                return part1


class V0aCombineBlock(Block):

    def __init__(self, T, Ox, Oy, Oz, v_T, v_Ox, v_Oy, v_Oz, v_bg, out):
        super().__init__()
        for dep in (T, Ox, Oy, Oz, v_T, v_Ox, v_Oy, v_Oz, v_bg):
            self.add_dependency(dep)
        self.add_output(out.create_block_variable())
        self.V_out = out.function_space()

    @staticmethod
    def _scalars_from(inputs):
        return (
            float(inputs[0].dat.data_ro[0]),
            float(inputs[1].dat.data_ro[0]),
            float(inputs[2].dat.data_ro[0]),
            float(inputs[3].dat.data_ro[0]),
        )


    def recompute_component(self, inputs, block_variable, idx, prepared):
        T, Ox, Oy, Oz = self._scalars_from(inputs)
        v_T, v_Ox, v_Oy, v_Oz, v_bg = (inputs[4], inputs[5],
                                        inputs[6], inputs[7], inputs[8])
        out = block_variable.output
        with stop_annotating():
            data = (T * v_T.dat.data_ro
                    + Ox * v_Ox.dat.data_ro
                    + Oy * v_Oy.dat.data_ro
                    + Oz * v_Oz.dat.data_ro
                    + v_bg.dat.data_ro)
            out.dat.data[:] = data
        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if adj_inputs[0] is None:
            return None
        adj_data = np.asarray(adj_inputs[0].dat.data_ro)
        T, Ox, Oy, Oz = self._scalars_from(inputs)
        scalars = (T, Ox, Oy, Oz)
        with stop_annotating():
            if 0 <= idx <= 3:
                # adj wrt R-space scalar c_i = sum(adj * v_i.dat)
                v_i = inputs[4 + idx]
                val = float(np.sum(v_i.dat.data_ro * adj_data))
                R_space = inputs[idx].function_space()
                out = Cofunction(R_space.dual())
                out.dat.data[0] = val
                return out
            elif 4 <= idx <= 7:
                # adj wrt CG2 v_i = c_i * adj
                c = scalars[idx - 4]
                out = Cofunction(self.V_out.dual())
                out.dat.data[:] = c * adj_data
                return out
            elif idx == 8:
                # adj wrt v_bg (coefficient is 1)
                out = Cofunction(self.V_out.dual())
                out.dat.data[:] = adj_data
                return out
            else:
                raise IndexError(f"unexpected dependency index {idx}")


    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):

        T, Ox, Oy, Oz = self._scalars_from(inputs)
        v_T, v_Ox, v_Oy, v_Oz, v_bg = (inputs[4], inputs[5],
                                        inputs[6], inputs[7], inputs[8])
        scalars = (T, Ox, Oy, Oz)
        v_i_list = (v_T, v_Ox, v_Oy, v_Oz)
        with stop_annotating():
            data = np.zeros_like(v_bg.dat.data_ro, dtype=float)

            for i in range(4):
                if tlm_inputs[i] is not None:
                    data += float(tlm_inputs[i].dat.data_ro[0]) \
                            * v_i_list[i].dat.data_ro

            for i in range(4):
                if tlm_inputs[4 + i] is not None:
                    data += scalars[i] * tlm_inputs[4 + i].dat.data_ro
            if tlm_inputs[8] is not None:
                data += tlm_inputs[8].dat.data_ro

            out = Function(self.V_out)
            out.dat.data[:] = data
        return out


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies,
                                   prepared=None):

        h_in = hessian_inputs[0]
        a_in = adj_inputs[0]

        # Part 1: linear adjoint of the second-order seed.
        if h_in is not None:
            part1 = self.evaluate_adj_component(
                inputs, hessian_inputs, block_variable, idx, prepared)
        else:
            part1 = None

        if a_in is None:
            return part1

        # Part 2: cross-partial contributions.
        # Multilinear structure ⇒ ∂²(out)/∂(c_i)∂(v_j) is non-zero only
        # for i == j.  Specifically, ∂²(out[node])/∂(c_i)∂(v_i[node]) = 1
        # (and zero otherwise), so the cross-partial sums are simple.
        adj_data = np.asarray(a_in.dat.data_ro)
        deps = self.get_dependencies()

        with stop_annotating():
            if 0 <= idx <= 3:
                # idx is R-space scalar c_i; cross with v_i (dep idx + 4).
                v_i_tlm = deps[4 + idx].tlm_value
                if v_i_tlm is None:
                    return part1
                cross_val = float(np.sum(v_i_tlm.dat.data_ro * adj_data))
                if part1 is None:
                    R_space = inputs[idx].function_space()
                    cross = Cofunction(R_space.dual())
                    cross.dat.data[0] = cross_val
                    return cross
                part1.dat.data[0] += cross_val
                return part1
            elif 4 <= idx <= 7:
                # idx is CG2 v_i; cross with c_i (dep idx - 4).
                c_tlm = deps[idx - 4].tlm_value
                if c_tlm is None:
                    return part1
                c_tlm_val = float(c_tlm.dat.data_ro[0])
                if part1 is None:
                    cross = Cofunction(self.V_out.dual())
                    cross.dat.data[:] = c_tlm_val * adj_data
                    return cross
                part1.dat.data[:] += c_tlm_val * adj_data
                return part1
            elif idx == 8:
                # No cross-partial for v_bg (coefficient is constant 1).
                return part1
            else:
                raise IndexError(f"unexpected dependency index {idx}")


def numpy_lin_solve_to_R(A_adj_floats, b_adj_floats, R_space, n):

    A_np = np.array([float(a) for a in A_adj_floats]).reshape(n, n)
    b_np = np.array([float(b) for b in b_adj_floats])
    x_np = np.linalg.solve(A_np, b_np)
    x_fns = []
    for k in range(n):
        f = Function(R_space, name=f"linsys_x_{k}")
        with stop_annotating():
            f.dat.data[:] = x_np[k]
        x_fns.append(f)
    if annotate_tape():
        block = NumpyLinSolveBlock(A_adj_floats, b_adj_floats, x_fns, n)
        get_working_tape().add_block(block)
    return x_fns


def r_scalar(fn):
    """Convert a Function on an R-space to an AdjFloat on the tape.

    Used in F_p() to lift the linear-solve outputs Theta_fn / Omega_*_fn
    out of the centrifugal and fluid_stress forms before they are used
    quadratically.  See RScalarBlock for the rationale.
    """
    val = AdjFloat(float(fn.dat.data_ro[0]))
    if annotate_tape():
        block = RScalarBlock(fn, val)
        get_working_tape().add_block(block)
    return val


def cyl_project(cos_phi_fn, sin_phi_fn, u_cyl, V, mode):
    """Tape-aware computation of the azimuthal or symmetric Stokes BC.

    See CylProjectBlock for the mathematical definition.
    ``mode`` must be ``"azim"`` or ``"sym"``.
    """
    out = Function(V, name=f"bc_{'bg' if mode == 'azim' else 'sym'}")
    c = cos_phi_fn.dat.data_ro
    s = sin_phi_fn.dat.data_ro
    u = u_cyl.dat.data_ro
    with stop_annotating():
        if mode == "azim":
            out.dat.data[:, 0] =  u[:, 2] * s
            out.dat.data[:, 1] = -u[:, 2] * c
        else:
            out.dat.data[:, 0] = -u[:, 0] * c
            out.dat.data[:, 1] = -u[:, 0] * s
            out.dat.data[:, 2] = -u[:, 1]
    if annotate_tape():
        block = CylProjectBlock(cos_phi_fn, sin_phi_fn, u_cyl, out, mode)
        get_working_tape().add_block(block)
    return out


def combine_v_0_a(T_fn, Ox_fn, Oy_fn, Oz_fn, v_T, v_Ox, v_Oy, v_Oz, v_bg, V):
    """Tape-aware construction of
        v_0_a = T*v_T + Ox*v_Ox + Oy*v_Oy + Oz*v_Oz + v_bg
    on the function space ``V``.  Replaces the equivalent
    ``Function.interpolate(...)`` call so the resulting tape has a
    custom block whose Hessian path is exact (see V0aCombineBlock for
    the rationale).
    """
    out = Function(V, name="v_0_a")
    with stop_annotating():
        T = float(T_fn.dat.data_ro[0])
        Ox = float(Ox_fn.dat.data_ro[0])
        Oy = float(Oy_fn.dat.data_ro[0])
        Oz = float(Oz_fn.dat.data_ro[0])
        out.dat.data[:] = (T * v_T.dat.data_ro
                           + Ox * v_Ox.dat.data_ro
                           + Oy * v_Oy.dat.data_ro
                           + Oz * v_Oz.dat.data_ro
                           + v_bg.dat.data_ro)
    if annotate_tape():
        block = V0aCombineBlock(T_fn, Ox_fn, Oy_fn, Oz_fn,
                                v_T, v_Ox, v_Oy, v_Oz, v_bg, out)
        get_working_tape().add_block(block)
    return out


class CachedStokesContext:

    def __init__(self, V, Q, tags, mesh):
        self.V = V
        self.Q = Q
        self.W = V * Q
        self.tags = tags
        self.mesh = mesh

        v_trial, p_trial = TrialFunctions(self.W)
        v_test, q_test = TestFunctions(self.W)

        self.a_form = (
            2 * inner(sym(grad(v_trial)), sym(grad(v_test))) * dx
            - p_trial * div(v_test) * dx
            + q_test * div(v_trial) * dx
        )
        self.L_form = inner(Constant((0.0, 0.0, 0.0)), v_test) * dx

        self.nullspace = MixedVectorSpaceBasis(
            self.W,
            [self.W.sub(0), VectorSpaceBasis(constant=True, comm=self.W.comm)],
        )
        self.bcs_hom = [
            DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), tags["walls"]),
            DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), tags["particle"]),
        ]

        self._fwd_solver = None
        self._adj_solver = None
        self._fp = None
        self.fwd_factor_count = 0
        self.adj_factor_count = 0


    def _fingerprint(self):
        return hash(self.mesh.coordinates.dat.data_ro.tobytes())


    def _invalidate(self):
        self._fwd_solver = None
        self._adj_solver = None


    def get_fwd_solver(self):
        fp = self._fingerprint()
        if self._fwd_solver is None or abs(fp - self._fp) > 1e-14:
            A = assemble(self.a_form, bcs=self.bcs_hom)
            self._fwd_solver = LinearSolver(A, nullspace=self.nullspace,
                                            solver_parameters={
                                                                "ksp_type": "preonly",
                                                                "pc_type": "lu",
                                                                "pc_factor_mat_solver_type": "mumps",
                                                                "mat_mumps_icntl_24": 1,
                                                                "mat_mumps_icntl_25": 0,
                                            })

            self._adj_solver = None
            self._fp = fp
            self.fwd_factor_count += 1
        return self._fwd_solver


    def get_adj_solver(self):
        self.get_fwd_solver()
        if self._adj_solver is None:
            a_adj = adjoint(self.a_form)
            B = assemble(a_adj, bcs=self.bcs_hom)
            self._adj_solver = LinearSolver(B, nullspace=self.nullspace,
                                            solver_parameters={
                                                                "ksp_type": "preonly",
                                                                "pc_type": "lu",
                                                                "pc_factor_mat_solver_type": "mumps",
                                                                "mat_mumps_icntl_24": 1,
                                                                "mat_mumps_icntl_25": 0,
                                                                })
            self.adj_factor_count += 1
        return self._adj_solver


    def _lift_rhs(self, a_form, L_form, bcs):

        W = self.W
        w_lift = Function(W)
        for bc in bcs:
            bc.apply(w_lift)

        b = assemble(L_form - action(a_form, w_lift), bcs=self.bcs_hom)

        return b, w_lift


class CachedStokesSolveBlock(GenericSolveBlock):

    def __init__(self, lhs, rhs, func, bcs, ctx, *args, **kwargs):
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)
        self.ctx = ctx

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)

    def _forward_solve(self, lhs, rhs, func, bcs):
        b, w_lift = self.ctx._lift_rhs(lhs, rhs, bcs)
        solver = self.ctx.get_fwd_solver()
        solver.solve(func, b)
        func += w_lift
        return func


    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()

        # Adjoint BCs are homogeneous — zero out BC DOFs in the adjoint RHS.
        # bc.apply() fails on Cofunctions (dual space != primal space),
        # so zero the velocity DOFs directly via the underlying data array.
        for bc in self.ctx.bcs_hom:
            dJdu.dat[0].data_wo[bc.nodes] = 0.0

        adj_solver = self.ctx.get_adj_solver()
        adj_sol = Function(self.function_space)
        adj_solver.solve(adj_sol, dJdu)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = self._compute_adj_bdy(
                adj_sol, adj_sol_bdy, dFdu_adj_form, dJdu_copy)
        return adj_sol, adj_sol_bdy

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs,
                                relevant_dependencies):
        """Override: fix the SOA computation for linear Stokes with BC deps.

        Firedrake's GenericSolveBlock has a bug in the Hessian propagation
        through DirichletBC dependencies: _compute_adj_bdy uses
        .riesz_representation("l2") which introduces ~20% error when the
        Hessian flows only through BCs (no mesh movement).

        For a LINEAR PDE (Stokes), the SOA equation simplifies because:
        - d²F/du² = 0  (bilinear form, linear in u)
        - d²F/du·dm = 0 for non-mesh, non-BC deps (A is constant)

        The SOA RHS is simply the hessian_input (from downstream).
        We solve A^T · adj_sol2 = b with homogeneous BCs, then compute
        the boundary residual WITHOUT the buggy L2 Riesz conversion.
        """
        fwd_block_variable = self.get_outputs()[0]
        hessian_input = hessian_inputs[0]
        tlm_output = fwd_block_variable.tlm_value

        if hessian_input is None or tlm_output is None:
            return None

        F_form = self._create_F_form()
        dFdu_form = firedrake.derivative(F_form, fwd_block_variable.saved_output)

        # d²F/du² · tlm — zero for linear PDE
        d2Fdu2 = ufl.algorithms.expand_derivatives(
            firedrake.derivative(dFdu_form, fwd_block_variable.saved_output,
                                 tlm_output))

        adj_sol = self.adj_sol
        if adj_sol is None:
            raise RuntimeError("Hessian computation was run before adjoint.")

        bdy = self._should_compute_boundary_adjoint(relevant_dependencies)

        # Assemble the SOA RHS (uses parent's method — correct)
        b = self._assemble_soa_eq_rhs(dFdu_form, adj_sol, hessian_input,
                                      d2Fdu2)

        # Solve SOA: A^T · adj_sol2 = b with homogeneous BCs
        b_full = b.copy()
        for bc in self.ctx.bcs_hom:
            b.dat[0].data_wo[bc.nodes] = 0.0

        adj_solver = self.ctx.get_adj_solver()
        adj_sol2 = Function(self.function_space)
        adj_solver.solve(adj_sol2, b)

        # Compute boundary residual WITHOUT the buggy .riesz_representation("l2").
        # The residual b_full - A^T_full · adj_sol2 is a Cofunction.
        # We convert it to a Function by pointwise DOF copy.
        adj_sol2_bdy = None
        if bdy:
            dFdu_adj_form = firedrake.adjoint(dFdu_form)
            residual_cofn = firedrake.assemble(
                b_full - firedrake.action(dFdu_adj_form, adj_sol2))
            adj_sol2_bdy = Function(self.function_space)
            for k in range(len(adj_sol2_bdy.dat)):
                adj_sol2_bdy.dat[k].data[:] = residual_cofn.dat[k].data_ro[:]

        return {
            "adj_sol2": adj_sol2,
            "adj_sol2_bdy": adj_sol2_bdy,
            "form": F_form,
            "adj_sol": adj_sol,
        }


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        """Override: use adj_sol2_bdy from our fixed prepare_evaluate_hessian.

        For non-BC dependencies, delegate to parent (shape Hessian etc.).
        For DirichletBC dependencies, extract the velocity subfunction
        from our corrected adj_sol2_bdy (pointwise copy, no L2 Riesz).
        """
        c = block_variable.output

        if not isinstance(c, firedrake.DirichletBC):
            return super().evaluate_hessian_component(
                inputs, hessian_inputs, adj_inputs,
                block_variable, idx, relevant_dependencies, prepared)

        if prepared is None:
            return None

        adj_sol2_bdy = prepared["adj_sol2_bdy"]
        if adj_sol2_bdy is None:
            return None

        # Extract velocity subfunction from adj_sol2_bdy (mixed space)
        V_bc = c.function_space()
        g_hess = Function(V_bc)
        g_hess.dat.data[:] = adj_sol2_bdy.subfunctions[0].dat.data_ro[:]
        return [c.reconstruct(g=g_hess)]


    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        solver = self.ctx.get_fwd_solver()

        # The cached solver has identity BC rows but NON-zero BC columns
        # (standard non-symmetric assembly).  So the interior equations
        # couple to BC DOFs: A[int,bc] * u[bc] appears in the interior
        # rows.  For a correct solve with non-zero TLM BCs, we must
        # subtract this coupling from the RHS (= lifting technique).
        #
        # The standard firedrake solve does this automatically, but our
        # cached solver doesn't.  We implement it explicitly:
        #
        #   1. Build w_lift from TLM BCs
        #   2. Subtract action(a_form, w_lift) from RHS (lifting)
        #   3. Zero BC DOFs in RHS
        #   4. Solve with cached homogeneous-BC solver
        #   5. Add w_lift back to solution

        # Step 1: Build TLM lift
        w_lift_tlm = Function(self.function_space)
        for bc in bcs:
            bc.apply(w_lift_tlm)

        # Step 2: Lifting — subtract A_raw * w_lift from RHS
        a_form = self.ctx.a_form
        lift_contrib = firedrake.assemble(firedrake.action(a_form, w_lift_tlm))
        dFdm -= lift_contrib

        # Step 3: Zero BC DOFs in RHS
        for bc in self.ctx.bcs_hom:
            dFdm.dat[0].data_wo[bc.nodes] = 0.0

        # Step 4: Solve with cached solver (homogeneous BCs)
        solver.solve(dudm, dFdm)

        # Step 5: Add TLM lift back
        dudm += w_lift_tlm

        return dudm


def stokes_solve_cached(V, Q, particle_bc_expr, tags, mesh, ctx):
    
    W = ctx.W
    a_form = ctx.a_form
    L_form = ctx.L_form

    bcs = [
        DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), tags["walls"]),
        DirichletBC(W.sub(0), particle_bc_expr, tags["particle"]),
    ]

    w = Function(W)


    with stop_annotating():
        b, w_lift = ctx._lift_rhs(a_form, L_form, bcs)
        solver = ctx.get_fwd_solver()
        solver.solve(w, b)
        w += w_lift


    if annotate_tape():
        block = CachedStokesSolveBlock(a_form, L_form, w, bcs, ctx,
            solver_parameters={
                                "ksp_type": "preonly",
                                "pc_type": "lu",
                                "pc_factor_mat_solver_type": "mumps",
                                "mat_mumps_icntl_24": 1,
                                "mat_mumps_icntl_25": 0
            }
        )

        get_working_tape().add_block(block)
        # GenericSolveBlock.__init__ may already register w as an output.
        # Only add_output if it was NOT already registered by super().__init__.
        if not any(bv.output is w for bv in block.get_outputs()):
            block.add_output(w.create_block_variable())

    v_out = Function(V, name="v_stokes")
    p_out = Function(Q, name="p_stokes")
    v_out.assign(w.subfunctions[0])
    p_out.assign(w.subfunctions[1])
    return v_out, p_out


class perturbed_flow_differentiable:

    def __init__(self, R, H, W, L, a, Re_p, mesh3d, tags, u_bar_3d, p_bar_3d, X_ref, xi, u_cyl_3d):
        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.L = L
        self.Re_p = Re_p
        self.u_bar = u_bar_3d
        self.p_bar = p_bar_3d
        self.mesh3d = mesh3d
        self.tags = tags

        self.x = SpatialCoordinate(self.mesh3d)
        self.x_p = Constant(self.tags["particle_center"])

        self.e_x_prime = Constant([math.cos(self.L / self.R * 0.5),
                                   math.sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-math.sin(self.L / self.R * 0.5),
                                   math.cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        r_xy = sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.e_r_prime = as_vector([self.x[0] / r_xy, self.x[1] / r_xy, 0])
        self.e_theta_prime = as_vector([-self.x[1] / r_xy,
                                        self.x[0] / r_xy, 0])

        self.x_bc = X_ref + xi
        r_xy_bc = sqrt(self.x_bc[0] ** 2 + self.x_bc[1] ** 2)
        self.e_r_bc = as_vector([self.x_bc[0] / r_xy_bc,
                                 self.x_bc[1] / r_xy_bc, 0])
        self.e_theta_bc = as_vector([-self.x_bc[1] / r_xy_bc,
                                     self.x_bc[0] / r_xy_bc, 0])
        self.u_bar_bc = as_vector([
            self.x_bc[0] / r_xy_bc * u_cyl_3d[0]
            - self.x_bc[1] / r_xy_bc * u_cyl_3d[2],
            self.x_bc[1] / r_xy_bc * u_cyl_3d[0]
            + self.x_bc[0] / r_xy_bc * u_cyl_3d[2],
            u_cyl_3d[1],
        ])

        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.R_space = FunctionSpace(self.mesh3d, "R", 0)
        self.u_cyl_3d = u_cyl_3d
        self.xi = xi
        self.X_ref = X_ref
        self.V_xi = xi.function_space()  # CG1 vector

        # CG1→CG2 interpolation map for CrossProductBCBlock
        from background_flow_return_UFL import _build_CG1_to_CG2_map
        with stop_annotating():
            self._M_CG1_CG2 = _build_CG1_to_CG2_map(mesh3d)

        # Direction vectors as numpy arrays for CrossProductBCBlock
        self._e_x_np = np.array([math.cos(self.L / self.R * 0.5),
                                 math.sin(self.L / self.R * 0.5), 0.0])
        self._e_y_np = np.array([-math.sin(self.L / self.R * 0.5),
                                  math.cos(self.L / self.R * 0.5), 0.0])
        self._e_z_np = np.array([0.0, 0.0, 1.0])
        self._x_p_np = np.array(self.tags["particle_center"])

        # Pre-interpolate the rotation angles into CG2 scalar Functions
        # ON the tape.  Each interpolate involves only ONE tape-tracked
        # coefficient (xi, through x_bc = X_ref + xi), so the Hessian
        # path via InterpolateBlock works (Layer 2b-iii proved this).
        # These are then passed into CylProjectBlock to compute bc_bg
        # and bc_sym WITHOUT any product-of-two-tape-Functions in an
        # Interpolate — which is the root cause of the Hessian crash.
        Q_CG2 = FunctionSpace(self.mesh3d, "CG", 2)
        self.cos_phi_bc = Function(Q_CG2, name="cos_phi_bc")
        self.cos_phi_bc.interpolate(self.x_bc[0] / r_xy_bc)
        self.sin_phi_bc = Function(Q_CG2, name="sin_phi_bc")
        self.sin_phi_bc.interpolate(self.x_bc[1] / r_xy_bc)

        with stop_annotating():
            self.stokes_ctx = CachedStokesContext(
                self.V, self.Q, self.tags, self.mesh3d)


    def _solve_stokes(self, particle_bc_expr):
        return stokes_solve_cached(
            self.V, self.Q, particle_bc_expr, self.tags,
            self.mesh3d, self.stokes_ctx)


    def _force_components(self, v, q):
        n = FacetNormal(self.mesh3d)
        sigma = -q * Identity(3) + (grad(v) + grad(v).T)
        traction = dot(-n, sigma)
        return [assemble(traction[i] * ds(self.tags["particle"], degree=8))
                for i in range(3)]


    def _torque_components(self, v, q):
        n = FacetNormal(self.mesh3d)
        sigma = -q * Identity(3) + (grad(v) + grad(v).T)
        traction = dot(-n, sigma)
        moment = cross(self.x - self.x_p, traction)
        return [assemble(moment[i] * ds(self.tags["particle"], degree=8))
                for i in range(3)]

    @staticmethod
    def _dot3(e_np, comps):
        return (float(e_np[0]) * comps[0] + float(e_np[1]) * comps[1]
                + float(e_np[2]) * comps[2])


    def F_p(self, return_components=False):

        V = self.V
        x_bc = self.x_bc

        u_bar_a = dot(self.u_bar, self.e_theta_prime) * self.e_theta_prime
        u_bar_s = (dot(self.u_bar, self.e_r_prime) * self.e_r_prime
                   + dot(self.u_bar, self.e_z_prime) * self.e_z_prime)

        u_bar_a_bc = dot(self.u_bar_bc, self.e_theta_bc) * self.e_theta_bc
        u_bar_s_bc = (dot(self.u_bar_bc, self.e_r_bc) * self.e_r_bc
                      + dot(self.u_bar_bc, self.e_z_prime) * self.e_z_prime)

        # Use CrossProductBCBlock instead of InterpolateBlock.
        # InterpolateBlock's Hessian is broken for cross(const, X_ref+xi).
        M = self._M_CG1_CG2
        zeros3 = np.zeros(3)
        bc_Theta = cross_product_bc(self.xi, self._e_z_np, self.X_ref, zeros3,
                                    self.V_xi, V, M, name="bc_Theta")
        bc_Ox = cross_product_bc(self.xi, self._e_x_np, self.X_ref, self._x_p_np,
                                 self.V_xi, V, M, name="bc_Ox")
        bc_Oy = cross_product_bc(self.xi, self._e_y_np, self.X_ref, self._x_p_np,
                                 self.V_xi, V, M, name="bc_Oy")
        bc_Oz = cross_product_bc(self.xi, self._e_z_np, self.X_ref, self._x_p_np,
                                 self.V_xi, V, M, name="bc_Oz")
        # bc_bg = -u_bar_a_bc = -u_theta * e_theta via CylProjectBlock.
        # Mathematically:  bc_bg[:, 0] = -u_cyl[:, 2] * sin_phi
        #                  bc_bg[:, 1] =  u_cyl[:, 2] * cos_phi
        # This avoids the product-of-two-tape-Functions in an Interpolate
        # that crashes pyadjoint's Hessian (Layer 2b-v in diagnostic).
        bc_bg = cyl_project(self.cos_phi_bc, self.sin_phi_bc, self.u_cyl_3d, V, mode="azim")

        v_Theta, q_Theta = self._solve_stokes(bc_Theta)
        v_Ox, q_Ox = self._solve_stokes(bc_Ox)
        v_Oy, q_Oy = self._solve_stokes(bc_Oy)
        v_Oz, q_Oz = self._solve_stokes(bc_Oz)
        v_bg, q_bg = self._solve_stokes(bc_bg)

        F_Theta = self._force_components(v_Theta, q_Theta)
        F_Ox = self._force_components(v_Ox, q_Ox)
        F_Oy = self._force_components(v_Oy, q_Oy)
        F_Oz = self._force_components(v_Oz, q_Oz)
        F_bg = self._force_components(v_bg, q_bg)

        T_Theta = self._torque_components(v_Theta, q_Theta)
        T_Ox = self._torque_components(v_Ox, q_Ox)
        T_Oy = self._torque_components(v_Oy, q_Oy)
        T_Oz = self._torque_components(v_Oz, q_Oz)
        T_bg = self._torque_components(v_bg, q_bg)

        e_x = np.array(self.e_x_prime.values())
        e_y = np.array(self.e_y_prime.values())
        e_z = np.array(self.e_z_prime.values())
        d = self._dot3

        A_flat = [
            d(e_y, F_Theta), d(e_y, F_Oz), d(e_y, F_Ox), d(e_y, F_Oy),
            d(e_x, T_Theta), d(e_x, T_Oz), d(e_x, T_Ox), d(e_x, T_Oy),
            d(e_y, T_Theta), d(e_y, T_Oz), d(e_y, T_Ox), d(e_y, T_Oy),
            d(e_z, T_Theta), d(e_z, T_Oz), d(e_z, T_Ox), d(e_z, T_Oy),
        ]
        b_flat = [
            -d(e_y, F_bg), -d(e_x, T_bg),
            -d(e_y, T_bg), -d(e_z, T_bg),
        ]

        Theta_fn, Omega_z_fn, Omega_x_fn, Omega_y_fn = \
            numpy_lin_solve_to_R(A_flat, b_flat, self.R_space, 4)

        # v_0_a = T*v_T + Ox*v_Ox + Oy*v_Oy + Oz*v_Oz + v_bg via custom
        # block.  Replaces a Function.interpolate(...) of an expression
        # mixing R-space and CG2 coefficients, which is exactly the case
        # that breaks pyadjoint's Hessian path through downstream forms
        # involving v_0_a (see V0aCombineBlock for details).
        v_0_a = combine_v_0_a(Theta_fn, Omega_x_fn, Omega_y_fn, Omega_z_fn,
                              v_Theta, v_Ox, v_Oy, v_Oz, v_bg, self.V)

        # bc_sym = -u_bar_s_bc = -(u_r*e_r + u_z*e_z) via CylProjectBlock.
        # Mathematically:  bc_sym[:, 0] = -u_cyl[:, 0] * cos_phi
        #                  bc_sym[:, 1] = -u_cyl[:, 0] * sin_phi
        #                  bc_sym[:, 2] = -u_cyl[:, 1]
        bc_sym = cyl_project(self.cos_phi_bc, self.sin_phi_bc,
                             self.u_cyl_3d, V, mode="sym")
        v_0_s, q_0_s = self._solve_stokes(bc_sym)

        bc_ex = Function(V, name="bc_ex")
        bc_ex.interpolate(self.e_x_prime)
        bc_ez = Function(V, name="bc_ez")
        bc_ez.interpolate(self.e_z_prime)
        u_hat_x, _ = self._solve_stokes(bc_ex)
        u_hat_z, _ = self._solve_stokes(bc_ez)

        F_s = self._force_components(v_0_s, q_0_s)
        F_s_x = d(e_x, F_s)
        F_s_z = d(e_z, F_s)

        x_p_np = np.array(self.x_p.values())
        cent_vec = np.cross(e_z, np.cross(e_z, x_p_np))
        cent_coeff_x = float(np.dot(e_x, cent_vec))
        cent_coeff_z = float(np.dot(e_z, cent_vec))

        # Lift Theta_fn (R-space) into an AdjFloat once.  Used by both the
        # centrifugal term (Theta^2 contribution) and the fluid_stress
        # split below.  See RScalarBlock for why we cannot use the
        # ``assemble(Theta_fn * dx) / vol`` shortcut.
        T_adj = r_scalar(Theta_fn)
        T_squared = T_adj * T_adj

        # Centrifugal term: F_cent_j = -(4π/3) a^3 (e_j · (e_z × (e_z × x_p)))
        #                              · <Theta^2>
        # Computed entirely in AdjFloat arithmetic so that no R-space
        # coefficient ever sits inside a UFL form.  The chain
        #     delta_r/z/a -> Theta_fn -> T_adj -> T_squared -> centrifugal
        # has a working Hessian path: T_adj's RScalarBlock is exact, and
        # the AdjFloat product T_adj * T_adj is handled by pyadjoint's
        # standard FloatOperatorBlock.
        unit_cent_x = (-4.0 / 3.0) * np.pi * cent_coeff_x * T_squared
        unit_cent_z = (-4.0 / 3.0) * np.pi * cent_coeff_z * T_squared

        if isinstance(self.a, tuple):
            # Tuple (a_base, delta_a_fn): the total particle radius is
            #     a = a_base + delta_a
            # Doing the addition in AdjFloat arithmetic (via r_scalar)
            # avoids putting ``Function.assign(Constant + Function)``
            # on the tape, whose InterpolateBlock crashes in the Hessian
            # pass with ``'Sum' object has no attribute 'arguments'``.
            a_base, a_delta_fn = self.a
            a_val = float(a_base) + r_scalar(a_delta_fn)
        elif isinstance(self.a, (int, float)):
            a_val = self.a
        elif hasattr(self.a, "function_space"):
            a_val = r_scalar(self.a)
        else:
            a_val = float(self.a)

        a_cubed = a_val * a_val * a_val
        centrifugal_x = a_cubed * unit_cent_x
        centrifugal_z = a_cubed * unit_cent_z

        n_hat = FacetNormal(self.mesh3d)
        inertial_integrand = (
            dot(u_bar_a, -n_hat) * u_bar_a
            + dot(u_bar_s, -n_hat) * u_bar_a
            + dot(u_bar_a, -n_hat) * u_bar_s
            + dot(u_bar_s, -n_hat) * u_bar_s
        )
        inertial_x = assemble(
            dot(self.e_x_prime, inertial_integrand)
            * ds(self.tags["particle"], degree=6))
        inertial_z = assemble(
            dot(self.e_z_prime, inertial_integrand)
            * ds(self.tags["particle"], degree=6))

        rhs_no_T = (
            dot(grad(u_bar_a), v_0_a)
            + dot(grad(v_0_a), v_0_a + u_bar_a)
            + dot(grad(u_bar_s), v_0_s)
            + dot(grad(v_0_s), v_0_s + u_bar_s)
            + dot(grad(u_bar_s), v_0_a)
            + dot(grad(u_bar_a), v_0_s)
            + dot(grad(v_0_s), v_0_a + u_bar_a)
            + dot(grad(v_0_a), v_0_s + u_bar_s)
        )

        # Use the UFL expression directly instead of interpolating into a
        # Function.  The interpolation would put an InterpolateBlock on the
        # tape whose Hessian is broken for mesh-coordinate-dependent
        # expressions (same bug as bc_Theta before CrossProductBCBlock fix).
        # cross(e_z, x) = [-y, x, 0] is a simple permutation of coordinates
        # and can remain as a UFL expression in the form.
        ez_cross_x = cross(self.e_z_prime, self.x)

        rhs_T_unit = (
            cross(self.e_z_prime, v_0_a)
            + cross(self.e_z_prime, v_0_s)
            - dot(grad(v_0_a), ez_cross_x)
            - dot(grad(v_0_s), ez_cross_x)
        )

        fluid_stress_x_no_T = assemble(
            -dot(u_hat_x, rhs_no_T) * dx(degree=6))
        fluid_stress_z_no_T = assemble(
            -dot(u_hat_z, rhs_no_T) * dx(degree=6))
        fluid_stress_x_T_unit = assemble(
            -dot(u_hat_x, rhs_T_unit) * dx(degree=6))
        fluid_stress_z_T_unit = assemble(
            -dot(u_hat_z, rhs_T_unit) * dx(degree=6))

        fluid_stress_x = fluid_stress_x_no_T + T_adj * fluid_stress_x_T_unit
        fluid_stress_z = fluid_stress_z_no_T + T_adj * fluid_stress_z_T_unit

        if isinstance(self.Re_p, (int, float)):
            inv_Re_p = 1.0 / self.Re_p
            F_p_x = (inv_Re_p * F_s_x
                      + fluid_stress_x + inertial_x + centrifugal_x)
            F_p_z = (inv_Re_p * F_s_z
                      + fluid_stress_z + inertial_z + centrifugal_z)
        else:

            _Re_p = r_scalar(self.Re_p)
            F_p_x = (F_s_x / _Re_p
                      + fluid_stress_x + inertial_x + centrifugal_x)
            F_p_z = (F_s_z / _Re_p
                      + fluid_stress_z + inertial_z + centrifugal_z)
        if not return_components:
            return F_p_x, F_p_z

        components = {
            "F_s_x": F_s_x, "F_s_z": F_s_z,
            "inertial_x": inertial_x, "inertial_z": inertial_z,
            "centrifugal_x": centrifugal_x, "centrifugal_z": centrifugal_z,
            "fluid_stress_x_no_T": fluid_stress_x_no_T,
            "fluid_stress_z_no_T": fluid_stress_z_no_T,
            "fluid_stress_x_T_unit": fluid_stress_x_T_unit,
            "fluid_stress_z_T_unit": fluid_stress_z_T_unit,
            "fluid_stress_x": fluid_stress_x,
            "fluid_stress_z": fluid_stress_z,
            "T_adj": T_adj,
        }
        return F_p_x, F_p_z, components


if __name__ == "__main__":

    from background_flow import build_3d_background_flow
    from perturbed_flow import perturbed_flow

    R_hat = 500.0
    H_hat = 2.0
    W_hat = 2.0
    a_hat = 0.05
    Re_p = 1.0

    L_hat = 4.0 * max(H_hat, W_hat)
    particle_maxh_rel = 0.2
    global_maxh_rel = 0.2
    results = []

    with stop_annotating():
        bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re_p)
        G_val, U_m_hat, u_2d, p_2d = bg.solve_2D_background_flow()
        mesh3d, tags = make_curved_channel_section_with_spherical_hole(R_hat, H_hat, W_hat, L_hat, a_hat,
                                    particle_maxh_rel * a_hat, global_maxh_rel * min(H_hat, W_hat), 0.0, 0.0)

    V_def   = VectorFunctionSpace(mesh3d, "CG", 1)
    R_space = FunctionSpace(mesh3d, "R", 0)

    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

        cx, cy, cz = tags["particle_center"]
        x = SpatialCoordinate(mesh3d)
        dist = sqrt((x[0]-cx)**2 + (x[1]-cy)**2 + (x[2]-cz)**2)
        bump_fn = Function(FunctionSpace(mesh3d, "CG", 1), name="bump")
        bump_fn.interpolate(max_value(Constant(0.0),
            1.0 - max_value(Constant(0.0), dist - Constant(a_hat))
                / (Constant(0.5*min(H_hat,W_hat)) - Constant(a_hat))))
        d_hat_fn = Function(V_def, name="d_hat")
        d_hat_fn.interpolate(as_vector([(x[i]-[cx,cy,cz][i])/dist for i in range(3)]))
        d_hat_data = d_hat_fn.dat.data_ro.copy()

    cos_th = math.cos(tags["theta"] / 2.0)
    sin_th = math.sin(tags["theta"] / 2.0)
    H_MAG  = 10.0
    ctrl_names  = ["delta_r", "delta_z", "delta_a"]
    force_names = ["F_p_x",   "F_p_z"]

    def make_m0():
        return [Function(R_space).assign(0.0) for _ in range(3)]

    def make_h(idx):
        return [Function(R_space).assign(H_MAG if k == idx else 0.0) for k in range(3)]

    def make_h_mixed():
        return [Function(R_space).assign(H_MAG) for _ in range(3)]

    def check(label, ok, info=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  '+info) if info else ''}")
        return ok

    def run_taylor(label, build_fn, h_list):
        print(f"\n  Taylor test: {label}")
        Jhat, m0 = build_fn()
        try:
            info = taylor_to_dict(Jhat, m0, h_list)
        except ZeroDivisionError:
            print(f"    J constant in this direction (all residuals=0)")
            print(f"    [PASS] R1 gradient  (trivially 0)")
            print(f"    [PASS] R2 hessian   (trivially 0)")
            stop_annotating()
            get_working_tape().clear_tape()
            return True, True
        for key in ["R0", "R1", "R2"]:
            print(f"    {key} rates: {[f'{r:.4f}' for r in info[key]['Rate']]}")
            print(f"    {key} resid: {[f'{r:.4e}' for r in info[key]['Residual']]}")
        r0_max = max(info["R0"]["Residual"])
        r1_min = min(info["R1"]["Rate"])
        r1_max = max(info["R1"]["Residual"])
        r2_min = min(info["R2"]["Rate"])
        r1_exact = r0_max > 1e-30 and r1_max < 1e-3 * r0_max
        ok_r1   = r1_min >= 1.9 or r1_exact
        ok_r2   = r2_min >= 2.9
        r1_note = " (exact)" if r1_exact else "  (need>=1.9)"
        print(f"    [{'PASS' if ok_r1 else 'FAIL'}] R1 gradient  min={r1_min:.4f}{r1_note}")
        print(f"    [{'PASS' if ok_r2 else 'FAIL'}] R2 Taylor    min={r2_min:.4f}  (need>=2.9)")
        stop_annotating()
        get_working_tape().clear_tape()

        # ── FD epsilon-sweep cross-check of Hessian ──
        # If AD is correct: |AD-FD| first falls as O(ε²) (trunc. error),
        # then rises as O(ε_mach/ε) (cancellation) → V-shape.
        # If AD is wrong: |AD-FD| plateaus at the bug magnitude.
        import numpy as _np
        n_ctrl = len(h_list)
        m_arr = _np.array([float(h_list[j].dat.data_ro[0]) for j in range(n_ctrl)])

        # AD Hessian·m at base point
        Jhat_h, _ = build_fn()
        Jhat_h.derivative()
        m_fns = [Function(R_space).assign(float(m_arr[j])) for j in range(n_ctrl)]
        Hm_ad_raw = Jhat_h.hessian(m_fns)
        Hm_ad = _np.array([float(Hm_ad_raw[j].dat.data_ro[0]) for j in range(n_ctrl)])
        stop_annotating(); get_working_tape().clear_tape()

        print(f"    --- FD Hessian epsilon-sweep ---")
        print(f"    H*m (AD):  [{', '.join(f'{v:+.6e}' for v in Hm_ad)}]")
        print(f"    {'eps':>10s}  {'max|AD-FD|/scale':>16s}  {'H*m (FD)':>50s}")

        best_rel = float('inf')
        for eps_fd in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
            Jhat_p, _ = build_fn()
            ctrl_p = [Function(R_space).assign(eps_fd * m_arr[j]) for j in range(n_ctrl)]
            Jhat_p(ctrl_p)
            dJ_p_raw = Jhat_p.derivative()
            grad_p = _np.array([float(dJ_p_raw[j].dat.data_ro[0]) for j in range(n_ctrl)])
            stop_annotating(); get_working_tape().clear_tape()

            Jhat_m, _ = build_fn()
            ctrl_m = [Function(R_space).assign(-eps_fd * m_arr[j]) for j in range(n_ctrl)]
            Jhat_m(ctrl_m)
            dJ_m_raw = Jhat_m.derivative()
            grad_m = _np.array([float(dJ_m_raw[j].dat.data_ro[0]) for j in range(n_ctrl)])
            stop_annotating(); get_working_tape().clear_tape()

            Hm_fd = (grad_p - grad_m) / (2 * eps_fd)
            scale = max(_np.max(_np.abs(Hm_fd)), _np.max(_np.abs(Hm_ad)), 1e-30)
            rel_diff = _np.max(_np.abs(Hm_ad - Hm_fd)) / scale
            best_rel = min(best_rel, rel_diff)
            fd_str = ', '.join(f'{v:+.6e}' for v in Hm_fd)
            print(f"    {eps_fd:10.0e}  {rel_diff:16.4e}  [{fd_str}]")

        ok_fd = best_rel < 1e-2
        print(f"    best |AD-FD|/scale = {best_rel:.4e}")
        print(f"    [{'PASS' if ok_fd else 'FAIL'}] FD cross-check  (need < 1%)")

        ok_r2_final = ok_r2 or ok_fd
        return ok_r1, ok_r2_final

    def make_build_fp(force_idx):
        # force_idx: 0=F_p_x, 1=F_p_z
        # delta_a enters xi (mesh deformation) AND particle radius a=(a_hat, da)
        def build():
            set_working_tape(Tape())
            continue_annotation()
            with stop_annotating():
                mesh3d.coordinates.assign(X_ref)
            dr = Function(R_space, name="delta_r").assign(0.0)
            dz = Function(R_space, name="delta_z").assign(0.0)
            da = Function(R_space, name="delta_a").assign(0.0)
            xi = build_xi_diff(dr, dz, bump_fn, cos_th, sin_th, V_def,
                               delta_a=da, d_hat_data=d_hat_data)
            mesh3d.coordinates.assign(X_ref + xi)
            u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
                R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
                X_ref=X_ref, xi=xi)
            pf = perturbed_flow_differentiable(
                R_hat, H_hat, W_hat, L_hat, (a_hat, da), Re_p,
                mesh3d, tags, u_bar_3d, p_bar_3d, X_ref, xi, u_cyl_3d)
            forces = pf.F_p()
            J = forces[force_idx]
            return (ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]),
                    make_m0())
        return build

    '''
    print("\n=== GROUP 1: correctness at delta=0 ===")

    with stop_annotating():
        # second-nondim scaling factors
        scale      = 1.0 / a_hat                   # length:   first → second
        u2d_scale  = 1.0 / (U_m_hat * a_hat)       # velocity: first → second
        R_hat2 = R_hat * scale;  H_hat2 = H_hat * scale;  W_hat2 = W_hat * scale
        L_hat2 = L_hat * scale;  a_hat2 = 1.0
        G_val2 = G_val * u2d_scale**2 * a_hat      # = G_val / (U_m_hat² · a_hat)
        Re_p2  = Re_p * U_m_hat * a_hat**2

        # scale 2D mesh and fields
        mesh2d = u_2d.function_space().mesh()
        coords2d_2 = mesh2d.coordinates.copy(deepcopy=True)
        coords2d_2.dat.data[:] *= scale
        mesh2d_2 = Mesh(coords2d_2)
        u_2d_2 = Function(FunctionSpace(mesh2d_2, u_2d.function_space().ufl_element()))
        p_2d_2 = Function(FunctionSpace(mesh2d_2, p_2d.function_space().ufl_element()))
        u_2d_2.dat.data[:] = u_2d.dat.data_ro[:] * u2d_scale
        p_2d_2.dat.data[:] = p_2d.dat.data_ro[:] * u2d_scale**2

        # scale 3D mesh and tags
        coords3d_2 = mesh3d.coordinates.copy(deepcopy=True)
        coords3d_2.dat.data[:] *= scale
        mesh3d_2 = Mesh(coords3d_2)
        tags3d_2  = dict(tags)
        tags3d_2["particle_center"] = tuple(v * scale for v in tags["particle_center"])

        # reference: perturbed_flow.py on second-nondim mesh (a_hat2=1, centrifugal correct)
        u_ref_2, p_ref_2 = build_3d_background_flow(
            R_hat2, H_hat2, W_hat2, G_val2, mesh3d_2, tags3d_2, u_2d_2, p_2d_2)
        pf_ref = perturbed_flow(R_hat2, H_hat2, W_hat2, L_hat2, a_hat2, Re_p2,
                                mesh3d_2, tags3d_2, u_ref_2, p_ref_2)
        F_ref_x, F_ref_z = pf_ref.F_p()

        # differentiable impl on same second-nondim mesh
        V_def_2  = VectorFunctionSpace(mesh3d_2, "CG", 1)
        X_ref_2  = Function(V_def_2)
        X_ref_2.interpolate(SpatialCoordinate(mesh3d_2))
        xi_zero_2 = Function(V_def_2).assign(0.0)
        u_bar_d2, p_bar_d2, u_cyl_d2 = build_3d_background_flow_differentiable(
            R_hat2, H_hat2, W_hat2, G_val2, mesh3d_2, tags3d_2, u_2d_2, p_2d_2,
            X_ref=X_ref_2, xi=xi_zero_2)
        pf_new = perturbed_flow_differentiable(
            R_hat2, H_hat2, W_hat2, L_hat2, a_hat2, Re_p2,
            mesh3d_2, tags3d_2, u_bar_d2, p_bar_d2, X_ref_2, xi_zero_2, u_cyl_d2)
        F_new_x, F_new_z = pf_new.F_p()

    for lbl, f_new, f_ref in [("F_p_x", float(F_new_x), F_ref_x), ("F_p_z", float(F_new_z), F_ref_z)]:

        rel = abs(f_new - f_ref) / max(abs(f_ref), 1e-15)

        ok = check(f"{lbl} new vs ref", rel < 1e-2,f"new={f_new:.6e}  ref={f_ref:.6e}  rel={rel:.2e}")

        results.append((f"G1 {lbl}", ok))
    
    # ── GROUP 0: Hessian bisect ───────────────────────────────────────
    # Strategy: elongate the chain one block at a time and check O(h³) rate.
    # Two variants per stage:
    #   A = no mesh movement (mesh stays at X_ref)
    #   B = with mesh movement (mesh.coordinates = X_ref + xi)
    # If A PASS, B FAIL → shape Hessian of that block is the culprit.
    # Perturbation: pure delta_r, m=(1,0,0), h in [1e-2..5e-4].
    print("\n=== GROUP 0: Hessian bisect (delta_r direction) ===")

    m_b = [1.0, 0.0, 0.0]
    h_b = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]

    def _bisect_build(dr_v, dz_v, da_v, fn_of_pf, move_mesh):
        set_working_tape(Tape())
        continue_annotation()
        with stop_annotating():
            mesh3d.coordinates.assign(X_ref)
        dr = Function(R_space).assign(dr_v)
        dz = Function(R_space).assign(dz_v)
        da = Function(R_space).assign(da_v)
        xi = build_xi_diff(dr, dz, bump_fn, cos_th, sin_th, V_def,
                           delta_a=da, d_hat_data=d_hat_data)
        if move_mesh:
            mesh3d.coordinates.assign(X_ref + xi)
        ub, pb, uc = build_3d_background_flow_differentiable(
            R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
            X_ref=X_ref, xi=xi)
        pf = perturbed_flow_differentiable(
            R_hat, H_hat, W_hat, L_hat, (a_hat, da), Re_p,
            mesh3d, tags, ub, pb, X_ref, xi, uc)
        J = fn_of_pf(pf, dr, dz, da)
        return J, dr, dz, da

    def run_bisect(label, fn_of_pf, move_mesh):
        tag = "mesh" if move_mesh else "no-mesh"
        print(f"\n  [{tag}] {label}")
        J_adj, dr0, dz0, da0 = _bisect_build(0.0, 0.0, 0.0, fn_of_pf, move_mesh)
        J0 = float(J_adj)
        controls = [Control(dr0), Control(dz0), Control(da0)]
        rf = ReducedFunctional(J_adj, controls)
        dJ = rf.derivative()
        grad_m = sum(float(dJ[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        m_fns = [Function(R_space).assign(float(m_b[k])) for k in range(3)]
        try:
            rf.derivative()
            Hm = rf.hessian(m_fns)
            mHm = sum(float(Hm[k].dat.data_ro[0]) * m_b[k] for k in range(3))
            hess_ok = True
        except Exception as e:
            print(f"    Hessian CRASHED: {type(e).__name__}: {e}")
            mHm = 0.0; hess_ok = False
        stop_annotating(); get_working_tape().clear_tape()

        # FD cross-check: mᵀHm_FD = (J(+eps) - 2*J(0) + J(-eps)) / eps²
        eps_fd = 1e-4
        Jp_adj, _, _, _ = _bisect_build(eps_fd*m_b[0], eps_fd*m_b[1], eps_fd*m_b[2], fn_of_pf, move_mesh)
        Jp = float(Jp_adj); stop_annotating(); get_working_tape().clear_tape()
        Jm_adj, _, _, _ = _bisect_build(-eps_fd*m_b[0], -eps_fd*m_b[1], -eps_fd*m_b[2], fn_of_pf, move_mesh)
        Jm = float(Jm_adj); stop_annotating(); get_working_tape().clear_tape()
        mHm_fd = (Jp - 2*J0 + Jm) / eps_fd**2

        print(f"    J0={J0:+.4e}  ∇J·m={grad_m:+.4e}")
        print(f"    mᵀHm (AD) = {mHm:+.4e}")
        print(f"    mᵀHm (FD) = {mHm_fd:+.4e}  (eps={eps_fd:.0e})")
        if abs(mHm_fd) > 1e-30:
            rel = abs(mHm - mHm_fd) / abs(mHm_fd)
            print(f"    |AD-FD|/|FD| = {rel:.4e}  {'OK' if rel < 0.01 else 'MISMATCH'}")
        else:
            print(f"    FD ~ 0, |AD| = {abs(mHm):.4e}")

        # Taylor convergence using both AD and FD Hessian
        print(f"    {'h':>8s}  {'|r2_AD| O(h³)':>14s}  rate   {'|r2_FD| O(h³)':>14s}  rate")
        prev_r2, prev_r2f, h_prev, rates, rates_fd = None, None, None, [], []
        for h in h_b:
            Jh_adj, _, _, _ = _bisect_build(h*m_b[0], h*m_b[1], h*m_b[2], fn_of_pf, move_mesh)
            Jh = float(Jh_adj)
            stop_annotating(); get_working_tape().clear_tape()
            r2 = abs(Jh - J0 - h*grad_m - 0.5*h**2*mHm)
            r2f = abs(Jh - J0 - h*grad_m - 0.5*h**2*mHm_fd)
            s2, s2f = "  —", "  —"
            if prev_r2 and prev_r2 > 0 and h_prev:
                rv = math.log(r2/prev_r2) / math.log(h/h_prev)
                s2 = f"{rv:.2f}"; rates.append(rv)
            if prev_r2f and prev_r2f > 0 and h_prev:
                rvf = math.log(r2f/prev_r2f) / math.log(h/h_prev)
                s2f = f"{rvf:.2f}"; rates_fd.append(rvf)
            print(f"    {h:8.1e}  {r2:14.4e}  {s2:>5s}   {r2f:14.4e}  {s2f:>5s}")
            prev_r2, prev_r2f, h_prev = r2, r2f, h
        err = max(abs(r - 3.0) for r in rates) if rates else float('inf')
        err_fd = max(abs(r - 3.0) for r in rates_fd) if rates_fd else float('inf')
        ok = bool(rates) and err <= 0.15
        ok_fd = bool(rates_fd) and err_fd <= 0.15
        print(f"    -> AD Hessian:  [{'PASS' if ok else 'FAIL'}]  max|rate-3|={err:.2f}")
        print(f"    -> FD Hessian:  [{'PASS' if ok_fd else 'FAIL'}]  max|rate-3|={err_fd:.2f}")
        return ok or ok_fd

    # S1: ||bc_Theta||² — just interpolate(cross(e_z, X_ref+xi)), linear in xi
    #     no-mesh: Hessian from dx only (via AssembleBlock + mesh coords)
    #     with-mesh: same + moved domain
    def fn_S1(pf, dr, dz, da):
        bc = cross_product_bc(pf.xi, pf._e_z_np, pf.X_ref, np.zeros(3),
                              pf.V_xi, pf.V, pf._M_CG1_CG2, name="bc_S1")
        return assemble(inner(bc, bc) * dx(domain=pf.mesh3d))

    # S2: ||v_Theta||² — one Stokes solve (CachedStokesSolveBlock)
    #     no-mesh: A constant, BC linear → Hessian=0 (exact)
    #     with-mesh: A moves with mesh → needs shape Hessian in SolveBlock
    def fn_S2(pf, dr, dz, da):
        bc = cross_product_bc(pf.xi, pf._e_z_np, pf.X_ref, np.zeros(3),
                              pf.V_xi, pf.V, pf._M_CG1_CG2, name="bc_S2")
        v, _ = pf._solve_stokes(bc)
        return assemble(inner(v, v) * dx(domain=pf.mesh3d))

    # S3: fluid_stress_x — full F_p chain, no centrifugal (all Stokes + NumpyLinSolve)
    def fn_S3(pf, dr, dz, da):
        _, _, comps = pf.F_p(return_components=True)
        return comps["fluid_stress_x"]

    # S4: F_p_x — full chain including centrifugal
    def fn_S4(pf, dr, dz, da):
        Fx, _ = pf.F_p()
        return Fx

    # ── S0 (isolated): CrossProductBCBlock only, with ds(particle) for good conditioning ──
    # No perturbed_flow_differentiable, no InterpolateBlocks, no Stokes.
    print("\n  === S0 (isolated): CrossProductBCBlock only ===")
    from background_flow_return_UFL import _build_CG1_to_CG2_map
    with stop_annotating():
        M_test = _build_CG1_to_CG2_map(mesh3d)
    V_CG2_test = VectorFunctionSpace(mesh3d, "CG", 2)
    e_z_np = np.array([0.0, 0.0, 1.0])

    def _isolated_build(dr_v, dz_v, da_v, move_mesh, use_particle_surf=True):
        set_working_tape(Tape())
        continue_annotation()
        with stop_annotating():
            mesh3d.coordinates.assign(X_ref)
        dr = Function(R_space).assign(dr_v)
        dz = Function(R_space).assign(dz_v)
        da = Function(R_space).assign(da_v)
        xi = build_xi_diff(dr, dz, bump_fn, cos_th, sin_th, V_def,
                           delta_a=da, d_hat_data=d_hat_data)
        if move_mesh:
            mesh3d.coordinates.assign(X_ref + xi)
        bc = cross_product_bc(xi, e_z_np, X_ref, np.zeros(3),
                              V_def, V_CG2_test, M_test, name="bc_iso")
        if use_particle_surf:
            J = assemble(inner(bc, bc) * ds(tags["particle"], domain=mesh3d))
        else:
            J = assemble(inner(bc, bc) * dx(domain=mesh3d))
        return J, dr, dz, da

    for mv in [False, True]:
        tag = "mesh" if mv else "no-mesh"
        print(f"\n  [{tag}] S0a: ||cross(e_z, X_ref+xi)||² ds(particle)  [ISOLATED]")
        J_adj, dr0, dz0, da0 = _isolated_build(0.0, 0.0, 0.0, mv)
        J0 = float(J_adj)
        controls = [Control(dr0), Control(dz0), Control(da0)]
        rf = ReducedFunctional(J_adj, controls)
        dJ = rf.derivative()
        grad_m = sum(float(dJ[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        m_fns = [Function(R_space).assign(float(m_b[k])) for k in range(3)]
        try:
            rf.derivative()
            Hm = rf.hessian(m_fns)
            mHm = sum(float(Hm[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        except Exception as e:
            print(f"    Hessian CRASHED: {type(e).__name__}: {e}")
            mHm = 0.0
        stop_annotating(); get_working_tape().clear_tape()

        # FD Hessian
        eps_fd = 1e-4
        Jp_adj, _, _, _ = _isolated_build(eps_fd*m_b[0], eps_fd*m_b[1], eps_fd*m_b[2], mv)
        Jp = float(Jp_adj); stop_annotating(); get_working_tape().clear_tape()
        Jm_adj, _, _, _ = _isolated_build(-eps_fd*m_b[0], -eps_fd*m_b[1], -eps_fd*m_b[2], mv)
        Jm = float(Jm_adj); stop_annotating(); get_working_tape().clear_tape()
        mHm_fd = (Jp - 2*J0 + Jm) / eps_fd**2

        print(f"    J0={J0:+.4e}  ∇J·m={grad_m:+.4e}")
        print(f"    mᵀHm (AD) = {mHm:+.4e}")
        print(f"    mᵀHm (FD) = {mHm_fd:+.4e}")
        if abs(mHm_fd) > 1e-30:
            rel = abs(mHm - mHm_fd) / abs(mHm_fd)
            print(f"    |AD-FD|/|FD| = {rel:.4e}  {'OK' if rel < 0.01 else 'MISMATCH'}")

        # Quick Taylor convergence
        print(f"    {'h':>8s}  {'|r2_AD|':>12s}  rate   {'|r2_FD|':>12s}  rate")
        prev_a, prev_f, h_prev = None, None, None
        for h in h_b:
            Jh_adj, _, _, _ = _isolated_build(h*m_b[0], h*m_b[1], h*m_b[2], mv)
            Jh = float(Jh_adj); stop_annotating(); get_working_tape().clear_tape()
            ra = abs(Jh - J0 - h*grad_m - 0.5*h**2*mHm)
            rf_val = abs(Jh - J0 - h*grad_m - 0.5*h**2*mHm_fd)
            sa = f"{math.log(ra/prev_a)/math.log(h/h_prev):.2f}" if prev_a and prev_a > 0 and h_prev else "  —"
            sf = f"{math.log(rf_val/prev_f)/math.log(h/h_prev):.2f}" if prev_f and prev_f > 0 and h_prev else "  —"
            print(f"    {h:8.1e}  {ra:12.4e}  {sa:>5s}   {rf_val:12.4e}  {sf:>5s}")
            prev_a, prev_f, h_prev = ra, rf_val, h

    # ── S0b: CrossProductBCBlock → CachedStokesSolveBlock (isolated, no InterpolateBlocks) ──
    # Tests the Stokes solve with CrossProductBCBlock BC, without any
    # InterpolateBlocks from perturbed_flow_differentiable on the tape.
    print("\n  --- S0b: CrossProductBCBlock + Stokes (isolated) ---")
    with stop_annotating():
        from perturbed_flow_return_UFL import CachedStokesContext
        stokes_ctx_test = CachedStokesContext(
            VectorFunctionSpace(mesh3d, "CG", 2),
            FunctionSpace(mesh3d, "CG", 1),
            tags, mesh3d)

    def _isolated_stokes_build(dr_v, dz_v, da_v, move_mesh):
        set_working_tape(Tape())
        continue_annotation()
        with stop_annotating():
            mesh3d.coordinates.assign(X_ref)
        dr = Function(R_space).assign(dr_v)
        dz = Function(R_space).assign(dz_v)
        da = Function(R_space).assign(da_v)
        xi = build_xi_diff(dr, dz, bump_fn, cos_th, sin_th, V_def,
                           delta_a=da, d_hat_data=d_hat_data)
        if move_mesh:
            mesh3d.coordinates.assign(X_ref + xi)
        V_s = VectorFunctionSpace(mesh3d, "CG", 2)
        Q_s = FunctionSpace(mesh3d, "CG", 1)
        bc = cross_product_bc(xi, e_z_np, X_ref, np.zeros(3),
                              V_def, V_s, M_test, name="bc_stokes")
        v, _ = stokes_solve_cached(V_s, Q_s, bc, tags, mesh3d, stokes_ctx_test)
        J = assemble(inner(v, v) * dx(domain=mesh3d))
        return J, dr, dz, da

    for mv in [False, True]:
        tag = "mesh" if mv else "no-mesh"
        print(f"\n  [{tag}] S0b: ||v_Stokes||² [CrossProductBC + CachedStokes, ISOLATED]")
        J_adj, dr0, dz0, da0 = _isolated_stokes_build(0.0, 0.0, 0.0, mv)
        J0 = float(J_adj)
        controls = [Control(dr0), Control(dz0), Control(da0)]
        rf = ReducedFunctional(J_adj, controls)
        dJ = rf.derivative()
        grad_m = sum(float(dJ[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        m_fns = [Function(R_space).assign(float(m_b[k])) for k in range(3)]
        try:
            rf.derivative()
            Hm = rf.hessian(m_fns)
            mHm = sum(float(Hm[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        except Exception as e:
            print(f"    Hessian CRASHED: {type(e).__name__}: {e}")
            mHm = 0.0
        stop_annotating(); get_working_tape().clear_tape()

        eps_fd = 1e-4
        Jp_adj, _, _, _ = _isolated_stokes_build(eps_fd*m_b[0], eps_fd*m_b[1], eps_fd*m_b[2], mv)
        Jp = float(Jp_adj); stop_annotating(); get_working_tape().clear_tape()
        Jm_adj, _, _, _ = _isolated_stokes_build(-eps_fd*m_b[0], -eps_fd*m_b[1], -eps_fd*m_b[2], mv)
        Jm = float(Jm_adj); stop_annotating(); get_working_tape().clear_tape()
        mHm_fd = (Jp - 2*J0 + Jm) / eps_fd**2

        print(f"    J0={J0:+.4e}  ∇J·m={grad_m:+.4e}")
        print(f"    mᵀHm (AD) = {mHm:+.4e}")
        print(f"    mᵀHm (FD) = {mHm_fd:+.4e}")
        if abs(mHm_fd) > 1e-30:
            rel = abs(mHm - mHm_fd) / abs(mHm_fd)
            print(f"    |AD-FD|/|FD| = {rel:.4e}  {'OK' if rel < 0.01 else 'MISMATCH'}")

        print(f"    {'h':>8s}  {'|r2_AD|':>12s}  rate   {'|r2_FD|':>12s}  rate")
        prev_a, prev_f, h_prev = None, None, None
        for h in h_b:
            Jh_adj, _, _, _ = _isolated_stokes_build(h*m_b[0], h*m_b[1], h*m_b[2], mv)
            Jh = float(Jh_adj); stop_annotating(); get_working_tape().clear_tape()
            ra = abs(Jh - J0 - h*grad_m - 0.5*h**2*mHm)
            rf_val = abs(Jh - J0 - h*grad_m - 0.5*h**2*mHm_fd)
            sa = f"{math.log(ra/prev_a)/math.log(h/h_prev):.2f}" if prev_a and prev_a > 0 and h_prev else "  —"
            sf = f"{math.log(rf_val/prev_f)/math.log(h/h_prev):.2f}" if prev_f and prev_f > 0 and h_prev else "  —"
            print(f"    {h:8.1e}  {ra:12.4e}  {sa:>5s}   {rf_val:12.4e}  {sf:>5s}")
            prev_a, prev_f, h_prev = ra, rf_val, h

    # ── S0c: standard firedrake.solve (GenericSolveBlock, no CachedStokes) ──
    # Tests whether the Hessian bug is in our CachedStokesSolveBlock
    # or in firedrake's GenericSolveBlock itself.
    print("\n  --- S0c: firedrake.solve (GenericSolveBlock, no cache) ---")

    def _isolated_standard_solve_build(dr_v, dz_v, da_v, move_mesh):
        set_working_tape(Tape())
        continue_annotation()
        with stop_annotating():
            mesh3d.coordinates.assign(X_ref)
        dr = Function(R_space).assign(dr_v)
        dz = Function(R_space).assign(dz_v)
        da = Function(R_space).assign(da_v)
        xi = build_xi_diff(dr, dz, bump_fn, cos_th, sin_th, V_def,
                           delta_a=da, d_hat_data=d_hat_data)
        if move_mesh:
            mesh3d.coordinates.assign(X_ref + xi)
        V_s = VectorFunctionSpace(mesh3d, "CG", 2)
        Q_s = FunctionSpace(mesh3d, "CG", 1)
        W_s = V_s * Q_s
        bc_fn = cross_product_bc(xi, e_z_np, X_ref, np.zeros(3),
                                 V_def, V_s, M_test, name="bc_std")
        v_t, p_t = TrialFunctions(W_s)
        v_te, q_te = TestFunctions(W_s)
        a_s = (2*inner(sym(grad(v_t)), sym(grad(v_te)))*dx
               - p_t*div(v_te)*dx + q_te*div(v_t)*dx)
        L_s = inner(Constant((0,0,0)), v_te)*dx
        bcs_s = [DirichletBC(W_s.sub(0), Constant((0,0,0)), tags["walls"]),
                 DirichletBC(W_s.sub(0), bc_fn, tags["particle"])]
        w_s = Function(W_s)
        ns = MixedVectorSpaceBasis(W_s, [W_s.sub(0),
             VectorSpaceBasis(constant=True, comm=W_s.comm)])
        solve(a_s == L_s, w_s, bcs=bcs_s, nullspace=ns,
              solver_parameters={"ksp_type": "preonly", "pc_type": "lu",
                                 "pc_factor_mat_solver_type": "mumps"})
        v_sol = w_s.subfunctions[0]
        J = assemble(inner(v_sol, v_sol)*dx(domain=mesh3d))
        return J, dr, dz, da

    for mv in [False, True]:
        tag = "mesh" if mv else "no-mesh"
        print(f"\n  [{tag}] S0c: ||v||² [firedrake.solve GenericSolveBlock, ISOLATED]")
        J_adj, dr0, dz0, da0 = _isolated_standard_solve_build(0.0, 0.0, 0.0, mv)
        J0 = float(J_adj)
        controls = [Control(dr0), Control(dz0), Control(da0)]
        rf = ReducedFunctional(J_adj, controls)
        dJ = rf.derivative()
        grad_m = sum(float(dJ[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        m_fns = [Function(R_space).assign(float(m_b[k])) for k in range(3)]
        try:
            rf.derivative()
            Hm = rf.hessian(m_fns)
            mHm = sum(float(Hm[k].dat.data_ro[0]) * m_b[k] for k in range(3))
        except Exception as e:
            print(f"    Hessian CRASHED: {type(e).__name__}: {e}")
            mHm = 0.0
        stop_annotating(); get_working_tape().clear_tape()

        eps_fd = 1e-4
        Jp_adj, _, _, _ = _isolated_standard_solve_build(eps_fd*m_b[0], eps_fd*m_b[1], eps_fd*m_b[2], mv)
        Jp = float(Jp_adj); stop_annotating(); get_working_tape().clear_tape()
        Jm_adj, _, _, _ = _isolated_standard_solve_build(-eps_fd*m_b[0], -eps_fd*m_b[1], -eps_fd*m_b[2], mv)
        Jm = float(Jm_adj); stop_annotating(); get_working_tape().clear_tape()
        mHm_fd = (Jp - 2*J0 + Jm) / eps_fd**2

        print(f"    J0={J0:+.4e}  ∇J·m={grad_m:+.4e}")
        print(f"    mᵀHm (AD) = {mHm:+.4e}")
        print(f"    mᵀHm (FD) = {mHm_fd:+.4e}")
        if abs(mHm_fd) > 1e-30:
            rel = abs(mHm - mHm_fd) / abs(mHm_fd)
            print(f"    |AD-FD|/|FD| = {rel:.4e}  {'OK' if rel < 0.01 else 'MISMATCH'}")

    print("\n  === End S0 ===\n")
    
    for fn, lbl in [(fn_S1, "S1: ||cross(e_z, X_ref+xi)||²  [CrossProductBCBlock]"),
                    (fn_S2, "S2: ||v_Theta||²  [CachedStokesSolveBlock]"),
                    (fn_S3, "S3: fluid_stress_x  [all Stokes + NumpyLinSolve]"),
                    (fn_S4, "S4: F_p_x  [full chain]")]:
        for mv in [False, True]:
            run_bisect(lbl, fn, move_mesh=mv)
    '''

    print("\n=== Test derivatives of F_p_x and F_p_z ===")

    h_dirs = [
        ("delta_r",           [1, 0, 0]),
        ("delta_z",           [0, 1, 0]),
        ("delta_a",           [0, 0, 1]),
        ("delta_r delta_z",   [1, 1, 0]),
        ("delta_r delta_a",   [1, 0, 1]),
        ("delta_z delta_a",   [0, 1, 1]),
        ("delta_r delta_z delta_a", [1, 1, 1]),
    ]

    for fi, fn in enumerate(force_names):
        for hlbl, hflags in h_dirs:
            h = [Function(R_space).assign(H_MAG * f) for f in hflags]
            ok1, ok2 = run_taylor(f"{fn} / {hlbl}", make_build_fp(fi), h)
            results += [(f"G2 {fn} R1 {hlbl}", ok1), (f"G3 {fn} R2 {hlbl}", ok2)]


    print(f"\n{'='*50}")
    n_pass = sum(ok for _, ok in results)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  {n_pass}/{len(results)} tests passed")
    import sys; sys.exit(0 if n_pass == len(results) else 1)