import os
os.environ["OMP_NUM_THREADS"] = "1"

import math

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, taylor_test
from pyadjoint import Block, AdjFloat, get_working_tape, ReducedFunctional, Control, Tape, set_working_tape, continue_annotation, taylor_to_dict
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock, solve_init_params

from nondimensionalization import second_nondimensionalisation
from background_flow_return_UFL import (
    background_flow_differentiable, build_3d_background_flow_differentiable,
    build_xi_diff,
)
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
import ufl
from firedrake.adjoint_utils.blocks.assembly import AssembleBlock
from firedrake.adjoint_utils.blocks.block_utils import isconstant
from ufl.domain import as_domain

def _patched_evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs,
        block_variable, idx, relevant_dependencies, prepared=None):

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

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        """Hessian for the linear system  A x = b.

        x = A^{-1} b  is a rational function of the entries of A and b.
        The adjoint (first derivative) is:
            mu = A^{-T} adj_x
            d_adj[A_{ij}] = -mu_i x_j   (for entry (i,j) of A)
            d_adj[b_k]    = mu_k

        The Hessian cross-term comes from the fact that mu and x both
        depend on the inputs.  The TLM gives dx = A^{-1}(db - dA x),
        and d(mu) = A^{-T}(d(adj_x) - dA^T mu) where d(adj_x) is the
        second-order seed (hessian_inputs).

        For the output x_idx (one scalar), the dependencies are ALL
        A_{ij} and b_k entries (idx 0..n*n+n-1).  The Hessian component
        for dependency `idx` sums:
          (1) linear adjoint of the second-order seed (same as adj)
          (2) cross-term involving the TLM direction of the other deps
        """
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
                if self.mode == "azim":
                    val = np.sum(-u[:, 2] * a[:, 1])
                else:
                    val = np.sum(-u[:, 0] * a[:, 0])
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
                        cross_data[:, 1] += -s_tlm.dat.data_ro * a[:, 1]  # note: not a[:,0]!

                if not np.any(cross_data):
                    return part1
                if part1 is None:
                    cross = Cofunction(self.V_out.dual())
                    cross.dat.data[:] = cross_data
                    return cross
                part1.dat.data[:] += cross_data
                return part1


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


class V0aCombineBlock(Block):
    """Tape-aware computation of
        v_0_a = T*v_T + Ox*v_Ox + Oy*v_Oy + Oz*v_Oz + v_bg
    where (T, Ox, Oy, Oz) are R-space Function coefficients and
    (v_T, v_Ox, v_Oy, v_Oz, v_bg) are CG2 vector Function coefficients.

    Replaces the equivalent ``v_0_a.interpolate(T*v_T + ... + v_bg)``
    which mixes R-space and CG2 coefficients in a single interpolate.
    Such mixed-coefficient interpolates are exactly the case that the
    Hessian path of pyadjoint cannot handle: when a downstream form
    involving v_0_a is differentiated twice, the InterpolateBlock's
    Hessian pass triggers a UFL ``replace`` failure inside firedrake's
    tsfc-interface (the same root cause as ``Theta_fn**2 * dx``).

    The map ``(T, Ox, Oy, Oz, v_T, v_Ox, v_Oy, v_Oz, v_bg) -> v_0_a`` is
    *multilinear*: linear in each argument, so the partial second
    derivatives ``∂²/∂(arg)²`` vanish, but the cross-partials between an
    R-space scalar (idx i in {0..3}) and the matching CG2 field
    (idx i+4) are non-zero -- they are responsible for the chain
        delta_r/z/a -> R-space scalar -> v_0_a -> form
    being correct under Hessian-vector products.

    Index layout (matches add_dependency calls):
        idx 0..3 : T, Ox, Oy, Oz   (R-space Functions)
        idx 4..7 : v_T, v_Ox, v_Oy, v_Oz   (CG2 vector Functions)
        idx 8    : v_bg            (CG2 vector Function, coefficient = 1)
    """

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

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
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

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        T, Ox, Oy, Oz = self._scalars_from(inputs)
        v_T, v_Ox, v_Oy, v_Oz, v_bg = (inputs[4], inputs[5],
                                        inputs[6], inputs[7], inputs[8])
        scalars = (T, Ox, Oy, Oz)
        v_i_list = (v_T, v_Ox, v_Oy, v_Oz)
        with stop_annotating():
            data = np.zeros_like(v_bg.dat.data_ro, dtype=float)
            # Linear contributions from R-space TLM seeds
            for i in range(4):
                if tlm_inputs[i] is not None:
                    data += float(tlm_inputs[i].dat.data_ro[0]) \
                            * v_i_list[i].dat.data_ro
            # Linear contributions from CG2 TLM seeds
            for i in range(4):
                if tlm_inputs[4 + i] is not None:
                    data += scalars[i] * tlm_inputs[4 + i].dat.data_ro
            if tlm_inputs[8] is not None:
                data += tlm_inputs[8].dat.data_ro

            out = Function(self.V_out)
            out.dat.data[:] = data
        return out

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # The map is multilinear (linear in each argument).  Pure-block
        # second derivatives ∂²/∂(arg)² vanish, so only the linear
        # adjoint of the second-order seed plus the cross-partial term
        # contribute.
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


def combine_v_0_a(T_fn, Ox_fn, Oy_fn, Oz_fn,
                  v_T, v_Ox, v_Oy, v_Oz, v_bg, V):
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


def stokes_solve(V, Q, particle_bc_expr, tags, mesh):

    W = V * Q
    v_trial, p_trial = TrialFunctions(W)
    v_test, q_test = TestFunctions(W)
    a_form = (
        2 * inner(sym(grad(v_trial)), sym(grad(v_test))) * dx
        - p_trial * div(v_test) * dx
        + q_test * div(v_trial) * dx
    )
    L_form = inner(Constant((0.0, 0.0, 0.0)), v_test) * dx
    bcs = [
        DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), tags["walls"]),
        DirichletBC(W.sub(0), particle_bc_expr, tags["particle"]),
    ]
    nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True, comm=W.comm)]
    )
    w = Function(W)
    solve(
        a_form == L_form, w, bcs=bcs,
        nullspace=nullspace,
        solver_parameters={
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1, "mat_mumps_icntl_25": 0,
        },
    )
    v_out = Function(V, name="v_stokes")
    p_out = Function(Q, name="p_stokes")
    v_out.assign(w.subfunctions[0])
    p_out.assign(w.subfunctions[1])
    return v_out, p_out


_STOKES_SP = {
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_24": 1, "mat_mumps_icntl_25": 0,
}


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
            self._fwd_solver = LinearSolver(
                A, nullspace=self.nullspace, solver_parameters=_STOKES_SP)
            self._adj_solver = None
            self._fp = fp
            self.fwd_factor_count += 1
        return self._fwd_solver

    def get_adj_solver(self):
        self.get_fwd_solver()
        if self._adj_solver is None:
            a_adj = adjoint(self.a_form)
            B = assemble(a_adj, bcs=self.bcs_hom)
            self._adj_solver = LinearSolver(
                B, nullspace=self.nullspace, solver_parameters=_STOKES_SP)
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
        block = CachedStokesSolveBlock(
            a_form, L_form, w, bcs, ctx,
            solver_parameters=_STOKES_SP,
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

    def __init__(self, R, H, W, L, a, Re_p, mesh3d, tags, u_bar_3d,
                 p_bar_3d, X_ref, xi, u_cyl_3d):
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
        return [assemble(traction[i] * ds(self.tags["particle"]))
                for i in range(3)]

    def _torque_components(self, v, q):
        n = FacetNormal(self.mesh3d)
        sigma = -q * Identity(3) + (grad(v) + grad(v).T)
        traction = dot(-n, sigma)
        moment = cross(self.x - self.x_p, traction)
        return [assemble(moment[i] * ds(self.tags["particle"]))
                for i in range(3)]

    @staticmethod
    def _dot3(e_np, comps):
        return (float(e_np[0]) * comps[0] + float(e_np[1]) * comps[1]
                + float(e_np[2]) * comps[2])

    def F_p(self, return_components=False):
        """Compute the dimensionless cross-sectional particle force (F_p_x, F_p_z).

        If ``return_components`` is True, additionally returns a dict of
        all AdjFloat sub-terms so they can be individually tested for AD
        correctness (gradient, Hessian) via ReducedFunctional.

        This method is generic in the choice of non-dimensionalisation:

          - hat_hat coordinates (paper's convention, particle radius = 1):
            pass `Re_p` as the 6th constructor argument. The result lives in
            the paper's force scale ρU_m²a⁴/ℓ² (eq 2.28a in Harding 2019).

          - hat coordinates (user's first non-dim, particle radius = a/L_c):
            pass the *duct* Reynolds number `Re` as the 6th constructor
            argument (NOT Re_p). The result lives in the Reynolds-scale
            ρU_c²L_c² which is the natural force scale in hat units.

        Why both work: the (1/self.Re_p) prefactor on the F_-1_s contribution
        is what converts the Stokes-scale stress integral (μUL) into the
        Reynolds-scale force (ρU²L²) used by centrifugal/inertial terms.
        That conversion factor is the local duct Reynolds number based on
        whichever non-dim units the mesh is in: Re_p in hat_hat (where
        Re_p = ρU_c_p·L_c_p/μ by construction), Re in hat. The centrifugal
        term carries an explicit `self.a**3` factor, so the unit-sphere
        assumption of the paper is *not* hard-coded — passing self.a = a_hat
        gives the right sphere volume in either system.

        See TEST 7 in __main__ for an empirical verification: it computes
        F_p in hat coordinates and confirms F̂_hat = F̂_hh · U_m_hat² · a_hat⁴
        (the F-scale ratio derived from the centrifugal term).
        """
        V = self.V
        x_bc = self.x_bc

        u_bar_a = dot(self.u_bar, self.e_theta_prime) * self.e_theta_prime
        u_bar_s = (dot(self.u_bar, self.e_r_prime) * self.e_r_prime
                   + dot(self.u_bar, self.e_z_prime) * self.e_z_prime)

        u_bar_a_bc = dot(self.u_bar_bc, self.e_theta_bc) * self.e_theta_bc
        u_bar_s_bc = (dot(self.u_bar_bc, self.e_r_bc) * self.e_r_bc
                      + dot(self.u_bar_bc, self.e_z_prime) * self.e_z_prime)

        bc_Theta = Function(V, name="bc_Theta")
        bc_Theta.interpolate(cross(self.e_z_prime, x_bc))
        bc_Ox = Function(V, name="bc_Ox")
        bc_Ox.interpolate(cross(self.e_x_prime, x_bc - self.x_p))
        bc_Oy = Function(V, name="bc_Oy")
        bc_Oy.interpolate(cross(self.e_y_prime, x_bc - self.x_p))
        bc_Oz = Function(V, name="bc_Oz")
        bc_Oz.interpolate(cross(self.e_z_prime, x_bc - self.x_p))
        # bc_bg = -u_bar_a_bc = -u_theta * e_theta via CylProjectBlock.
        # Mathematically:  bc_bg[:, 0] = -u_cyl[:, 2] * sin_phi
        #                  bc_bg[:, 1] =  u_cyl[:, 2] * cos_phi
        # This avoids the product-of-two-tape-Functions in an Interpolate
        # that crashes pyadjoint's Hessian (Layer 2b-v in diagnostic).
        bc_bg = cyl_project(self.cos_phi_bc, self.sin_phi_bc,
                            self.u_cyl_3d, V, mode="azim")

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

        # Split the fluid_stress rhs into a Theta_fn-free piece and a
        # piece that is multiplied by Theta_fn, so that NEITHER ufl form
        # has Theta_fn (an R-space coefficient) as a top-level coefficient.
        # The Theta_fn factor is reapplied via AdjFloat arithmetic
        # (T_adj * fluid_stress_*_T_unit) below.  Combined with v_0_a
        # being built via V0aCombineBlock (which carries no R-space
        # coefficient through the interpolate path), this means
        #     pyadjoint never differentiates a single ufl form with an
        #     R-space coefficient appearing more than linearly,
        # which is the precondition for the Hessian path through F_p to
        # not hit the
        #     ValueError: Derivatives should be applied before executing
        #                 replace.
        # bug inside firedrake's tsfc-interface.
        #
        # Original rhs (with Theta_fn factored out where it appears
        # explicitly):
        #     rhs = rhs_no_T  +  Theta_fn * rhs_T_unit
        # where
        #     rhs_no_T  contains every term that does NOT have Theta_fn
        #               as a direct factor.  v_0_a in these terms still
        #               carries an indirect Theta_fn dependency through
        #               the V0aCombineBlock chain, but that lives off the
        #               form and is differentiated via the custom block.
        #     rhs_T_unit = cross(e_z, v_0_a + v_0_s)
        #                 - dot(grad(v_0_a + v_0_s), cross(e_z, x))
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
        rhs_T_unit = (
            cross(self.e_z_prime, v_0_a)
            + cross(self.e_z_prime, v_0_s)
            - dot(grad(v_0_a), cross(self.e_z_prime, self.x))
            - dot(grad(v_0_s), cross(self.e_z_prime, self.x))
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
            # Re_p is a Function(R_space) on the tape — extract via the
            # RScalarBlock path so its dependency on the controls flows
            # cleanly through to dF/d(delta_a).  The previous
            # ``assemble(Re_p * dx) / vol`` shortcut had the same form-
            # induced issues that motivated RScalarBlock in the first
            # place.
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


def _banner(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _pass_fail(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return passed


if __name__ == "__main__":

    results = []

    R_hat = 500; H_hat = 2; W_hat = 2; a_hat = 0.135
    Re = 1.0; L_c = H_hat / 2; U_c = 0.008366733466944444

    set_working_tape(Tape())
    continue_annotation()

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_val, U_m_hat, u_bar, p_bar_tilde = bg.solve_2D_background_flow()

    R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, U_c_p, u_2d_hh, p_2d_hh, Re_p = \
        second_nondimensionalisation(
            R_hat, H_hat, W_hat, a_hat, L_c, U_c,
            G_val, Re, u_bar, p_bar_tilde, U_m_hat, print_values=False)

    L_hh = 4 * max(H_hh, W_hh)
    particle_maxh = 0.2 * a_hh
    global_maxh = 0.2 * min(H_hh, W_hh)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R_hh, H_hh, W_hh, L_hh, a_hh,
        particle_maxh, global_maxh, r_off=-4, z_off=2)

    R_space = FunctionSpace(mesh3d, "R", 0)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)
    V_scalar = FunctionSpace(mesh3d, "CG", 1)
    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

        cx, cy, cz = tags["particle_center"]
        x_ufl = SpatialCoordinate(mesh3d)
        dist = sqrt((x_ufl[0] - cx)**2 + (x_ufl[1] - cy)**2 + (x_ufl[2] - cz)**2)

        a_c = Constant(a_hh)
        r_cut = Constant(0.5 * min(H_hh, W_hh))
        bump_expr = max_value(Constant(0.0),
                              1.0 - max_value(Constant(0.0), dist - a_c)
                                  / (r_cut - a_c))

        bump = Function(V_scalar, name="bump")
        bump.interpolate(bump_expr)

        d_hat_fn = Function(V_def, name="d_hat")
        d_hat_fn.interpolate(as_vector([
            (x_ufl[0] - cx) / dist,
            (x_ufl[1] - cy) / dist,
            (x_ufl[2] - cz) / dist,
        ]))
        d_hat_data = d_hat_fn.dat.data_ro.copy()

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    from perturbed_flow import perturbed_flow

    ctrl_names = ["delta_r", "delta_z", "delta_a"]
    n_ctrl = 3

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _build_pf_at(dr_v, dz_v, da_v, component):
        """Fresh tape at (dr, dz, da), return ReducedFunctional for F_p_x or F_p_z."""
        set_working_tape(Tape())
        continue_annotation()
        dr_fn = Function(R_space, name="delta_r").assign(dr_v)
        dz_fn = Function(R_space, name="delta_z").assign(dz_v)
        da_fn = Function(R_space, name="delta_a").assign(da_v)
        xi_l = build_xi_diff(dr_fn, dz_fn, bump, cos_th, sin_th, V_def,
                             delta_a=da_fn, d_hat_data=d_hat_data)
        mesh3d.coordinates.assign(X_ref + xi_l)
        u_l, p_l, uc_l = build_3d_background_flow_differentiable(
            R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh,
            X_ref=X_ref, xi=xi_l)
        pf_l = perturbed_flow_differentiable(
            R_hh, H_hh, W_hh, L_hh, (a_hh, da_fn), Re_p,
            mesh3d, tags, u_l, p_l, X_ref, xi_l, uc_l)
        Fxl, Fzl = pf_l.F_p()
        F = Fxl if component == "x" else Fzl
        controls = [Control(dr_fn), Control(dz_fn), Control(da_fn)]
        Jhat = ReducedFunctional(F, controls)
        m0 = [Function(R_space).assign(dr_v),
              Function(R_space).assign(dz_v),
              Function(R_space).assign(da_v)]
        return Jhat, m0, controls

    def _eval_Fp(dr_v, dz_v, da_v):
        """Evaluate F_p_x, F_p_z at (dr, dz, da) via fresh forward solve (no tape)."""
        with stop_annotating():
            set_working_tape(Tape())
            dr_fn = Function(R_space).assign(dr_v)
            dz_fn = Function(R_space).assign(dz_v)
            da_fn = Function(R_space).assign(da_v)
            xi_l = build_xi_diff(dr_fn, dz_fn, bump, cos_th, sin_th, V_def,
                                 delta_a=da_fn, d_hat_data=d_hat_data)
            mesh3d.coordinates.assign(X_ref + xi_l)
            u_l, p_l, uc_l = build_3d_background_flow_differentiable(
                R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh,
                X_ref=X_ref, xi=xi_l)
            pf_l = perturbed_flow_differentiable(
                R_hh, H_hh, W_hh, L_hh, (a_hh, da_fn), Re_p,
                mesh3d, tags, u_l, p_l, X_ref, xi_l, uc_l)
            Fxl, Fzl = pf_l.F_p()
        return float(Fxl), float(Fzl)

    def _eval_Fp_ref(dr_v, dz_v):
        """Evaluate F_p via perturbed_flow.py at (dr, dz, da=0)."""
        with stop_annotating():
            set_working_tape(Tape())
            dr_fn = Function(R_space).assign(dr_v)
            dz_fn = Function(R_space).assign(dz_v)
            da_fn = Function(R_space).assign(0.0)
            xi_l = build_xi_diff(dr_fn, dz_fn, bump, cos_th, sin_th, V_def,
                                 delta_a=da_fn, d_hat_data=d_hat_data)
            mesh3d.coordinates.assign(X_ref + xi_l)
            u_l, p_l, _ = build_3d_background_flow_differentiable(
                R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh,
                X_ref=X_ref, xi=xi_l)
            pf_ref = perturbed_flow(
                R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
                mesh3d, tags, u_l, p_l)
            Fx, Fz = pf_ref.F_p()
        return float(Fx), float(Fz)

    def _grad_ad(dr_v, dz_v, da_v, component):
        """AD gradient at (dr, dz, da) for component 'x' or 'z'."""
        Jh, _, _ = _build_pf_at(dr_v, dz_v, da_v, component)
        d = Jh.derivative()
        return np.array([float(d[k].dat.data_ro[0]) for k in range(n_ctrl)])

    # =====================================================================
    #  TEST A: Forward F_p at multiple points vs perturbed_flow.py
    # =====================================================================

    _banner("TEST A: FORWARD F_p vs perturbed_flow.py")

    # Compare differentiable F_p against perturbed_flow.py at several
    # (delta_r, delta_z) points (delta_a = 0 so centrifugal terms match,
    # since perturbed_flow.py has no a^3 factor — it assumes a_hh = 1).
    test_points_A = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.0, 0.3),
        (0.5, 0.3),
        (-0.3, 0.2),
    ]
    tol_A = 1e-10
    testA_pass = True
    for dr_v, dz_v in test_points_A:
        fp_diff = _eval_Fp(dr_v, dz_v, 0.0)
        fp_ref = _eval_Fp_ref(dr_v, dz_v)
        for comp, idx in [("x", 0), ("z", 1)]:
            if abs(fp_ref[idx]) > 1e-30:
                rel = abs(fp_diff[idx] - fp_ref[idx]) / abs(fp_ref[idx])
            else:
                rel = abs(fp_diff[idx] - fp_ref[idx])
            ok = rel < tol_A
            testA_pass &= _pass_fail(
                f"F_{comp} at (r={dr_v}, z={dz_v})", ok,
                f"diff={fp_diff[idx]:+.10e} ref={fp_ref[idx]:+.10e} rel={rel:.2e}")
    results.append(("Test A: Forward F_p vs perturbed_flow.py", testA_pass))
    # =====================================================================
    #  TEST B: First derivatives — AD vs centred FD + Taylor R1
    # =====================================================================

    _banner("TEST B: FIRST DERIVATIVES (AD vs FD + Taylor R1)")

    def _print_matrix(label, M, indent="    "):
        print(f"\n{indent}{label}:")
        header = indent + " " * 10 + "  ".join(f"{c:>14}" for c in ctrl_names)
        print(header)
        for i, name in enumerate(ctrl_names):
            row = "  ".join(f"{M[i, j]:+14.6e}" for j in range(n_ctrl))
            print(f"{indent}{name:>9}  {row}")

    base_B = (0.0, 0.0, 0.0)
    eps_fd_B = 1e-5
    h_taylor_B = [0.05, 0.05, 0.005]

    # (a) Centred FD gradient (each eval = fresh mesh at perturbed point)
    print("  (a) Centred FD gradient (eps={:.0e})".format(eps_fd_B))
    grad_FD_x = np.zeros(n_ctrl)
    grad_FD_z = np.zeros(n_ctrl)
    for k in range(n_ctrl):
        bp = list(base_B); bp[k] += eps_fd_B
        bm = list(base_B); bm[k] -= eps_fd_B
        fp_p = _eval_Fp(*bp)
        fp_m = _eval_Fp(*bm)
        grad_FD_x[k] = (fp_p[0] - fp_m[0]) / (2 * eps_fd_B)
        grad_FD_z[k] = (fp_p[1] - fp_m[1]) / (2 * eps_fd_B)

    # (b) AD gradient
    print("  (b) AD gradient via ReducedFunctional.derivative()")
    grad_AD_x = _grad_ad(*base_B, "x")
    grad_AD_z = _grad_ad(*base_B, "z")

    print("  AD:  dF_x/d(r,z,a) = [{:+.10e}, {:+.10e}, {:+.10e}]".format(*grad_AD_x))
    print("  FD:  dF_x/d(r,z,a) = [{:+.10e}, {:+.10e}, {:+.10e}]".format(*grad_FD_x))
    print("  AD:  dF_z/d(r,z,a) = [{:+.10e}, {:+.10e}, {:+.10e}]".format(*grad_AD_z))
    print("  FD:  dF_z/d(r,z,a) = [{:+.10e}, {:+.10e}, {:+.10e}]".format(*grad_FD_z))

    tol_fd_B = 1e-4
    testB_pass = True
    for comp_label, g_ad, g_fd in [("F_x", grad_AD_x, grad_FD_x),
                                    ("F_z", grad_AD_z, grad_FD_z)]:
        for k, cn in enumerate(ctrl_names):
            denom = max(abs(g_fd[k]), 1e-30)
            rel = abs(g_ad[k] - g_fd[k]) / denom
            ok = rel < tol_fd_B
            testB_pass &= _pass_fail(
                f"d{comp_label}/d{cn[6:]}  AD vs FD", ok,
                f"rel={rel:.2e}")

    # (c) Taylor R1 per control
    print("\n  (c) Taylor R1 (expected rate ~2.0 if gradient correct)")
    tol_R1 = 1.9
    for comp in ("x", "z"):
        for k, cn in enumerate(ctrl_names):
            Jhat_t, m0_t, _ = _build_pf_at(*base_B, comp)
            h = [Function(R_space).assign(h_taylor_B[i] if i == k else 0.0)
                 for i in range(n_ctrl)]
            try:
                rate = taylor_test(Jhat_t, m0_t, h)
                ok = rate >= tol_R1
                testB_pass &= _pass_fail(
                    f"R1  F_{comp}  {cn}", ok, f"rate={rate:.4f}")
            except Exception as e:
                print(f"  [FAIL] R1  F_{comp}  {cn}:  "
                      f"CRASH {type(e).__name__}: {str(e)[:80]}")
                testB_pass = False

    results.append(("Test B: First derivatives", testB_pass))

    # =====================================================================
    #  TEST C: Second derivatives — FD Hessian vs AD Hessian
    # =====================================================================
    #
    #  (a) H_FD: 3x3 Hessian via centred FD of AD gradient
    #      (each gradient eval = fresh tape at perturbed point)
    #  (b) H_AD: 3x3 Hessian via Jhat.hessian() (reverse-over-forward)
    #  (c) Schwarz symmetry of H_AD
    #  (d) Per-entry comparison H_AD vs H_FD

    _banner("TEST C: SECOND DERIVATIVES (H_FD vs H_AD)")

    base_C = (0.0, 0.0, 0.0)
    eps_hess = 1e-3

    testC_pass = True

    for comp in ("x", "z"):
        comp_label = f"F_p_{comp}"
        print(f"\n  ----- {comp_label} -----")

        # (a) H_FD
        print(f"  (a) H_FD via centred FD of AD gradient (eps={eps_hess:.0e})")
        H_FD = np.zeros((n_ctrl, n_ctrl))
        for k in range(n_ctrl):
            bp = list(base_C); bp[k] += eps_hess
            bm = list(base_C); bm[k] -= eps_hess
            g_p = _grad_ad(*bp, comp)
            g_m = _grad_ad(*bm, comp)
            H_FD[:, k] = (g_p - g_m) / (2 * eps_hess)
        _print_matrix(f"H_FD ({comp_label})", H_FD)

        # Symmetrize H_FD for display (FD is symmetric up to O(eps^2))
        H_FD_sym = 0.5 * (H_FD + H_FD.T)

        # (b) H_AD
        print(f"\n  (b) H_AD via Jhat.hessian()")
        Jhat_h, m0_h, _ = _build_pf_at(*base_C, comp)
        _ = Jhat_h.derivative()

        H_AD = np.zeros((n_ctrl, n_ctrl))
        h_ad_ok = True
        try:
            for k in range(n_ctrl):
                phi = [Function(R_space).assign(1.0 if i == k else 0.0)
                       for i in range(n_ctrl)]
                _ = Jhat_h.derivative()
                Hphi = Jhat_h.hessian(phi)
                H_AD[:, k] = np.array(
                    [float(Hphi[i].dat.data_ro[0]) for i in range(n_ctrl)])
        except Exception as e:
            import traceback
            print(f"    [SKIP] H_AD failed: {type(e).__name__}: {str(e)[:120]}")
            traceback.print_exc()
            testC_pass = False
            h_ad_ok = False

        if h_ad_ok:
            _print_matrix(f"H_AD ({comp_label})", H_AD)

            # (c) Schwarz symmetry
            sym_err = float(np.max(np.abs(H_AD - H_AD.T)))
            sym_scale = max(float(np.max(np.abs(H_AD))), 1e-30)
            sym_rel = sym_err / sym_scale
            sym_ok = sym_rel < 1e-8
            testC_pass &= _pass_fail(
                f"Schwarz {comp_label}", sym_ok,
                f"|H-H^T|/|H| = {sym_rel:.2e}")

            # (d) Per-entry comparison
            print(f"\n  (d) H_AD vs H_FD per entry:")
            print(f"      {'entry':>8}  {'H_AD':>14}  {'H_FD':>14}  "
                  f"{'|diff|':>10}  {'rel':>10}")
            hfd_scale = max(float(np.max(np.abs(H_FD))), 1e-30)
            tol_hess = 5e-2
            for i in range(n_ctrl):
                for j in range(n_ctrl):
                    diff = abs(H_AD[i, j] - H_FD[i, j])
                    denom = max(abs(H_FD[i, j]), 1e-30)
                    rel = diff / denom if abs(H_FD[i, j]) > 1e-4 * hfd_scale else float('nan')
                    rel_str = f"{rel:.2e}" if not np.isnan(rel) else "  noise"
                    entry = f"({ctrl_names[i][6]},{ctrl_names[j][6]})"
                    print(f"      {entry:>8}  {H_AD[i,j]:+14.6e}  "
                          f"{H_FD[i,j]:+14.6e}  {diff:10.2e}  {rel_str:>10}")
                    if not np.isnan(rel):
                        testC_pass &= (rel < tol_hess)

            max_rel = float(np.max(np.abs(H_AD - H_FD))) / hfd_scale
            _pass_fail(f"H_AD vs H_FD {comp_label}", max_rel < tol_hess,
                       f"max rel={max_rel:.2e}")

    results.append(("Test C: Second derivatives", testC_pass))

    # =====================================================================
    #  SUMMARY
    # =====================================================================

    _banner("SUMMARY")
    all_pass = True
    for name, passed in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}")
        all_pass &= passed

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)
    print("=" * 70)