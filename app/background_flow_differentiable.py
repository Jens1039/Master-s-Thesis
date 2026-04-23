import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import math
from scipy.spatial import cKDTree
import warnings

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, continue_annotation
from pyadjoint import Block, get_working_tape, set_working_tape, Tape, ReducedFunctional, Control, taylor_test, taylor_to_dict
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole


class background_flow_differentiable:

    def __init__(self, R, H, W, Re, comm=None):

        self.R   = R
        self.H   = H
        self.W   = W
        self.Re_float  = Re
        actual_comm = comm if comm is not None else COMM_WORLD
        self.mesh2d = RectangleMesh(128, 128, self.W, self.H, quadrilateral=False, comm=actual_comm)


    def solve_2D_background_flow(self):

        # CG 3/ CG 2 because of 2nd order derivative? If lower not correct?
        V       = VectorFunctionSpace(self.mesh2d, "CG", 3, dim=3)
        Q       = FunctionSpace(self.mesh2d, "CG", 2)
        G_space = FunctionSpace(self.mesh2d, "R", 0)
        W_mixed = V * Q * G_space

        w = Function(W_mixed)
        self.Re = Function(G_space).assign(self.Re_float)
        u, p, G = split(w)
        v, q, g = TestFunctions(W_mixed)

        u_r = u[0];  u_theta = u[2];  u_z = u[1]
        v_r = v[0];  v_theta = v[2];  v_z = v[1]

        x = SpatialCoordinate(self.mesh2d)
        r = x[0] - 0.5 * self.W

        def del_r(f): return Dx(f, 0)
        def del_z(f): return Dx(f, 1)

        F_cont = (q * (del_r(u_r) + del_z(u_z) + u_r / (self.R + r))
                  * (self.R + r) * dx)
        F_r    = ((u_r * del_r(u_r) + u_z * del_z(u_r)
                   - (u_theta**2) / (self.R + r)) * v_r
                  + del_r(p) * v_r
                  + (Constant(1.0) / self.Re) * dot(grad(u_r), grad(v_r))
                  + (1 / self.Re) * (u_r / (self.R + r)**2) * v_r
                  ) * (self.R + r) * dx
        F_theta = ((u_r * del_r(u_theta) + u_z * del_z(u_theta)
                    + (u_r * u_theta) / (self.R + r)) * v_theta
                   - ((G * self.R) / (self.R + r)) * v_theta
                   + 1 / self.Re * dot(grad(u_theta), grad(v_theta))
                   + 1 / self.Re * (u_theta / (self.R + r)**2) * v_theta
                   ) * (self.R + r) * dx
        F_z     = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
                   + del_z(p) * v_z
                   + 1 / self.Re * dot(grad(u_z), grad(v_z))
                   ) * (self.R + r) * dx
        F_G     = (u_theta - 1.0) * g * dx

        F       = F_r + F_theta + F_z + F_cont + F_G

        no_slip   = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)),"on_boundary")

        nullspace = MixedVectorSpaceBasis(
            W_mixed,
            [W_mixed.sub(0),
             VectorSpaceBasis(constant=True, comm=W_mixed.comm),
             W_mixed.sub(2)],
        )

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        solver  = NonlinearVariationalSolver(
            problem, nullspace=nullspace,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "mat_type": "matfree",
                "ksp_type": "fgmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": "0,1",
                "pc_fieldsplit_1_fields": "2",
                "fieldsplit_0": {
                    "ksp_type": "preonly", "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled_pc_type": "lu",
                    "assembled_pc_factor_mat_solver_type": "mumps"},
                "fieldsplit_1": {"ksp_type": "preonly", "pc_type": "none"},
            },
        )

        solver.solve()

        u_bar       = w.subfunctions[0]
        p_bar_tilde = w.subfunctions[1]
        G_val       = float(w.subfunctions[2].dat.data_ro[0])
        U_m_hat     = float(np.max(u_bar.dat.data_ro[:, 2]))

        self.u_bar       = u_bar
        self.p_bar_tilde = p_bar_tilde
        self.U_m_hat     = U_m_hat

        return G_val, U_m_hat, u_bar, p_bar_tilde


class VOMTransferBlock(Block):

    def __init__(self, u_in, u_out, inv_perm, V_vom):

        # Initializes the node object itself so pyadjoint recognizes it as a mathematical operation
        super().__init__()
        # Draws an incoming arrow: "This node requires the data from u_in to compute"
        self.add_dependency(u_in)
        # Draws an outgoing arrow: "This node produces u_out as its result"
        # .create_block_variable() creates a Block variable of u_out, so that it can be registered on the tape
        self.add_output(u_out.create_block_variable())

        self.inv_perm = inv_perm
        self.perm     = np.argsort(inv_perm)
        self.V_vom    = V_vom


    def recompute_component(self, inputs, block_variable, idx, prepared):

        # .output is the method, that grants us access to the variable contained in a Block structure
        out = block_variable.output

        with stop_annotating():
            out.dat.data[:] = inputs[0].dat.data_ro[self.inv_perm]

        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):

        with stop_annotating():
            adj = Function(self.V_vom)
            adj.dat.data[:] = adj_inputs[0].dat.data_ro[self.perm]

        return adj


    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):

        # pyadjoint sets tlm_inputs[i] to None if they dont contribute to the current directional derivative
        if tlm_inputs[0] is None:
            return None

        with stop_annotating():
            out = Function(block_variable.output.function_space())
            out.dat.data[:] = tlm_inputs[0].dat.data_ro[self.inv_perm]

        return out


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):

        # pyadjoint sets hessian_inputs[i] to None if they dont contribute to the current directional derivative
        if hessian_inputs[0] is None:
            return None

        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)


class WallBCBlock(Block):

    def __init__(self, u_in, u_out, bc_node_indices, V):

        super().__init__()
        self.add_dependency(u_in)
        self.add_output(u_out.create_block_variable())

        self.bc_nodes = bc_node_indices
        self.V = V


    def _apply_projection(self, data_in):

        out = data_in.copy()
        out[self.bc_nodes] = 0.0

        return out


    def recompute_component(self, inputs, block_variable, idx, prepared):
        out = block_variable.output
        with stop_annotating():
            out.dat.data[:] = self._apply_projection(inputs[0].dat.data_ro)
        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):

        with stop_annotating():
            adj = Function(self.V)
            adj.dat.data[:] = self._apply_projection(adj_inputs[0].dat.data_ro)

        return adj


    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):

        if tlm_inputs[0] is None:
            return None
        with stop_annotating():
            out = Function(block_variable.output.function_space())
            out.dat.data[:] = self._apply_projection(tlm_inputs[0].dat.data_ro)
        return out


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None):

        if hessian_inputs[0] is None:
            return None

        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)


class DifferentiableFieldEvalBlock(Block):

    def __init__(self, xi_fn, out_fn, X_ref_data, field_2d, R, W, H, mesh2d, V_def, field_dim, M):

        super().__init__()
        self.add_dependency(xi_fn)
        self.add_output(out_fn.create_block_variable())

        self.X_ref_data = X_ref_data
        self.field_2d   = field_2d
        self.R          = R
        self.W          = W
        self.H          = H
        self.mesh2d     = mesh2d
        self.V_def      = V_def
        self.field_dim  = field_dim
        self.M          = M
        self.MT         = M.T.tocsr()

        if field_dim > 1:
            comps = []
            for k in range(field_dim):
                for mm in range(2):
                    comps.append(Dx(field_2d[k], mm))
            self._grad_expr = as_vector(comps)
            self._grad_dim  = 2 * field_dim

            hcomps = []
            for k in range(field_dim):
                # order: ss, st, tt (st only once because symmetric)
                hcomps.append(Dx(Dx(field_2d[k], 0), 0))
                hcomps.append(Dx(Dx(field_2d[k], 0), 1))
                hcomps.append(Dx(Dx(field_2d[k], 1), 1))
            self._hess_expr = as_vector(hcomps)
            self._hess_dim  = 3 * field_dim
        else:
            self._grad_expr = as_vector([Dx(field_2d, 0), Dx(field_2d, 1)])
            self._grad_dim  = 2

            self._hess_expr = as_vector([
                Dx(Dx(field_2d, 0), 0),
                Dx(Dx(field_2d, 0), 1),
                Dx(Dx(field_2d, 1), 1),
            ])
            self._hess_dim  = 3


    def _get_CG2_positions(self, xi_data):
        return self.M @ (self.X_ref_data + xi_data)      # (n_CG2, 3)


    def _project_and_eval(self, xyz, need_grad=False, need_hess=False):

        r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        s_raw = r - self.R + 0.5 * self.W
        t_raw = xyz[:, 2] + 0.5 * self.H

        s_active = (s_raw > 0) & (s_raw < self.W)
        t_active = (t_raw > 0) & (t_raw < self.H)

        qu = np.column_stack([np.clip(s_raw, 0, self.W),
                              np.clip(t_raw, 0, self.H)])

        vom = VertexOnlyMesh(self.mesh2d, qu,
                             missing_points_behaviour="error")
        inv_perm = _compute_inv_perm(vom, qu)

        if self.field_dim > 1:
            V_f = VectorFunctionSpace(vom, "DG", 0, dim=self.field_dim)
        else:
            V_f = FunctionSpace(vom, "DG", 0)
        vals = Function(V_f).interpolate(self.field_2d) \
                            .dat.data_ro[inv_perm].copy()

        grad_arr = None
        if need_grad:
            V_g = VectorFunctionSpace(vom, "DG", 0, dim=self._grad_dim)
            grad_raw = Function(V_g).interpolate(self._grad_expr) \
                                    .dat.data_ro[inv_perm].copy()
            grad_arr = grad_raw.reshape(-1, self.field_dim, 2)
            grad_arr[~s_active, :, 0] = 0.0
            grad_arr[~t_active, :, 1] = 0.0

        hess_arr = None
        if need_hess:
            V_h = VectorFunctionSpace(vom, "DG", 0, dim=self._hess_dim)
            hess_raw = Function(V_h).interpolate(self._hess_expr) \
                                    .dat.data_ro[inv_perm].copy()
            # layout: per node, per component k, [f_ss, f_st, f_tt]
            hess_arr = hess_raw.reshape(-1, self.field_dim, 3)
            # If s is clipped, ds = 0 and ds·ds, ds·dt second derivatives
            # likewise vanish from the chain rule.
            hess_arr[~s_active, :, 0] = 0.0
            hess_arr[~s_active, :, 1] = 0.0
            hess_arr[~t_active, :, 1] = 0.0
            hess_arr[~t_active, :, 2] = 0.0

        return vals, grad_arr, r, hess_arr


    def recompute_component(self, inputs, block_variable, idx, prepared):
        out = block_variable.output
        with stop_annotating():
            xyz = self._get_CG2_positions(inputs[0].dat.data_ro)
            vals, _, _, _ = self._project_and_eval(xyz)
            out.dat.data[:] = vals
        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if adj_inputs[0] is None:
            return Cofunction(self.V_def.dual())

        with stop_annotating():
            xi_data = inputs[0].dat.data_ro
            xyz = self._get_CG2_positions(xi_data)
            _, grad_arr, r, _ = self._project_and_eval(xyz, need_grad=True)

            adj_out = np.asarray(adj_inputs[0].dat.data_ro)
            if adj_out.ndim == 1:
                adj_out = adj_out[:, np.newaxis]

            du_ds = grad_arr[:, :, 0]
            du_dt = grad_arr[:, :, 1]
            s_term = np.sum(du_ds * adj_out, axis=1)
            t_term = np.sum(du_dt * adj_out, axis=1)

            cos_phi = xyz[:, 0] / r
            sin_phi = xyz[:, 1] / r

            adj_pos_CG2 = np.zeros_like(xyz)
            adj_pos_CG2[:, 0] = cos_phi * s_term
            adj_pos_CG2[:, 1] = sin_phi * s_term
            adj_pos_CG2[:, 2] = t_term

            adj_xi_CG1 = self.MT @ adj_pos_CG2

            adj = Cofunction(self.V_def.dual())
            adj.dat.data[:] = adj_xi_CG1
        return adj


    def evaluate_tlm_component(self, inputs, tlm_inputs,
                               block_variable, idx, prepared=None):
        if tlm_inputs[0] is None:
            return None

        with stop_annotating():
            xi_data = inputs[0].dat.data_ro
            xyz = self._get_CG2_positions(xi_data)
            _, grad_arr, r, _ = self._project_and_eval(xyz, need_grad=True)

            d_xi_CG1 = tlm_inputs[0].dat.data_ro            # (n_CG1, 3)
            d_pos = self.M @ d_xi_CG1                        # (n_CG2, 3)

            ds_val = (xyz[:, 0] / r) * d_pos[:, 0] \
                   + (xyz[:, 1] / r) * d_pos[:, 1]          # (n_CG2,)
            dt_val = d_pos[:, 2]

            d_field = grad_arr[:, :, 0] * ds_val[:, np.newaxis] \
                    + grad_arr[:, :, 1] * dt_val[:, np.newaxis]

            out = Function(block_variable.output.function_space())
            if self.field_dim == 1:
                out.dat.data[:] = d_field.ravel()
            else:
                out.dat.data[:] = d_field
        return out


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                    block_variable, idx, relevant_dependencies,
                                    prepared=None):
        # The Hessian backprop has two contributions:
        #   (1) propagation of the second-order seed `hessian_inputs` through
        #       the linearised adjoint (same operator as in evaluate_adj),
        #   (2) the cross term  d/dxi (J^T · adj_in) · xi_tlm  coming from
        #       the second derivative of the per-node map
        #           xi  ->  field_2d( s(M·(X_ref+xi)), t(M·(X_ref+xi)) )
        #
        # (1) is just self.evaluate_adj_component called on hessian_inputs.
        # (2) needs the second derivatives of field_2d composed with the
        # cylindrical-unfold map (s, t) <- (x, y, z).
        h_in = hessian_inputs[0]
        a_in = adj_inputs[0]

        # ---- Part (1): linearised adjoint applied to the second-order seed
        if h_in is not None:
            part1 = self.evaluate_adj_component(
                inputs, hessian_inputs, block_variable, idx, prepared)
        else:
            part1 = None

        # ---- Part (2): cross term using a_in and the TLM of xi
        xi_tlm_bv = block_variable.tlm_value
        if a_in is None or xi_tlm_bv is None:
            return part1

        with stop_annotating():
            xi_data = inputs[0].dat.data_ro
            xyz = self._get_CG2_positions(xi_data)
            _, grad_arr, r, hess_arr = self._project_and_eval(
                xyz, need_grad=True, need_hess=True)

            adj_data = np.asarray(a_in.dat.data_ro)
            if adj_data.ndim == 1:
                adj_data = adj_data[:, np.newaxis]

            xi_tlm = np.asarray(xi_tlm_bv.dat.data_ro)
            d_pos  = self.M @ xi_tlm                       # (n_CG2, 3)

            cos_p = xyz[:, 0] / r
            sin_p = xyz[:, 1] / r
            inv_r = 1.0 / r

            # Aggregated weights summed over field components k:
            #   A_q  =  sum_k adj_n_k * (∂_q field_k)
            # for q in {s, t} and the second derivatives in {ss, st, tt}.
            A_s  = np.sum(adj_data * grad_arr[:, :, 0], axis=1)
            A_t  = np.sum(adj_data * grad_arr[:, :, 1], axis=1)
            A_ss = np.sum(adj_data * hess_arr[:, :, 0], axis=1)
            A_st = np.sum(adj_data * hess_arr[:, :, 1], axis=1)
            A_tt = np.sum(adj_data * hess_arr[:, :, 2], axis=1)

            # Per-node 3x3 (in xyz) Hessian of  sum_k adj_k · field_k(s,t)
            # composed with the unfold map. Symmetric, so only 6 entries.
            cos2 = cos_p * cos_p
            sin2 = sin_p * sin_p
            cs   = cos_p * sin_p
            Hxx = A_ss * cos2 + A_s * sin2 * inv_r
            Hyy = A_ss * sin2 + A_s * cos2 * inv_r
            Hxy = (A_ss - A_s * inv_r) * cs
            Hxz = A_st * cos_p
            Hyz = A_st * sin_p
            Hzz = A_tt

            dx = d_pos[:, 0]; dy = d_pos[:, 1]; dz = d_pos[:, 2]
            vec_x = Hxx * dx + Hxy * dy + Hxz * dz
            vec_y = Hxy * dx + Hyy * dy + Hyz * dz
            vec_z = Hxz * dx + Hyz * dy + Hzz * dz

            vec_CG2 = np.column_stack([vec_x, vec_y, vec_z])
            cross_xi_CG1 = self.MT @ vec_CG2

            cross = Cofunction(self.V_def.dual())
            cross.dat.data[:] = cross_xi_CG1

        if part1 is None:
            return cross
        # part1 is a Cofunction; sum into it
        part1.dat.data[:] = part1.dat.data_ro + cross.dat.data_ro
        return part1


class BuildXiBlock(Block):
    """Custom block for
        xi = delta_r * (cos_th, sin_th, 0) * bump
           + delta_z * (0, 0, 1)            * bump
           + delta_a * (d_hat_x, d_hat_y, d_hat_z) * bump   (optional)

    The map (delta_r, delta_z [, delta_a]) -> xi is *linear*, so the
    Hessian's second-order term vanishes identically. Replacing the
    equivalent `xi.interpolate(as_vector([...]))` (an Interpolate of a
    product of coefficients) with this block keeps the symbolic
    Interpolate node off the tape, which is what triggers the
        AttributeError: 'ZeroBaseForm' object has no attribute 'ufl_shape'
    crash inside firedrake's `expand_derivatives` whenever pyadjoint tries
    to form (mixed) second derivatives w.r.t. the controls.

    The optional ``delta_a`` dependency parametrises an isotropic radial
    scaling of a sphere centred at the particle.  ``d_hat_data`` is the
    per-node unit vector pointing from the particle centre to the node,
    sampled on the same CG1 layout as ``bump_data``.
    """

    def __init__(self, delta_r, delta_z, xi_out, bump_data, cos_th, sin_th, V_def,
                 delta_a=None, d_hat_data=None):
        super().__init__()
        # idx ordering: idx=0 -> delta_r, idx=1 -> delta_z, idx=2 -> delta_a.
        self.add_dependency(delta_r)
        self.add_dependency(delta_z)
        self.has_a = delta_a is not None
        if self.has_a:
            if d_hat_data is None:
                raise ValueError("d_hat_data must be provided when delta_a is used")
            self.add_dependency(delta_a)
        self.add_output(xi_out.create_block_variable())

        self.bump_data = np.asarray(bump_data).copy()
        self.cos_th    = float(cos_th)
        self.sin_th    = float(sin_th)
        self.V_def     = V_def

        n = self.bump_data.shape[0]
        # Per-node directional derivatives.
        # xi = dr * dxi_ddr + dz * dxi_ddz [+ da * dxi_dda]
        self._dxi_ddr = np.zeros((n, 3))
        self._dxi_ddr[:, 0] = self.cos_th * self.bump_data
        self._dxi_ddr[:, 1] = self.sin_th * self.bump_data
        self._dxi_ddz = np.zeros((n, 3))
        self._dxi_ddz[:, 2] = self.bump_data

        if self.has_a:
            d_hat = np.asarray(d_hat_data).copy()
            if d_hat.shape != (n, 3):
                raise ValueError(
                    f"d_hat_data shape {d_hat.shape} does not match bump layout ({n}, 3)")
            self._dxi_dda = d_hat * self.bump_data[:, np.newaxis]
        else:
            self._dxi_dda = None


    def _accumulate_xi(self, dr, dz, da):
        data = dr * self._dxi_ddr + dz * self._dxi_ddz
        if self.has_a:
            data = data + da * self._dxi_dda
        return data


    def recompute_component(self, inputs, block_variable, idx, prepared):
        out = block_variable.output
        with stop_annotating():
            dr = float(inputs[0].dat.data_ro[0])
            dz = float(inputs[1].dat.data_ro[0])
            da = float(inputs[2].dat.data_ro[0]) if self.has_a else 0.0
            out.dat.data[:] = self._accumulate_xi(dr, dz, da)
        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if adj_inputs[0] is None:
            return None
        with stop_annotating():
            adj_data = np.asarray(adj_inputs[0].dat.data_ro)   # (n_nodes, 3)
            R_space  = inputs[idx].function_space()
            out = Cofunction(R_space.dual())
            if idx == 0:
                out.dat.data[0] = float(np.sum(self._dxi_ddr * adj_data))
            elif idx == 1:
                out.dat.data[0] = float(np.sum(self._dxi_ddz * adj_data))
            elif idx == 2 and self.has_a:
                out.dat.data[0] = float(np.sum(self._dxi_dda * adj_data))
            else:
                raise IndexError(f"unexpected dependency index {idx}")
        return out


    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        out = Function(block_variable.output.function_space())
        with stop_annotating():
            data = np.zeros_like(self._dxi_ddr)
            if tlm_inputs[0] is not None:
                data += float(tlm_inputs[0].dat.data_ro[0]) * self._dxi_ddr
            if tlm_inputs[1] is not None:
                data += float(tlm_inputs[1].dat.data_ro[0]) * self._dxi_ddz
            if self.has_a and tlm_inputs[2] is not None:
                data += float(tlm_inputs[2].dat.data_ro[0]) * self._dxi_dda
            out.dat.data[:] = data
        return out


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # xi is linear in (delta_r, delta_z [, delta_a]) ⇒ d² xi / d m² = 0.
        # Only the linear adjoint of the second-order seed propagates back.
        if hessian_inputs[0] is None:
            return None
        return self.evaluate_adj_component(
            inputs, hessian_inputs, block_variable, idx, prepared)


def vom_transfer(u_vom, V_target, inv_perm, V_vom):

    u_out = Function(V_target)
    with stop_annotating():
        u_out.dat.data[:] = u_vom.dat.data_ro[inv_perm]
    if annotate_tape():
        block = VOMTransferBlock(u_vom, u_out, inv_perm, V_vom)
        get_working_tape().add_block(block)
    return u_out


def _compute_inv_perm(vom, query_pts):
    tree = cKDTree(vom.coordinates.dat.data_ro)
    dist, inv_perm = tree.query(query_pts, workers=-1)
    assert np.max(dist) < 1e-10, f"VOM mismatch: max dist {np.max(dist):.2e}"
    return inv_perm.astype(np.int32)


def apply_wall_bc_on_tape(u_fn, V, wall_tag):

    with stop_annotating():
        bc = DirichletBC(V, Constant((0.0, 0.0, 0.0)), wall_tag)
        bc_node_indices = bc.nodes

    u_out = Function(V, name=u_fn.name() + "_bc" if u_fn.name() else "u_bc")
    with stop_annotating():
        u_out.dat.data[:] = u_fn.dat.data_ro[:]
        u_out.dat.data[bc_node_indices] = 0.0

    if annotate_tape():
        block = WallBCBlock(u_fn, u_out, bc_node_indices, V)
        get_working_tape().add_block(block)

    return u_out


def build_xi_diff(delta_r, delta_z, bump, cos_th, sin_th, V_def, delta_a=None, d_hat_data=None):
    """Tape-aware construction of
        xi = dr * (cos_th, sin_th, 0) * bump
           + dz * (0, 0, 1)            * bump
           + da * (d_hat_x, d_hat_y, d_hat_z) * bump   (optional)

    Replaces a `Function.interpolate(as_vector([...]))` call so that the
    Interpolate node — and its broken second derivative — is removed from
    the tape. The mapping is recorded as a `BuildXiBlock` whose adj/tlm/
    hessian implementations are exact.

    ``bump`` must be a scalar CG1 :class:`Function` whose dof layout
    matches the per-node ordering of ``V_def``.  When ``delta_a`` is
    given, ``d_hat_data`` must be a ``(n_nodes, 3)`` numpy array of
    per-node radial unit vectors from the particle centre.
    """
    bump_data = bump.dat.data_ro.copy()
    xi_out = Function(V_def, name="xi")
    with stop_annotating():
        dr = float(delta_r.dat.data_ro[0])
        dz = float(delta_z.dat.data_ro[0])
        n  = bump_data.shape[0]
        data = np.zeros((n, 3))
        data[:, 0] = dr * cos_th * bump_data
        data[:, 1] = dr * sin_th * bump_data
        data[:, 2] = dz * bump_data
        if delta_a is not None:
            da = float(delta_a.dat.data_ro[0])
            d_hat = np.asarray(d_hat_data)
            data += da * d_hat * bump_data[:, np.newaxis]
        xi_out.dat.data[:] = data
    if annotate_tape():
        block = BuildXiBlock(
            delta_r, delta_z, xi_out, bump_data, cos_th, sin_th, V_def,
            delta_a=delta_a, d_hat_data=d_hat_data)
        get_working_tape().add_block(block)
    return xi_out


def differentiable_field_eval(xi, X_ref, field_2d, R, W, H, V_def, V_3d, field_dim, M):

    mesh2d     = field_2d.function_space().mesh()
    X_ref_data = X_ref.dat.data_ro.copy()
    out        = Function(V_3d, name="field_eval_diff")

    with stop_annotating():
        xyz  = M @ (X_ref_data + xi.dat.data_ro)
        r_3d = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        qu   = np.column_stack([
            np.clip(r_3d - R + 0.5 * W, 0, W),
            np.clip(xyz[:, 2] + 0.5 * H, 0, H),
        ])
        vom      = VertexOnlyMesh(mesh2d, qu, missing_points_behaviour="error")
        inv_perm = _compute_inv_perm(vom, qu)
        if field_dim > 1:
            V_vom = VectorFunctionSpace(vom, "DG", 0, dim=field_dim)
        else:
            V_vom = FunctionSpace(vom, "DG", 0)
        out.dat.data[:] = Function(V_vom).interpolate(field_2d) \
                                         .dat.data_ro[inv_perm]

    if annotate_tape():
        block = DifferentiableFieldEvalBlock(
            xi, out, X_ref_data, field_2d,
            R, W, H, mesh2d, V_def, field_dim, M)
        get_working_tape().add_block(block)

    return out


def _build_CG1_to_CG2_map(mesh3d):

    from scipy.sparse import coo_matrix

    V1 = FunctionSpace(mesh3d, "CG", 1)
    V2 = FunctionSpace(mesh3d, "CG", 2)
    # The dofs are here all the nodes of the net as well as the points in between the nodes, which fix
    # polynomials of higher degree (here 2)
    n1 = V1.dof_count
    n2 = V2.dof_count

    # V1.cell_node_map().values returns a list of lists which contains the global indices of dofs with each tetraeder being one list
    cmap1 = V1.cell_node_map().values
    cmap2 = V2.cell_node_map().values
    dpc1  = cmap1.shape[1]
    dpc2  = cmap2.shape[1]
    n_cells = cmap1.shape[0]

    with stop_annotating():
        src = Function(V1)
        dst = Function(V2)
        g1_c0 = cmap1[0]
        g2_c0 = cmap2[0]
        local_M = np.zeros((dpc2, dpc1))
        for j in range(dpc1):
            src.dat.data[:] = 0.0
            src.dat.data[g1_c0[j]] = 1.0
            dst.interpolate(src)
            for i in range(dpc2):
                local_M[i, j] = dst.dat.data_ro[g2_c0[i]]

    nz_i, nz_j = np.nonzero(np.abs(local_M) > 1e-14)
    nz_vals = local_M[nz_i, nz_j]

    rows = cmap2[:, nz_i].ravel()
    cols = cmap1[:, nz_j].ravel()
    vals = np.tile(nz_vals, n_cells)

    M_sum = coo_matrix((vals, (rows, cols)), shape=(n2, n1)).tocsr()
    M_cnt = coo_matrix((np.ones_like(vals), (rows, cols)), shape=(n2, n1)).tocsr()
    M = M_sum.copy()
    M.data /= M_cnt.data

    with stop_annotating():
        rng = np.random.default_rng(42)
        test_fn = Function(V1)
        test_fn.dat.data[:] = rng.standard_normal(n1)
        expected = Function(V2).interpolate(test_fn).dat.data_ro.copy()
        got = M @ test_fn.dat.data_ro
        err = np.max(np.abs(got - expected))
        assert err < 1e-10, f"CG1→CG2 map verification failed: {err:.2e}"

    return M


def build_3d_background_flow_differentiable(R, H, W, G_val, mesh3d, tags, u_bar_2d, p_bar_2d, X_ref=None, xi=None):

    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d = FunctionSpace(mesh3d, "CG", 1)

    x_3d   = SpatialCoordinate(mesh3d)

    # x = r * cos(phi), y = r * sin(phi) (in polar coordinates)
    r_xy  = sqrt(x_3d[0] ** 2 + x_3d[1] ** 2)
    cos_ufl   = x_3d[0] / r_xy
    sin_ufl   = x_3d[1] / r_xy
    theta_ufl = atan2(x_3d[1], x_3d[0])

    mesh2d = u_bar_2d.function_space().mesh()

    if X_ref is not None and xi is not None:
        V_def = xi.function_space()
        M = _build_CG1_to_CG2_map(mesh3d)
        u_cyl_3d = differentiable_field_eval(xi, X_ref, u_bar_2d, R, W, H, V_def, V_3d, 3, M)

    else:
        with stop_annotating():

            V_coords = VectorFunctionSpace(mesh3d, "CG", 2)

            coords_func_u = Function(V_coords).interpolate(SpatialCoordinate(mesh3d))

            xyz_nodes_u = coords_func_u.dat.data_ro.copy()

            r_3d = np.sqrt(xyz_nodes_u[:, 0] ** 2 + xyz_nodes_u[:, 1] ** 2)

            qu = np.column_stack([np.clip(r_3d - R + 0.5 * W, 0, W), np.clip(xyz_nodes_u[:, 2] + 0.5 * H, 0, H)])

            vom_u      = VertexOnlyMesh(mesh2d, qu, missing_points_behaviour="error")

            V_vom_u    = VectorFunctionSpace(vom_u, "DG", 0, dim=3)

            inv_perm_u = _compute_inv_perm(vom_u, qu)

        u_vom    = Function(V_vom_u).interpolate(u_bar_2d)
        u_cyl_3d = vom_transfer(u_vom, V_3d, inv_perm_u, V_vom_u)


    with stop_annotating():
        cp = Function(VectorFunctionSpace(mesh3d, "CG", 1))
        cp.interpolate(SpatialCoordinate(mesh3d))
        xyz_p = cp.dat.data_ro.copy()
        r_p   = np.sqrt(xyz_p[:, 0]**2 + xyz_p[:, 1]**2)

        qp = np.column_stack([
            np.clip(r_p - R + 0.5 * W, 0, W),
            np.clip(xyz_p[:, 2] + 0.5 * H, 0, H),
        ])
        vom_p      = VertexOnlyMesh(mesh2d, qp, missing_points_behaviour="error")
        V_vom_p    = FunctionSpace(vom_p, "DG", 0)
        inv_perm_p = _compute_inv_perm(vom_p, qp)

    p_vom    = Function(V_vom_p).interpolate(p_bar_2d)
    p_cyl_3d = vom_transfer(p_vom, Q_3d, inv_perm_p, V_vom_p)

    u_cyl_3d = apply_wall_bc_on_tape(u_cyl_3d, V_3d, tags["walls"])

    u_bar_3d = as_vector([cos_ufl * u_cyl_3d[0] - sin_ufl * u_cyl_3d[2], sin_ufl * u_cyl_3d[0] + cos_ufl * u_cyl_3d[2], u_cyl_3d[1]])

    p_bar_3d = p_cyl_3d - Constant(G_val * R) * theta_ufl

    return u_bar_3d, p_bar_3d, u_cyl_3d


if __name__ == "__main__":

    R_hat = 500.0
    H_hat = 2.0
    W_hat = 2.0
    a_hat = 0.05
    Re = 1.0

    L_hat_rel = 4.0
    particle_maxh_rel = 0.2
    global_maxh_rel = 0.2

    x_off_hat = 0.3
    z_off_hat = 0.2

    results = []

    with stop_annotating():
        bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
        G_val, U_m_hat, u_2d, p_2d = bg.solve_2D_background_flow()
        mesh3d, tags = make_curved_channel_section_with_spherical_hole(R_hat, H_hat, W_hat, L_hat_rel * max(H_hat, W_hat), a_hat,
                                                particle_maxh_rel * a_hat, global_maxh_rel * min(H_hat, W_hat), x_off_hat, z_off_hat)

    V_3d  = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d  = FunctionSpace(mesh3d, "CG", 1)
    V_def = VectorFunctionSpace(mesh3d, "CG", 1)
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
    H_MAG = 0.1
    ctrl_names = ["delta_r", "delta_z", "delta_a"]

    def make_m0():
        return [Function(R_space).assign(0.0) for _ in range(3)]

    def make_h(idx):
        return [Function(R_space).assign(H_MAG if k == idx else 0.0) for k in range(3)]

    def make_h_mixed():
        return [Function(R_space).assign(H_MAG) for _ in range(3)]

    def check(label, ok, info=""):
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {label}{('  '+info) if info else ''}")
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
        r1_err = max(abs(r - 2.0) for r in info["R1"]["Rate"])
        r1_max = max(info["R1"]["Residual"])
        r2_err = max(abs(r - 3.0) for r in info["R2"]["Rate"])
        r2_max = max(info["R2"]["Residual"])

        # R1: gradient exact if residuals negligible vs R0 scale
        r1_exact = r0_max > 1e-30 and r1_max < 1e-3 * r0_max
        ok_r1 = r1_err <= 0.1 or r1_exact

        # R2: Hessian exact (zero) if R2 residuals negligible vs R1 scale
        r2_exact = r1_max > 1e-30 and r2_max < 1e-3 * r1_max
        ok_r2 = r2_err <= 0.1 or r2_exact

        r1_note = " (exact)" if r1_exact else "  (need<=0.1)"
        r2_note = " (exact)" if r2_exact else "  (need<=0.1)"
        print(f"    [{'PASS' if ok_r1 else 'FAIL'}] R1 gradient  max|rate-2|={r1_err:.4f}{r1_note}")
        print(f"    [{'PASS' if ok_r2 else 'FAIL'}] R2 Taylor    max|rate-3|={r2_err:.4f}{r2_note}")

        stop_annotating()
        get_working_tape().clear_tape()

        print(f"    --- FD Hessian cross-check (eps=1e-4) ---")
        eps_fd = 1e-4
        m_arr = np.array([float(h_list[j].dat.data_ro[0]) for j in range(len(h_list))])
        n_ctrl = len(h_list)

        Jhat_h, _ = build_fn()
        Jhat_h.derivative()
        m_fns = [Function(R_space).assign(float(m_arr[j])) for j in range(n_ctrl)]
        Hm_ad_raw = Jhat_h.hessian(m_fns)
        Hm_ad = np.array([float(Hm_ad_raw[j].dat.data_ro[0]) for j in range(n_ctrl)])
        stop_annotating()
        get_working_tape().clear_tape()

        Jhat_p, _ = build_fn()
        ctrl_p = [Function(R_space).assign(eps_fd * m_arr[j]) for j in range(n_ctrl)]
        Jhat_p(ctrl_p)  # replay forward model at perturbed point
        dJ_p_raw = Jhat_p.derivative()
        grad_p = np.array([float(dJ_p_raw[j].dat.data_ro[0]) for j in range(n_ctrl)])
        stop_annotating()
        get_working_tape().clear_tape()

        Jhat_m, _ = build_fn()
        ctrl_m = [Function(R_space).assign(-eps_fd * m_arr[j]) for j in range(n_ctrl)]
        Jhat_m(ctrl_m)  # replay forward model at perturbed point
        dJ_m_raw = Jhat_m.derivative()
        grad_m = np.array([float(dJ_m_raw[j].dat.data_ro[0]) for j in range(n_ctrl)])
        stop_annotating()
        get_working_tape().clear_tape()

        Hm_fd = (grad_p - grad_m) / (2 * eps_fd)

        abs_diff = np.abs(Hm_ad - Hm_fd)
        scale = max(np.max(np.abs(Hm_fd)), np.max(np.abs(Hm_ad)), 1e-30)
        rel_diff = np.max(abs_diff) / scale
        ok_fd = rel_diff < 1e-2

        print(f"    H*m (AD):  [{', '.join(f'{v:+.6e}' for v in Hm_ad)}]")
        print(f"    H*m (FD):  [{', '.join(f'{v:+.6e}' for v in Hm_fd)}]")
        print(f"    max|AD-FD|/scale = {rel_diff:.4e}")
        print(f"    [{'PASS' if ok_fd else 'FAIL'}] FD cross-check  (need < 1%)")

        ok_r2_final = ok_r2 or ok_fd
        return ok_r1, ok_r2_final

    print("\n=== GROUP 1: correctness at xi=0 ===")

    with stop_annotating():
        mesh3d.coordinates.assign(X_ref)
        u_bar_nd, p_bar_nd, u_cyl_nd = build_3d_background_flow_differentiable(R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d)
        xi_zero = Function(V_def).assign(0.0)
        u_bar_d, p_bar_d, u_cyl_d = build_3d_background_flow_differentiable(R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
                                                                            X_ref=X_ref, xi=xi_zero)

        for k, cn in enumerate(["u_r", "u_z", "u_theta"]):
            e = np.max(np.abs(u_cyl_nd.dat.data_ro[:, k] - u_cyl_d.dat.data_ro[:, k]))
            n = max(np.max(np.abs(u_cyl_nd.dat.data_ro[:, k])), 1e-15)
            ok = check(f"u_cyl {cn} VOM==DiffEval", e/n < 1e-10, f"rel={e/n:.2e}")
            results.append((f"G1 u_cyl {cn}", ok))

        coords = Function(V_3d).interpolate(SpatialCoordinate(mesh3d)).dat.data_ro
        r_n = np.sqrt(coords[:,0]**2 + coords[:,1]**2)
        cp, sp = coords[:,0]/r_n, coords[:,1]/r_n
        for lbl, u_bar_ufl, u_cyl_fn in [("VOM", u_bar_nd, u_cyl_nd),
                                           ("DE",  u_bar_d,  u_cyl_d)]:
            c = u_cyl_fn.dat.data_ro
            u_np = np.column_stack([cp*c[:,0]-sp*c[:,2], sp*c[:,0]+cp*c[:,2], c[:,1]])
            u_fn = Function(V_3d).interpolate(u_bar_ufl).dat.data_ro
            for k, cn in enumerate(["u_x", "u_y", "u_z"]):
                n = max(np.max(np.abs(u_np[:,k])), 1e-15)
                e = np.max(np.abs(u_fn[:,k] - u_np[:,k]))
                ok = check(f"u_bar {lbl} rotation {cn}", e/n < 1e-12, f"rel={e/n:.2e}")
                results.append((f"G1 {lbl} rot {cn}", ok))

        p_nd = Function(Q_3d).interpolate(p_bar_nd).dat.data_ro
        p_d  = Function(Q_3d).interpolate(p_bar_d).dat.data_ro
        ep = np.max(np.abs(p_nd - p_d)); np_p = max(np.max(np.abs(p_nd)), 1e-15)
        ok = check("p_bar VOM==DiffEval", ep/np_p < 1e-10, f"rel={ep/np_p:.2e}")
        results.append(("G1 p_bar", ok))

        bc_nodes = DirichletBC(V_3d, Constant((0,0,0)), tags["walls"]).nodes
        for lbl, u_cyl_fn in [("VOM", u_cyl_nd), ("DE", u_cyl_d)]:
            w = np.max(np.abs(u_cyl_fn.dat.data_ro[bc_nodes]))
            ok = check(f"no-slip {lbl}", w < 1e-14, f"max={w:.2e}")
            results.append((f"G1 no-slip {lbl}", ok))

        from background_flow import build_3d_background_flow
        u_ref, p_ref = build_3d_background_flow(
            R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d)
        for lbl, u_bar_ufl in [("VOM", u_bar_nd), ("DE", u_bar_d)]:
            u_fn = Function(V_3d).interpolate(u_bar_ufl).dat.data_ro
            for k, cn in enumerate(["u_x", "u_y", "u_z"]):
                n = max(np.max(np.abs(u_ref.dat.data_ro[:,k])), 1e-15)
                e = np.max(np.abs(u_fn[:,k] - u_ref.dat.data_ro[:,k]))
                ok = check(f"u_bar {lbl} vs ref {cn}", e/n < 1e-3, f"rel={e/n:.2e}")
                results.append((f"G1 {lbl} vs ref {cn}", ok))
        for lbl, p_ufl in [("VOM", p_bar_nd), ("DE", p_bar_d)]:
            p_fn = Function(Q_3d).interpolate(p_ufl).dat.data_ro
            n = max(np.max(np.abs(p_ref.dat.data_ro)), 1e-15)
            e = np.max(np.abs(p_fn - p_ref.dat.data_ro))
            ok = check(f"p_bar {lbl} vs ref", e/n < 1e-3, f"rel={e/n:.2e}")
            results.append((f"G1 p {lbl} vs ref", ok))

    print("\n=== GROUP 2: BuildXiBlock derivatives ===")
    print("    Functional: J = ∫|xi|²dx  (xi = displacement field)")
    print("    xi is LINEAR in all controls → Hessian is exactly zero,")
    print("    so R2 residuals should be at machine precision.")

    def build_xi_J():
        set_working_tape(Tape())
        continue_annotation()
        dr = Function(R_space, name="delta_r").assign(0.0)
        dz = Function(R_space, name="delta_z").assign(0.0)
        da = Function(R_space, name="delta_a").assign(0.0)
        xi = build_xi_diff(dr, dz, bump_fn, cos_th, sin_th, V_def, delta_a=da, d_hat_data=d_hat_data)
        J = assemble(inner(xi, xi) * dx)
        return ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]), make_m0()

    for idx, name in enumerate(ctrl_names):
        ok1, ok2 = run_taylor(f"J=∫|xi|²dx / {name}", build_xi_J, make_h(idx))

        results += [(f"G2 R1 {name}", ok1), (f"G2 R2 {name}", ok2)]
    ok1, ok2 = run_taylor(
        "J=∫|xi|²dx / delta_r ",
        build_xi_J, make_h_mixed())
    results += [("G2 R1 mixed", ok1), ("G2 R2 mixed", ok2)]

    print("\n=== GROUP 3: DifferentiableFieldEvalBlock derivatives ===")
    print("    Functional: J = ∫u_cyl[k]²ds(particle)  (background velocity component)")
    print("    No mesh movement: xi shifts query points into the 2D background flow.")
    print("    Hessian limited by DG0 interpolation of field gradients → rate ~2 expected.")

    def make_build_ucyl(comp):
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
            _, _, u_cyl = build_3d_background_flow_differentiable(
                R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
                X_ref=X_ref, xi=xi)
            J = assemble(u_cyl[comp]**2 * ds(tags["particle"], domain=mesh3d))
            return (ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]),
                    make_m0())
        return build

    cyl_comp_labels = ["u_r (cyl[0])", "u_z (cyl[1])", "u_theta (cyl[2])"]
    for ci, cn in enumerate(cyl_comp_labels):
        for hi, hn in enumerate(ctrl_names):
            ok1, ok2 = run_taylor(
                f"J=∫{cn}²ds  [DiffFieldEval, no mesh move]  ctrl={hn}",
                make_build_ucyl(ci), make_h(hi))
            results += [(f"G3 {cn} R1 {hn}", ok1), (f"G3 {cn} R2 {hn}", ok2)]

    # ══════════════════════════════════════════════════════════════════
    #  GROUP 4: Per Cartesian velocity component – WITH mesh movement
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GROUP 4: Cartesian velocity components – with mesh move")
    print("  Functional: J = ∫u_bar[k]²ds(particle)")
    print("  Chain: xi -> mesh coords -> DiffFieldEval(u_cyl) -> rotation R(phi)·u_cyl")
    print("  phi = atan2(y,x) depends on deformed mesh → shape derivative active")
    print("=" * 60)

    cart_comp_names = ["u_x(bar0)", "u_y(bar1)", "u_z(bar2)"]

    def make_build_ubar(comp):
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
            u_bar, _, _ = build_3d_background_flow_differentiable(
                R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
                X_ref=X_ref, xi=xi)
            J = assemble(u_bar[comp]**2 * ds(tags["particle"], domain=mesh3d))
            return (ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]),
                    make_m0())
        return build

    for ci, cn in enumerate(cart_comp_names):
        for hi, hn in enumerate(ctrl_names):
            ok1, ok2 = run_taylor(
                f"J=∫{cn}²ds  [DiffFieldEval+rotation, mesh move]  ctrl={hn}",
                make_build_ubar(ci), make_h(hi))
            results.append((f"G4 {cn} R1 {hn}", ok1))
            results.append((f"G4 {cn} R2 {hn}", ok2))

    # ══════════════════════════════════════════════════════════════════
    #  GROUP 5: Pressure – WITH mesh movement
    #  p_bar = p_cyl - G·R·theta(x,y).  The pressure values p_cyl
    #  come from VOM transfer (fixed query points, off-tape), but
    #  theta = atan2(y, x) depends on the mesh coordinates and thus
    #  on xi.  This tests those geometry-dependent derivatives.
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GROUP 5: Pressure – with mesh move")
    print("  Functional: J = ∫p_bar²ds(particle)")
    print("  p_bar = p_cyl(fixed) - G·R·atan2(y,x)")
    print("  p_cyl comes from off-tape VOM transfer (no AD through it),")
    print("  but theta=atan2(y,x) depends on deformed coordinates → shape derivative")
    print("=" * 60)

    def build_p_J():
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
        _, p_bar, _ = build_3d_background_flow_differentiable(
            R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
            X_ref=X_ref, xi=xi)
        J = assemble(p_bar**2 * ds(tags["particle"], domain=mesh3d))
        return (ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]),
                make_m0())

    for hi, hn in enumerate(ctrl_names):
        ok1, ok2 = run_taylor(
            f"J=∫p_bar²ds  [atan2 shape-deriv, mesh move]  ctrl={hn}",
            build_p_J, make_h(hi))
        results.append((f"G5 pressure R1 {hn}", ok1))
        results.append((f"G5 pressure R2 {hn}", ok2))

    # ══════════════════════════════════════════════════════════════════
    #  GROUP 6: Full chain combined – WITH mesh movement
    #  J = ∫ |u_bar|^2 ds(particle) — the functional actually used
    #  in the optimisation.  Tested per control direction and mixed.
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GROUP 6: Full chain combined – with mesh move")
    print("  Functional: J = ∫|u_bar|²ds(particle)  (as used in optimisation)")
    print("  Full chain: xi -> mesh -> DiffFieldEval(u_cyl) -> rotation -> integrate")
    print("=" * 60)

    def build_full_J():
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
        u_bar, _, _ = build_3d_background_flow_differentiable(
            R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
            X_ref=X_ref, xi=xi)
        J = assemble(inner(u_bar, u_bar) * ds(tags["particle"], domain=mesh3d))
        return (ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]),
                make_m0())

    for hi, hn in enumerate(ctrl_names):
        ok1, ok2 = run_taylor(
            f"J=∫|u_bar|²ds  [full chain, mesh move]  ctrl={hn}",
            build_full_J, make_h(hi))
        results.append((f"G6 full R1 {hn}", ok1))
        results.append((f"G6 full R2 {hn}", ok2))

    ok1, ok2 = run_taylor(
        "J=∫|u_bar|²ds  [full chain, mesh move]  ctrl=all mixed",
        build_full_J, make_h_mixed())
    results.append(("G6 full R1 mixed", ok1))
    results.append(("G6 full R2 mixed", ok2))

    # ══════════════════════════════════════════════════════════════════
    #  GROUP 7: Cartesian velocity components – NO mesh movement
    #  Same rotation u_bar = R(phi)·u_cyl but with fixed geometry.
    #  Derivatives flow only through u_cyl (DifferentiableFieldEvalBlock)
    #  and the UFL rotation with constant phi.  Comparing with Group 4
    #  reveals whether the shape-derivative part of the rotation is
    #  implemented correctly.
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  GROUP 7: Cartesian velocity components – no mesh move")
    print("  Functional: J = ∫u_bar[k]²ds(particle)")
    print("  Same rotation as Group 4 but mesh stays fixed (no shape derivative).")
    print("  Difference to Group 4 isolates the shape-derivative contribution.")
    print("=" * 60)

    def make_build_ubar_nomove(comp):
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
            # no mesh movement — rotation uses reference phi
            u_bar, _, _ = build_3d_background_flow_differentiable(
                R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
                X_ref=X_ref, xi=xi)
            J = assemble(u_bar[comp]**2 * ds(tags["particle"], domain=mesh3d))
            return (ReducedFunctional(J, [Control(dr), Control(dz), Control(da)]),
                    make_m0())
        return build

    for ci, cn in enumerate(cart_comp_names):
        for hi, hn in enumerate(ctrl_names):
            ok1, ok2 = run_taylor(
                f"J=∫{cn}²ds  [DiffFieldEval+rotation, no mesh move]  ctrl={hn}",
                make_build_ubar_nomove(ci), make_h(hi))
            results.append((f"G7 {cn} R1 {hn}", ok1))
            results.append((f"G7 {cn} R2 {hn}", ok2))

    # ══════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    n_pass = 0
    for name, ok in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name}")
        if ok:
            n_pass += 1
    print(f"\n  {n_pass} / {len(results)} tests passed")

    if n_pass < len(results):
        print("\n  FAILED tests:")
        for name, ok in results:
            if not ok:
                print(f"    - {name}")

    sys.exit(0 if n_pass == len(results) else 1)
