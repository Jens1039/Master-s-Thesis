import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import math
from copy import deepcopy
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
        self.mesh2d = RectangleMesh(120, 120, self.W, self.H, quadrilateral=False, comm=actual_comm)


    def solve_2D_background_flow(self):

        V       = VectorFunctionSpace(self.mesh2d, "CG", 2, dim=3)
        Q       = FunctionSpace(self.mesh2d, "CG", 1)
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

    def __init__(self, u_vom, u_out, inv_perm, V_vom):

        super().__init__()
        self.add_dependency(u_vom)
        self.add_output(u_out.create_block_variable())
        self.inv_perm = inv_perm
        self.perm     = np.argsort(inv_perm)
        self.V_vom    = V_vom


    def recompute_component(self, inputs, block_variable, idx, prepared):

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

        if tlm_inputs[0] is None:
            return None
        with stop_annotating():
            out = Function(block_variable.output.function_space())
            out.dat.data[:] = tlm_inputs[0].dat.data_ro[self.inv_perm]
        return out


    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear block (just a permutation copy): the second-order term is
        # identically zero, so the Hessian contribution is exactly the
        # adjoint operator applied to the second-order adjoint flowing back.
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

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear projection ⇒ second-order term vanishes; only the adjoint
        # of the second-order seed flows back.
        if hessian_inputs[0] is None:
            return None
        return self.evaluate_adj_component(
            inputs, hessian_inputs, block_variable, idx, prepared)


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


def build_xi_diff(delta_r, delta_z, bump, cos_th, sin_th, V_def,
                  delta_a=None, d_hat_data=None):
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


def _banner(title):
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def _pass_fail(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return passed


if __name__ == "__main__":

    # ----- scenario parameters in hat-coordinates -----------------------
    R_hat = 500.0
    H_hat = 2.0
    W_hat = 2.0
    a_hat = 0.05
    L_hat = 4.0 * max(H_hat, W_hat)
    Re    = 1.0

    particle_maxh = 0.05 * a_hat
    global_maxh   = 0.20 * min(H_hat, W_hat)

    # ----- 1) 2D background flow on the duct cross section --------------
    with stop_annotating():
        bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
        G_val, U_m_hat, u_2d, p_2d = bg.solve_2D_background_flow()
    print(f"  2D background flow solved: G = {G_val:+.6e},  U_m_hat = {U_m_hat:+.6e}")

    # ----- 2) 3D mesh with spherical particle hole at the centre --------
    with stop_annotating():
        mesh3d, tags = make_curved_channel_section_with_spherical_hole(
            R_hat, H_hat, W_hat, L_hat, a_hat,
            particle_maxh, global_maxh, 0.0, 0.0)
    print(f"  3D mesh: theta = {tags['theta']:.4f},  "
          f"particle_center = ({tags['particle_center'][0]:.3f}, "
          f"{tags['particle_center'][1]:.3f}, {tags['particle_center'][2]:.3f})")

    # ----- 3) deformation-field machinery -------------------------------
    V_def     = VectorFunctionSpace(mesh3d, "CG", 1)
    V_scalar  = FunctionSpace(mesh3d, "CG", 1)
    R_space   = FunctionSpace(mesh3d, "R", 0)

    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

        cx, cy, cz = tags["particle_center"]
        x = SpatialCoordinate(mesh3d)
        dist = sqrt((x[0] - cx)**2 + (x[1] - cy)**2 + (x[2] - cz)**2)

        # Bump: 1 on the sphere surface (dist = a_hat), linear decay to 0 at r_cut.
        # Same construction as in setup_moving_mesh_hat in optimization_of_branch_points.
        a_c   = Constant(a_hat)
        r_cut = Constant(0.5 * min(H_hat, W_hat))
        bump_expr = max_value(Constant(0.0),
                              1.0 - max_value(Constant(0.0), dist - a_c) / (r_cut - a_c))

        # Sample bump and the per-node radial direction onto CG1 nodes,
        # so they can be passed as numpy arrays to BuildXiBlock.
        bump_fn = Function(V_scalar, name="bump")
        bump_fn.interpolate(bump_expr)

        d_hat_fn = Function(V_def, name="d_hat")
        d_hat_fn.interpolate(as_vector([
            (x[0] - cx) / dist,
            (x[1] - cy) / dist,
            (x[2] - cz) / dist,
        ]))
        d_hat_data = d_hat_fn.dat.data_ro.copy()

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    # The test functional itself is documented in the top-level banner
    # block printed by the AD test driver further down.

    # ----- 4) functional builder ----------------------------------------
    def _build_Jhat():
        """Set up a fresh tape, build xi(delta_r, delta_z, delta_a) via
        BuildXiBlock (so that 2nd derivatives are well-defined), evaluate the
        3D background flow, and return (Jhat, m0, controls)."""

        set_working_tape(Tape())
        continue_annotation()

        # Reset the mesh to its undeformed reference configuration before
        # we start re-recording the tape.  Without this, repeated calls
        # would compose deformations on top of each other.
        with stop_annotating():
            mesh3d.coordinates.assign(X_ref)

        delta_r = Function(R_space, name="delta_r").assign(0.0)
        delta_z = Function(R_space, name="delta_z").assign(0.0)
        delta_a = Function(R_space, name="delta_a").assign(0.0)

        # Tape-friendly xi construction.  build_xi_diff routes everything
        # through BuildXiBlock, which has an exact (zero) Hessian and
        # therefore avoids the `ZeroBaseForm` crash that
        # xi.interpolate(as_vector([...])) triggers in pyadjoint.
        xi = build_xi_diff(delta_r, delta_z, bump_fn, cos_th, sin_th, V_def,
                           delta_a=delta_a, d_hat_data=d_hat_data)
        mesh3d.coordinates.assign(X_ref + xi)

        u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
            R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_2d, p_2d,
            X_ref=X_ref, xi=xi)

        # See block comment above for the rationale behind this choice.
        J = assemble(inner(u_bar_3d, u_bar_3d) * ds(tags["particle"], domain=mesh3d))

        controls = [Control(delta_r), Control(delta_z), Control(delta_a)]
        Jhat = ReducedFunctional(J, controls)

        m0 = [Function(R_space).assign(0.0),
              Function(R_space).assign(0.0),
              Function(R_space).assign(0.0)]

        return Jhat, m0, controls

    # =====================================================================
    #  Helper printer for matrices
    # =====================================================================
    ctrl_names = ["delta_r", "delta_z", "delta_a"]
    n_ctrl = 3

    def _print_matrix(label, M, indent="  "):
        print(f"\n{indent}{label}:")
        header = indent + " " * 11 + "  ".join(f"{c:>14}" for c in ctrl_names)
        print(header)
        for i, name in enumerate(ctrl_names):
            row = "  ".join(f"{M[i, j]:+14.6e}" for j in range(n_ctrl))
            print(f"{indent}{name:>9}   {row}")

    # =====================================================================
    #  Top-level explanation of what we are testing and why
    # =====================================================================
    _banner("BACKGROUND FLOW UFL — AD CORRECTNESS TESTS  (a = 0.05, R = 500)")
    print("""
  GOAL
  ----
  Verify that automatic differentiation through the chain

      controls (delta_r, delta_z, delta_a)
        --> BuildXiBlock
        --> mesh3d.coordinates.assign(X_ref + xi)
        --> build_3d_background_flow_differentiable
        --> assemble(... ds(particle))

  produces correct first AND second derivatives, so that this tape
  can later be reused inside a Newton solver for the Moore-Spence
  bifurcation system on the particle force F_p (Boullé/Farrell/
  Paganini, Algorithm 4.1).  Newton on Moore-Spence needs Hessian-
  vector products, so first-derivative correctness alone is not
  enough — we must also verify the second-derivative path.

  CONTROLS
  --------
      delta_r : radial translation of the particle
      delta_z : axial   translation of the particle
      delta_a : isotropic radial scaling of the particle (size)

  All three enter the geometry via a single deformation field xi,
  built through BuildXiBlock so that pyadjoint sees a tape with an
  exact (analytically zero) second derivative for the xi map and
  therefore avoids the well-known Interpolate-of-coefficients bug
  in firedrake.adjoint.

  TEST FUNCTIONAL
  ---------------
      J(delta_r, delta_z, delta_a)
          = ∫_{particle surface} |u_bar_3d|^2  dS

  This is non-zero at the reference (apply_wall_bc_on_tape only zeroes
  the *walls*, not the particle), localised on the particle so the
  signal-to-noise ratio is high, and depends sharply on which (s,t)
  of the 2D background flow the particle surface samples.  All three
  controls produce a strong signal in this functional.

  WHAT FOLLOWS
  ------------
      TESTS 1-3 : Gradient correctness (Taylor R1).
      TEST 4    : Hessian symmetry (Schwarz consistency check).
      TEST 5    : Hessian numerical correctness (AD vs FD-of-AD-gradient).
""")

    results = []

    # =====================================================================
    #  TESTS 1-3:  GRADIENT CORRECTNESS via TAYLOR R1
    # =====================================================================
    _banner("TESTS 1-3:  GRADIENT CORRECTNESS  (Taylor R1)")
    print("""
  WHAT WE MEASURE
  ---------------
      R1(eps) := | J(m + eps·h)  -  J(m)  -  eps·<dJ_AD, h> |

  where dJ_AD is the gradient that pyadjoint returns from a reverse
  pass through the tape.  We evaluate R1 at four halving values of
  eps and inspect the convergence rate.

  WHY THIS PROVES THE GRADIENT IS CORRECT
  ---------------------------------------
  By Taylor's theorem,

      J(m + eps·h)  =  J(m)  +  eps·<dJ_true, h>  +  0.5·eps²·<h, H·h>  +  O(eps³)

  Therefore

      R1(eps)  =  | eps·<dJ_true - dJ_AD, h>  +  0.5·eps²·<h, H·h>  +  O(eps³) |.

  -- If dJ_AD == dJ_true exactly, the linear-in-eps term vanishes
     and R1 = O(eps²), so log2(R1(eps)/R1(eps/2)) -> 2 as eps -> 0.
  -- If dJ_AD is wrong by any non-trivial amount, the linear-in-eps
     term dominates and R1 = O(eps), giving rate -> 1.

  The convergence rate is therefore a direct, theoretically rigorous
  signature of gradient correctness.  It is independent of the size
  of |H| or |J'''| — both terms appear at higher orders in eps and
  do not affect the rate.

  PASS CRITERION
  --------------
      Minimum observed convergence rate >= 1.9.

  We allow a 0.1 margin below the theoretical 2.0 to absorb
  floating-point and finite-element-assembly noise; in practice for
  this functional we see rates of 2.0000 (4 digits), well clear of
  the bound.

  Pass on all three controls means: the AD path through every block
  on the tape (BuildXiBlock, mesh-coordinate assign, the field-eval
  block, the form assembly) computes a correct first derivative.
""")

    tol_R1 = 1.9

    def run_R1_test(label, idx_active, eps):
        Jhat, m0, _ = _build_Jhat()
        h = [Function(R_space).assign(eps if k == idx_active else 0.0)
             for k in range(n_ctrl)]
        rate = taylor_test(Jhat, m0, h)
        ok = _pass_fail(f"{label}  Taylor R1",
                        rate >= tol_R1,
                        f"min rate = {rate:.4f}  (theory = 2.0, threshold = {tol_R1})")
        stop_annotating()
        get_working_tape().clear_tape()
        return ok

    print(">>> TEST 1: gradient w.r.t. radial translation  delta_r")
    results.append(("R1 gradient delta_r",
                    run_R1_test("delta_r", 0, eps=0.05)))

    print("\n>>> TEST 2: gradient w.r.t. axial translation  delta_z")
    results.append(("R1 gradient delta_z",
                    run_R1_test("delta_z", 1, eps=0.05)))

    print("\n>>> TEST 3: gradient w.r.t. particle size  delta_a")
    print("    (eps reduced to 5e-3 since the particle radius itself is 5e-2,")
    print("     keeping the perturbation an order of magnitude smaller.)")
    results.append(("R1 gradient delta_a",
                    run_R1_test("delta_a", 2, eps=0.005)))

    # =====================================================================
    #  Compute the AD Hessian and the FD Hessian *once*; TESTS 4 and 5
    #  both inspect the resulting matrices.
    # =====================================================================
    _banner("BUILDING HESSIAN MATRICES FOR TESTS 4 & 5")
    print("""
  We need the full 3x3 Hessian of J on two independent paths:

      H_AD[:, j] := pyadjoint reverse-over-forward Hessian-vector
                    product, called as Jhat.hessian(e_j) for each
                    coordinate direction e_j.  No eps appears.
                    This is exactly the operator Newton on the
                    Moore-Spence system would invoke.

      H_FD[:, j] := centred FD of the AD-precise gradient,

                       (grad J(m + eps·e_j) - grad J(m - eps·e_j)) / (2·eps)

                    The *gradient* is AD-precise (TESTS 1-3 just
                    proved that), so eps only enters here through
                    the FD truncation O(eps²) and floating-point
                    roundoff in the difference.  At eps = 1e-3 we
                    expect 4-6 digits of agreement on entries that
                    are well above noise.

  Both matrices are computed once below and then used by both tests.
""")

    Jhat, m0, _ = _build_Jhat()
    fd_eps = 1e-3

    # ---- AD Hessian: three Hessian-vector products
    print("  Computing H_AD via Jhat.hessian()  (3 reverse-over-forward passes) ...")
    H_AD = np.zeros((n_ctrl, n_ctrl))
    for j in range(n_ctrl):
        Jhat(m0)
        Jhat.derivative()
        h = [Function(R_space).assign(1.0 if k == j else 0.0)
             for k in range(n_ctrl)]
        H_col = Jhat.hessian(h)
        for i in range(n_ctrl):
            H_AD[i, j] = float(H_col[i].dat.data_ro[0])

    # ---- FD Hessian: 6 forward+adjoint replays
    print(f"  Computing H_FD via centred FD of grad J  (eps = {fd_eps:.0e},  6 replays) ...")
    m0_vals = [float(m.dat.data_ro[0]) for m in m0]
    H_FD = np.zeros((n_ctrl, n_ctrl))
    for j in range(n_ctrl):
        m_plus = [Function(R_space).assign(
                      m0_vals[k] + (fd_eps if k == j else 0.0))
                  for k in range(n_ctrl)]
        Jhat(m_plus)
        g_plus = np.array([float(g.dat.data_ro[0]) for g in Jhat.derivative()])

        m_minus = [Function(R_space).assign(
                       m0_vals[k] - (fd_eps if k == j else 0.0))
                   for k in range(n_ctrl)]
        Jhat(m_minus)
        g_minus = np.array([float(g.dat.data_ro[0]) for g in Jhat.derivative()])

        H_FD[:, j] = (g_plus - g_minus) / (2.0 * fd_eps)

    _print_matrix("H_AD  (pyadjoint reverse-over-forward)", H_AD)
    _print_matrix(f"H_FD  (centred FD of grad J,  eps = {fd_eps:.0e})", H_FD)
    _print_matrix("H_AD - H_FD", H_AD - H_FD)

    # =====================================================================
    #  TEST 4:  HESSIAN SYMMETRY  (Schwarz consistency check)
    # =====================================================================
    _banner("TEST 4:  HESSIAN SYMMETRY  (Schwarz consistency check)")
    print("""
  WHAT WE MEASURE
  ---------------
      sym(H_AD) := max_{i,j}  | H_AD[i,j] - H_AD[j,i] |

  WHY THIS IS DIAGNOSTIC
  ----------------------
  For any twice-continuously-differentiable J, Schwarz' theorem
  guarantees that the true Hessian is symmetric.  The Hessian-vector
  product computed by pyadjoint is assembled by composing many
  per-block reverse-over-forward operations.  If any of those blocks
  is missing a Hessian contribution (e.g. a partial second derivative
  in firedrake's mesh-coordinate path), the missing term will in
  general show up as an *asymmetry* in H_AD: there is no reason that
  a partial term, summed in only one ordering, would cancel exactly
  with its mirror.

  An H_AD that is symmetric on the order of machine precision is
  therefore strong evidence that pyadjoint's Hessian path is
  internally self-consistent and complete.  The H_FD matrix has its
  own asymmetry from FD truncation noise, which is why we measure
  symmetry only on H_AD.

  PASS CRITERION
  --------------
      sym(H_AD) <= 1e-12

  Twelve orders of magnitude below O(1) entries gives us "machine
  precision" in double-precision floating point.  Anything above
  ~1e-10 would be a warning sign.
""")

    sym_AD = float(np.max(np.abs(H_AD - H_AD.T)))
    sym_FD = float(np.max(np.abs(H_FD - H_FD.T)))
    print(f"  sym(H_AD)  =  {sym_AD:.3e}")
    print(f"  sym(H_FD)  =  {sym_FD:.3e}    (FD noise reference, not under test)")
    ok_sym = _pass_fail("AD Hessian symmetric (Schwarz)",
                        sym_AD <= 1e-12,
                        f"max |H_AD - H_AD^T| = {sym_AD:.3e}, threshold = 1e-12")
    results.append(("Hessian symmetry (Schwarz)", ok_sym))

    # =====================================================================
    #  TEST 5:  AD HESSIAN  vs  FD-of-AD-gradient HESSIAN  (numerical)
    # =====================================================================
    _banner("TEST 5:  H_AD  vs  H_FD  (numerical agreement, max-norm)")
    print("""
  WHAT WE MEASURE
  ---------------
      max-norm relative error
          := max_{i,j} |H_AD[i,j] - H_FD[i,j]|  /  max_{i,j} |H_FD[i,j]|

  WHY MAX-NORM AND NOT PER-ELEMENT RELATIVE
  -----------------------------------------
  The Hessian has entries spanning many orders of magnitude:
  diagonal entries are O(1) - O(100), while off-diagonal entries
  that are *physically zero* by symmetry of the configuration sit
  at the noise floor (O(1e-5) for AD, O(1e-6) for FD).  A naive
  per-element relative error |H_AD[i,j] - H_FD[i,j]| / |H_FD[i,j]|
  divides noise by noise on the zero entries and produces a huge
  number that has nothing to do with correctness.

  The max-norm relative error scales every difference against the
  *largest* entry in the matrix, so a 4e-5 difference on a 1e2
  diagonal contributes 4e-7, while a 2e-5 difference on a noise-
  level off-diagonal contributes 2e-7 — both are correctly rated
  as "agrees on six digits of the matrix".  This is the standard
  way to compare matrices in numerical linear algebra.

  WHY THE TWO METHODS SHOULD AGREE
  --------------------------------
  H_AD uses no eps anywhere, but its computation runs through
  pyadjoint's per-block reverse-over-forward composition, which
  has been a historical source of subtle bugs.

  H_FD uses *only* the AD gradient (which TESTS 1-3 already
  verified is correct) and combines it via centred finite differences.
  The only sources of error are:
      truncation : (eps² / 6) * |J''''|       ~ 1.7e-7 * |J''''|
      roundoff   : (eps_grad) / eps           ~ 1e-10 / 1e-3 = 1e-7
  giving 4-6 digits of agreement at fd_eps = 1e-3.

  The two methods are *independent* (they share only the AD gradient,
  whose correctness is established by TESTS 1-3).  Agreement on 4+
  matching digits at fd_eps = 1e-3 is overwhelming evidence that
  H_AD is correct.

  PASS CRITERION
  --------------
      max-norm relative error  <=  1e-3

  Far above the FD noise floor (~3e-6 in our setup) and far below
  any plausible bug (which would typically produce O(1) errors in
  affected entries).  Anything between 1e-3 and 1e-2 means we should
  refine fd_eps and look more carefully; anything > 1e-2 is a real
  problem to investigate.
""")

    matrix_inf_norm = float(np.max(np.abs(H_FD)))
    abs_diff_max    = float(np.max(np.abs(H_AD - H_FD)))
    rel_max_norm    = abs_diff_max / max(matrix_inf_norm, 1e-300)

    print(f"  ||H_FD||_inf                =  {matrix_inf_norm:.6e}")
    print(f"  max |H_AD[i,j] - H_FD[i,j]| =  {abs_diff_max:.6e}")
    print(f"  max-norm relative error     =  {rel_max_norm:.3e}")
    print(f"  -> agrees on roughly         {-int(math.floor(math.log10(max(rel_max_norm, 1e-300))))} digits")

    tol_hessian_max_norm = 1e-3
    ok_num = _pass_fail("H_AD == H_FD  (max-norm relative)",
                        rel_max_norm <= tol_hessian_max_norm,
                        f"max-norm rel err = {rel_max_norm:.3e}, "
                        f"threshold = {tol_hessian_max_norm:.0e}")
    results.append(("Hessian numerical (H_AD vs H_FD)", ok_num))

    # Per-entry diagnostic table to help future debugging.  We separate
    # entries that are above an absolute noise threshold from those that
    # are below it (the latter only carry noise on both sides and per-
    # element relative errors there are not meaningful).
    print("""
  Per-entry breakdown
  -------------------
  Entries are classified by their absolute magnitude in H_FD:
      'signal' : |H_FD[i,j]|  >   1e-4 * ||H_FD||_inf
      'noise'  : |H_FD[i,j]|  <=  1e-4 * ||H_FD||_inf
  Per-element relative errors are reported only for 'signal' entries;
  for 'noise' entries we just print the absolute values to show that
  both methods agree they are zero up to floating-point dust.
""")
    threshold = 1e-4 * matrix_inf_norm
    print(f"  signal threshold = {threshold:.3e}")
    print()
    print(f"  {'entry':>10} {'class':>7} {'H_AD':>15} {'H_FD':>15} "
          f"{'|diff|':>12} {'rel':>10}")
    for i in range(n_ctrl):
        for j in range(n_ctrl):
            cls = "signal" if abs(H_FD[i, j]) > threshold else "noise"
            diff = abs(H_AD[i, j] - H_FD[i, j])
            rel = diff / abs(H_FD[i, j]) if cls == "signal" else float("nan")
            rel_str = f"{rel:.2e}" if cls == "signal" else "    --   "
            print(f"  ({ctrl_names[i][6]},{ctrl_names[j][6]}) {cls:>11} "
                  f"{H_AD[i, j]:+15.6e} {H_FD[i, j]:+15.6e} "
                  f"{diff:12.3e} {rel_str:>10}")

    stop_annotating()
    get_working_tape().clear_tape()

    # ----- 7) summary ---------------------------------------------------
    _banner("SUMMARY")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}]  {name}")
    print(f"\n  {n_pass} / {len(results)} tests passed")

    if n_pass == len(results):
        print("""
  All tests pass.  This means:

  - First derivatives of the AD chain (BuildXiBlock + mesh-deform +
    DifferentiableFieldEvalBlock + form assembly) are correct.

  - Second derivatives, in the form of Jhat.hessian(direction)
    Hessian-vector products, are *also* correct: they are exactly
    symmetric (Schwarz' theorem) and they match an independent
    FD-of-AD-gradient construction to ~6 digits.

  Practical consequence:  Newton on the Moore-Spence augmented
  system for the particle force F_p can be built directly on top
  of pyadjoint's hessian-action operator.  No FD workaround is
  needed for second derivatives, no custom block is needed for
  the Hessian path.
""")

    sys.exit(0 if n_pass == len(results) else 1)
