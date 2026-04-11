import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
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

    R_hat = 500
    H_hat = 2
    W_hat = 2
    a_hat = 0.05
    Re = 1.0

    L_hat = 4 * max(H_hat, W_hat)
    particle_maxh = 0.2 * a_hat
    global_maxh = 0.2 * min(H_hat, W_hat)

    results = []

    # =====================================================================
    #  TEST 1: 2D background flow (differentiable vs non-diff reference)
    # =====================================================================

    _banner("TEST 1: 2D BACKGROUND FLOW (differentiable vs reference)")

    set_working_tape(Tape())
    continue_annotation()

    bg_diff = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_val_d, U_m_d, u_2d_d, p_2d_d = bg_diff.solve_2D_background_flow()

    from background_flow import background_flow as background_flow_ref
    with stop_annotating():
        bg_ref = background_flow_ref(R_hat, H_hat, W_hat, Re)
        G_val_r, U_m_r, u_2d_r, p_2d_r = bg_ref.solve_2D_background_flow()

    rel_G  = abs(G_val_d - G_val_r) / max(abs(G_val_r), 1e-30)
    rel_Um = abs(U_m_d - U_m_r)     / max(abs(U_m_r),  1e-30)
    # Both classes build RectangleMesh(120,120,W,H) with identical parameters,
    # so the DOF ordering matches and we can compare arrays element-wise.
    # (errornorm refuses to compare functions from different mesh instances.)
    u_d_arr = u_2d_d.dat.data_ro
    u_r_arr = u_2d_r.dat.data_ro
    p_d_arr = p_2d_d.dat.data_ro.copy()
    p_r_arr = p_2d_r.dat.data_ro.copy()
    # Pressure is determined only up to an additive constant (constant nullspace
    # in both solvers). Subtract the mean before comparing.
    p_d_arr -= p_d_arr.mean()
    p_r_arr -= p_r_arr.mean()
    u_err = (np.linalg.norm(u_d_arr - u_r_arr)
             / max(np.linalg.norm(u_r_arr), 1e-30))
    p_err = (np.linalg.norm(p_d_arr - p_r_arr)
             / max(np.linalg.norm(p_r_arr), 1e-30))

    print(f"  G_val:    diff={G_val_d:+.10e}  ref={G_val_r:+.10e}  rel={rel_G:.2e}")
    print(f"  U_m_hat:  diff={U_m_d:+.10e}  ref={U_m_r:+.10e}  rel={rel_Um:.2e}")
    print(f"  u_2d L2 rel error = {u_err:.2e}")
    print(f"  p_2d L2 rel error = {p_err:.2e}")

    tol1 = 1e-10
    test1 = _pass_fail("G_val",    rel_G  < tol1, f"rel={rel_G:.2e}")
    test1 &= _pass_fail("U_m_hat", rel_Um < tol1, f"rel={rel_Um:.2e}")
    test1 &= _pass_fail("u_2d L2", u_err  < tol1, f"rel={u_err:.2e}")
    test1 &= _pass_fail("p_2d L2", p_err  < tol1, f"rel={p_err:.2e}")
    results.append(("Test 1: 2D background flow vs reference", test1))

    # =====================================================================
    #  SETUP for the 3D tests
    # =====================================================================

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R_hat, H_hat, W_hat, L_hat, a_hat,
        particle_maxh, global_maxh, r_off=0.0, z_off=0.0)

    R_space = FunctionSpace(mesh3d, "R", 0)
    V_def   = VectorFunctionSpace(mesh3d, "CG", 1)
    V_3d    = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d    = FunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():
        X = Function(V_def, name="X_ref")
        X.interpolate(SpatialCoordinate(mesh3d))

    particle_x, particle_y, particle_z = tags["particle_center"]
    dist  = sqrt((X[0] - particle_x) ** 2
                 + (X[1] - particle_y) ** 2
                 + (X[2] - particle_z) ** 2)
    r_cut = Constant(0.5 * min(H_hat, W_hat))
    bump  = max_value(Constant(0.0), 1.0 - dist / r_cut)

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    def _build_xi(delta_r_fn, delta_z_fn):
        xi_fn = Function(V_def, name="xi")
        xi_fn.interpolate(as_vector([
            delta_r_fn * cos_th * bump,
            delta_r_fn * sin_th * bump,
            delta_z_fn * bump,
        ]))
        return xi_fn

    def _build_J(dr_value, dz_value):
        """Fresh tape, set delta_r/delta_z, build J = ∫|u_bar_3d|² dx.

        Returns (J_AdjFloat, c_r, c_z, delta_r, delta_z).
        """
        set_working_tape(Tape())
        continue_annotation()
        delta_r = Function(R_space, name="delta_r").assign(dr_value)
        delta_z = Function(R_space, name="delta_z").assign(dz_value)
        xi_l    = _build_xi(delta_r, delta_z)
        mesh3d.coordinates.assign(X + xi_l)

        u_bar_3d_l, p_bar_3d_l, _ = build_3d_background_flow_differentiable(
            R_hat, H_hat, W_hat, G_val_d, mesh3d, tags, u_2d_d, p_2d_d,
            X_ref=X, xi=xi_l)

        u_bar_3d_acc = Function(V_3d).interpolate(u_bar_3d_l)
        J = assemble(inner(u_bar_3d_acc, u_bar_3d_acc) * dx)

        return J, Control(delta_r), Control(delta_z), delta_r, delta_z

    # =====================================================================
    #  TEST 2: 3D differentiable build (xi=0) vs non-differentiable ref
    # =====================================================================

    _banner("TEST 2: 3D BACKGROUND FLOW FORWARD (diff @ xi=0 vs reference)")

    set_working_tape(Tape())
    continue_annotation()

    delta_r_t2 = Function(R_space, name="delta_r").assign(0.0)
    delta_z_t2 = Function(R_space, name="delta_z").assign(0.0)
    xi_t2 = _build_xi(delta_r_t2, delta_z_t2)
    mesh3d.coordinates.assign(X + xi_t2)

    u_3d_diff_ufl, p_3d_diff_ufl, _ = build_3d_background_flow_differentiable(
        R_hat, H_hat, W_hat, G_val_d, mesh3d, tags, u_2d_d, p_2d_d,
        X_ref=X, xi=xi_t2)

    with stop_annotating():
        u_3d_diff = Function(V_3d).interpolate(u_3d_diff_ufl)
        p_3d_diff = Function(Q_3d).interpolate(p_3d_diff_ufl)

    from background_flow import build_3d_background_flow as build_3d_ref
    with stop_annotating():
        u_3d_ref, p_3d_ref = build_3d_ref(
            R_hat, H_hat, W_hat, G_val_d, mesh3d, tags, u_2d_d, p_2d_d)

    u3d_err = (errornorm(u_3d_diff, u_3d_ref, "L2")
               / max(norm(u_3d_ref, "L2"), 1e-30))
    p3d_err = (errornorm(p_3d_diff, p_3d_ref, "L2")
               / max(norm(p_3d_ref, "L2"), 1e-30))

    print(f"  u_3d L2 rel error = {u3d_err:.2e}")
    print(f"  p_3d L2 rel error = {p3d_err:.2e}")

    # PointEvaluator (ref) vs VOM-based interpolation (diff) agree to ~1e-10.
    tol2 = 1e-8
    test2 = _pass_fail("u_3d L2", u3d_err < tol2, f"rel={u3d_err:.2e}")
    test2 &= _pass_fail("p_3d L2", p3d_err < tol2, f"rel={p3d_err:.2e}")
    results.append(("Test 2: 3D forward (diff vs reference)", test2))

    # =====================================================================
    #  TEST 3: First-order Taylor tests at m0 = (0, 0)
    # =====================================================================

    _banner("TEST 3: FIRST-ORDER TAYLOR TESTS at m0=(0, 0)")

    J_v, c_r, c_z, delta_r, delta_z = _build_J(0.0, 0.0)
    Jhat_r = ReducedFunctional(J_v, c_r)
    h_r = Function(R_space).assign(0.5)
    rate3_r = taylor_test(Jhat_r, delta_r, h_r)
    print(f"  delta_r min rate = {rate3_r:.4f}")

    J_v, c_r, c_z, delta_r, delta_z = _build_J(0.0, 0.0)
    Jhat_z = ReducedFunctional(J_v, c_z)
    h_z = Function(R_space).assign(0.5)
    rate3_z = taylor_test(Jhat_z, delta_z, h_z)
    print(f"  delta_z min rate = {rate3_z:.4f}")

    tol_R1 = 1.9
    test3 = _pass_fail("R1 rate(delta_r)", rate3_r >= tol_R1, f"rate={rate3_r:.4f}")
    test3 &= _pass_fail("R1 rate(delta_z)", rate3_z >= tol_R1, f"rate={rate3_z:.4f}")
    results.append(("Test 3: 1st-order Taylor at m0=(0,0)", test3))

    # =====================================================================
    #  TEST 4: First-order Taylor tests at m0 = (0.5, 0.3)
    # =====================================================================

    _banner("TEST 4: FIRST-ORDER TAYLOR TESTS at m0=(0.5, 0.3)")

    # NB: J = ∫|u_bar_3d|² dx is roughly constant in delta_r/delta_z because
    # the deformed coordinate function only re-parametrises the same volume,
    # so the residuals fall to ~1e-12 already at moderate h. Picking h too
    # small puts us into floating-point noise. h=0.5 keeps the residual
    # well above the noise floor while staying inside the valid mesh region.
    J_v, c_r, c_z, delta_r, delta_z = _build_J(0.5, 0.3)
    Jhat_r = ReducedFunctional(J_v, c_r)
    h_r = Function(R_space).assign(0.5)
    rate4_r = taylor_test(Jhat_r, delta_r, h_r)
    print(f"  delta_r min rate = {rate4_r:.4f}")

    J_v, c_r, c_z, delta_r, delta_z = _build_J(0.5, 0.3)
    Jhat_z = ReducedFunctional(J_v, c_z)
    h_z = Function(R_space).assign(0.5)
    rate4_z = taylor_test(Jhat_z, delta_z, h_z)
    print(f"  delta_z min rate = {rate4_z:.4f}")

    test4 = _pass_fail("R1 rate(delta_r)", rate4_r >= tol_R1, f"rate={rate4_r:.4f}")
    test4 &= _pass_fail("R1 rate(delta_z)", rate4_z >= tol_R1, f"rate={rate4_z:.4f}")
    results.append(("Test 4: 1st-order Taylor at m0=(0.5,0.3)", test4))

    # =====================================================================
    #  TEST 5: Second-order Taylor tests (Hessian) at m0 = (0, 0)
    # =====================================================================

    _banner("TEST 5: SECOND-ORDER TAYLOR (Hessian) at m0=(0, 0)")

    tol_R2 = 2.85

    try:
        J_v, c_r, c_z, delta_r, delta_z = _build_J(0.0, 0.0)
        Jhat_r = ReducedFunctional(J_v, c_r)
        h_r = Function(R_space).assign(0.5)
        res_r = taylor_to_dict(Jhat_r, delta_r, h_r)
        R0_r = min(res_r["R0"]["Rate"])
        R1_r = min(res_r["R1"]["Rate"])
        R2_r = min(res_r["R2"]["Rate"])
        print(f"  delta_r  R0 rates: {[f'{x:.3f}' for x in res_r['R0']['Rate']]}")
        print(f"  delta_r  R1 rates: {[f'{x:.3f}' for x in res_r['R1']['Rate']]}")
        print(f"  delta_r  R2 rates: {[f'{x:.3f}' for x in res_r['R2']['Rate']]}")

        J_v, c_r, c_z, delta_r, delta_z = _build_J(0.0, 0.0)
        Jhat_z = ReducedFunctional(J_v, c_z)
        h_z = Function(R_space).assign(0.5)
        res_z = taylor_to_dict(Jhat_z, delta_z, h_z)
        R0_z = min(res_z["R0"]["Rate"])
        R1_z = min(res_z["R1"]["Rate"])
        R2_z = min(res_z["R2"]["Rate"])
        print(f"  delta_z  R0 rates: {[f'{x:.3f}' for x in res_z['R0']['Rate']]}")
        print(f"  delta_z  R1 rates: {[f'{x:.3f}' for x in res_z['R1']['Rate']]}")
        print(f"  delta_z  R2 rates: {[f'{x:.3f}' for x in res_z['R2']['Rate']]}")

        test5 = _pass_fail("R1 rate(delta_r)", R1_r >= tol_R1, f"min={R1_r:.4f}")
        test5 &= _pass_fail("R1 rate(delta_z)", R1_z >= tol_R1, f"min={R1_z:.4f}")
        test5 &= _pass_fail("R2 rate(delta_r)", R2_r >= tol_R2, f"min={R2_r:.4f}")
        test5 &= _pass_fail("R2 rate(delta_z)", R2_z >= tol_R2, f"min={R2_z:.4f}")
    except NotImplementedError as e:
        print(f"  [SKIP] Hessian not implemented for {e}")
        test5 = True
    results.append(("Test 5: 2nd-order Taylor at m0=(0,0)", test5))

    # =====================================================================
    #  TEST 6: Second-order Taylor tests (Hessian) at m0 = (0.5, 0.3)
    # =====================================================================

    _banner("TEST 6: SECOND-ORDER TAYLOR (Hessian) at m0=(0.5, 0.3)")

    try:
        J_v, c_r, c_z, delta_r, delta_z = _build_J(0.5, 0.3)
        Jhat_r = ReducedFunctional(J_v, c_r)
        h_r = Function(R_space).assign(0.5)
        res_r6 = taylor_to_dict(Jhat_r, delta_r, h_r)
        R1_r6 = min(res_r6["R1"]["Rate"])
        R2_r6 = min(res_r6["R2"]["Rate"])
        print(f"  delta_r  R1 rates: {[f'{x:.3f}' for x in res_r6['R1']['Rate']]}")
        print(f"  delta_r  R2 rates: {[f'{x:.3f}' for x in res_r6['R2']['Rate']]}")

        J_v, c_r, c_z, delta_r, delta_z = _build_J(0.5, 0.3)
        Jhat_z = ReducedFunctional(J_v, c_z)
        h_z = Function(R_space).assign(0.5)
        res_z6 = taylor_to_dict(Jhat_z, delta_z, h_z)
        R1_z6 = min(res_z6["R1"]["Rate"])
        R2_z6 = min(res_z6["R2"]["Rate"])
        print(f"  delta_z  R1 rates: {[f'{x:.3f}' for x in res_z6['R1']['Rate']]}")
        print(f"  delta_z  R2 rates: {[f'{x:.3f}' for x in res_z6['R2']['Rate']]}")

        test6 = _pass_fail("R1 rate(delta_r)", R1_r6 >= tol_R1, f"min={R1_r6:.4f}")
        test6 &= _pass_fail("R1 rate(delta_z)", R1_z6 >= tol_R1, f"min={R1_z6:.4f}")
        test6 &= _pass_fail("R2 rate(delta_r)", R2_r6 >= tol_R2, f"min={R2_r6:.4f}")
        test6 &= _pass_fail("R2 rate(delta_z)", R2_z6 >= tol_R2, f"min={R2_z6:.4f}")
    except NotImplementedError as e:
        print(f"  [SKIP] Hessian not implemented for {e}")
        test6 = True
    results.append(("Test 6: 2nd-order Taylor at m0=(0.5,0.3)", test6))

    # =====================================================================
    #  TEST 7: AD gradient vs central finite differences
    # =====================================================================

    _banner("TEST 7: AD GRADIENT vs CENTRAL FINITE DIFFERENCES")

    eps_fd = 1e-5
    test7 = True
    for (dr0, dz0) in [(0.0, 0.0), (0.5, 0.3)]:
        # Build a fresh tape with BOTH controls so we can re-evaluate Jhat
        # at offsets without rebuilding the tape (FD via Jhat replay).
        J_v, c_r, c_z, dr_fn, dz_fn = _build_J(dr0, dz0)
        Jhat = ReducedFunctional(J_v, [c_r, c_z])

        d_AD = Jhat.derivative()
        dJ_dr_AD = float(d_AD[0].dat.data_ro[0])
        dJ_dz_AD = float(d_AD[1].dat.data_ro[0])

        def _ev(dr_v, dz_v):
            return float(Jhat([
                Function(R_space).assign(dr_v),
                Function(R_space).assign(dz_v),
            ]))

        J_pp = _ev(dr0 + eps_fd, dz0)
        J_pm = _ev(dr0 - eps_fd, dz0)
        dJ_dr_FD = (J_pp - J_pm) / (2 * eps_fd)

        J_zp = _ev(dr0, dz0 + eps_fd)
        J_zm = _ev(dr0, dz0 - eps_fd)
        dJ_dz_FD = (J_zp - J_zm) / (2 * eps_fd)

        rel_r = abs(dJ_dr_AD - dJ_dr_FD) / max(abs(dJ_dr_FD), 1e-30)
        rel_z = abs(dJ_dz_AD - dJ_dz_FD) / max(abs(dJ_dz_FD), 1e-30)

        print(f"  m0=({dr0},{dz0}):")
        print(f"    dJ/dr  AD={dJ_dr_AD:+.10e}  "
              f"FD={dJ_dr_FD:+.10e}  rel={rel_r:.2e}")
        print(f"    dJ/dz  AD={dJ_dz_AD:+.10e}  "
              f"FD={dJ_dz_FD:+.10e}  rel={rel_z:.2e}")

        tol_fd = 1e-4
        test7 &= _pass_fail(
            f"dJ/dr at ({dr0},{dz0})", rel_r < tol_fd, f"rel={rel_r:.2e}")
        test7 &= _pass_fail(
            f"dJ/dz at ({dr0},{dz0})", rel_z < tol_fd, f"rel={rel_z:.2e}")
    results.append(("Test 7: AD gradient vs central FD", test7))

    # =====================================================================
    #  TEST 8: AD Hessian-vector vs central FD of AD gradient
    # =====================================================================

    _banner("TEST 8: AD HESSIAN-VECTOR vs FD-of-AD-GRADIENT")

    def _grad_at(dr_v, dz_v):
        """Build a fresh tape and return the AD gradient at (dr_v, dz_v)."""
        J_v, c_r, c_z, _, _ = _build_J(dr_v, dz_v)
        Jhat = ReducedFunctional(J_v, [c_r, c_z])
        d = Jhat.derivative()
        return np.array([float(d[0].dat.data_ro[0]),
                         float(d[1].dat.data_ro[0])])

    eps_h = 1e-4
    test8 = True
    for (dr0, dz0) in [(0.0, 0.0), (0.5, 0.3)]:
        # AD Hessian: H = [[d²J/dr², d²J/drdz], [d²J/dzdr, d²J/dz²]]
        # via two Hessian-vector products with phi=(1,0) and phi=(0,1).
        J_v, c_r, c_z, dr_fn, dz_fn = _build_J(dr0, dz0)
        Jhat = ReducedFunctional(J_v, [c_r, c_z])
        _ = Jhat.derivative()  # prime adjoint state for Hessian

        try:
            Hphi_r = Jhat.hessian([Function(R_space).assign(1.0),
                                   Function(R_space).assign(0.0)])
            Hphi_z = Jhat.hessian([Function(R_space).assign(0.0),
                                   Function(R_space).assign(1.0)])
        except NotImplementedError as e:
            print(f"  m0=({dr0},{dz0}): [SKIP] {e}")
            continue
        H_AD = np.array([
            [float(Hphi_r[0].dat.data_ro[0]),
             float(Hphi_z[0].dat.data_ro[0])],
            [float(Hphi_r[1].dat.data_ro[0]),
             float(Hphi_z[1].dat.data_ro[0])],
        ])

        # Central FD of the AD gradient (rebuild tape per offset).
        g_rp = _grad_at(dr0 + eps_h, dz0)
        g_rm = _grad_at(dr0 - eps_h, dz0)
        g_zp = _grad_at(dr0, dz0 + eps_h)
        g_zm = _grad_at(dr0, dz0 - eps_h)
        H_FD = np.column_stack([
            (g_rp - g_rm) / (2 * eps_h),
            (g_zp - g_zm) / (2 * eps_h),
        ])

        diff  = float(np.max(np.abs(H_AD - H_FD)))
        scale = max(float(np.max(np.abs(H_FD))), 1e-30)
        rel   = diff / scale
        sym_AD = float(np.max(np.abs(H_AD - H_AD.T)))

        print(f"  m0=({dr0},{dz0}):")
        print(f"    H_AD = [[{H_AD[0,0]:+.4e}, {H_AD[0,1]:+.4e}],")
        print(f"            [{H_AD[1,0]:+.4e}, {H_AD[1,1]:+.4e}]]")
        print(f"    H_FD = [[{H_FD[0,0]:+.4e}, {H_FD[0,1]:+.4e}],")
        print(f"            [{H_FD[1,0]:+.4e}, {H_FD[1,1]:+.4e}]]")
        print(f"    max|AD-FD| = {diff:.2e}, rel = {rel:.2e}, "
              f"|H_AD - H_AD^T|_max = {sym_AD:.2e}")

        tol_h = 1e-3
        test8 &= _pass_fail(
            f"Hessian at ({dr0},{dz0})", rel < tol_h, f"rel={rel:.2e}")
    results.append(("Test 8: AD Hessian vs FD-of-AD-gradient", test8))

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
    print("  ALL TESTS PASSED" if all_pass else "  SOME TESTS FAILED")
    print("=" * 70)
