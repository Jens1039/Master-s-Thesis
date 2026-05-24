import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, get_working_tape
from pyadjoint import Block
import numpy as np
import sys
from scipy.spatial import cKDTree

from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole
from config_paper_parameters import *
from nondimensionalization import nondimensionalisation


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

        # The upstream consumer is typically an AssembleBlock built from
        # ``Function(V_vom).interpolate(field_on_source)``, whose test space
        # is ``DualSpace(V_vom)``.  ufl.action() requires the adjoint input
        # to live in that dual space, so we must return a Cofunction —
        # returning a primal Function triggers
        #     "Incompatible function spaces in Action".
        with stop_annotating():
            adj = Cofunction(self.V_vom.dual())
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
            hess_raw = Function(V_h).interpolate(self._hess_expr).dat.data_ro[inv_perm].copy()
            hess_arr = hess_raw.reshape(-1, self.field_dim, 3)
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


    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):

        if tlm_inputs[0] is None:
            return None

        with stop_annotating():
            xi_data = inputs[0].dat.data_ro
            xyz = self._get_CG2_positions(xi_data)
            _, grad_arr, r, _ = self._project_and_eval(xyz, need_grad=True)

            d_xi_CG1 = tlm_inputs[0].dat.data_ro
            d_pos = self.M @ d_xi_CG1

            ds_val = (xyz[:, 0] / r) * d_pos[:, 0] + (xyz[:, 1] / r) * d_pos[:, 1]
            dt_val = d_pos[:, 2]

            d_field = grad_arr[:, :, 0] * ds_val[:, np.newaxis] + grad_arr[:, :, 1] * dt_val[:, np.newaxis]

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


class background_flow_differentiable:

    def __init__(self, R, H, W, Re, comm=None, mesh2d=None):

        self.R   = R
        self.H   = H
        self.W   = W
        self.Re_float  = Re
        if mesh2d is not None:
            self.mesh2d = mesh2d
        else:
            actual_comm = comm if comm is not None else COMM_WORLD
            self.mesh2d = RectangleMesh(128, 128, self.W, self.H, quadrilateral=False, diagonal="crossed", comm=actual_comm)


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

        no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)),"on_boundary")

        nullspace = MixedVectorSpaceBasis(W_mixed,[W_mixed.sub(0), VectorSpaceBasis(constant=True, comm=W_mixed.comm),W_mixed.sub(2)])

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        # Schur fieldsplit is mandatory because the G-constraint is an
        # R-space global (monolithic aij assembly is not supported for
        # mixed systems with R blocks). The earlier ``matfree + AssembledPC
        # + FGMRES`` configuration converged fine in the forward direction
        # but its adjoint (constructed by pyadjoint by inheriting these
        # solver_parameters for a LinearVariationalSolver) was diverging
        # with DIVERGED_LINEAR_SOLVE whenever the adjoint RHS lived in the
        # antisymmetric subspace of u_bar — exposed by
        # shape_optimization_symmetric at z_bif=0.
        #
        # mat_type=nest gives each sub-block its own concrete matrix type
        # (assembled aij for u,p; the G block stays as a Global), so direct
        # LU on the (u,p) sub-block runs without an AssembledPC transpose
        # wrapper. The Schur complement is a single global scalar — fact=full
        # makes the outer KSP trivial in 1-2 GMRES iterations.
        #
        # transpose_nullspace=nullspace is also passed explicitly:
        # solve_init_params copies forward_kwargs verbatim into adj_kwargs,
        # so without this the adjoint LVS would have no information about
        # the constant-pressure null space.
        solver  = NonlinearVariationalSolver(problem,
            nullspace=nullspace, transpose_nullspace=nullspace,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "mat_type": "nest",
                "ksp_type": "gmres",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-50,
                "ksp_max_it": 100,
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": "0,1",
                "pc_fieldsplit_1_fields": "2",
                "fieldsplit_0": {
                    "ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps",
                },
                "fieldsplit_1": {
                    "ksp_type": "preonly",
                    "pc_type": "none",
                },
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

    R_hat, H_hat, W_hat, a_hat, L_c, U_c, Re = nondimensionalisation(R, H, W, a, Q, rho, mu)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R_hat, H_hat, W_hat, L_hat, a_hat, particle_maxh, global_maxh, x_off=0.0, z_off=0.0)

    from background_flow import background_flow, build_3d_background_flow

    # Use ONE shared 2D solution to eliminate source (1)
    G_hat, U_m_hat, u_bar_2d, p_bar_2d = bg.solve_2D_background_flow()

    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)

    print(f"\n{'='*72}")
    print("Isolating sources of u_bar_3d discrepancy")
    print(f"{'='*72}")

    # ===================================================================
    # Layer A: PointEvaluator vs VertexOnlyMesh on the SAME 2D field
    # ===================================================================
    # Both use u_bar_2d from the differentiable solver, same query points.

    with stop_annotating():
        # -- Path 1: PointEvaluator (original method) --
        V_coords = VectorFunctionSpace(mesh3d, "CG", 2)
        coords_func = Function(V_coords).interpolate(SpatialCoordinate(mesh3d))
        xyz = coords_func.dat.data_ro.copy()
        r_3d = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        qu = np.column_stack([np.clip(r_3d - R_hat + 0.5*W_hat, 0, W_hat),
                              np.clip(xyz[:, 2] + 0.5*H_hat, 0, H_hat)])

        mesh2d = u_bar_2d.function_space().mesh()

        pe = PointEvaluator(mesh2d, qu, tolerance=1e-8)
        u_cyl_pe = np.array(pe.evaluate(u_bar_2d))  # (n_CG2, 3): [u_r, u_z, u_theta]

        # -- Path 2: VertexOnlyMesh (differentiable method) --
        vom = VertexOnlyMesh(mesh2d, qu, missing_points_behaviour="error")
        V_vom = VectorFunctionSpace(vom, "DG", 0, dim=3)
        inv_perm = _compute_inv_perm(vom, qu)
        u_cyl_vom = Function(V_vom).interpolate(u_bar_2d).dat.data_ro[inv_perm]

    diff_A = np.max(np.abs(u_cyl_pe - u_cyl_vom))
    rel_A = diff_A / (np.max(np.abs(u_cyl_pe)) + 1e-30)
    print(f"\nLayer A: PointEvaluator vs VertexOnlyMesh (same 2D field, same query pts)")
    print(f"   max |u_cyl_pe - u_cyl_vom| = {diff_A:.4e}  (rel = {rel_A:.4e})")
    for comp, name in enumerate(["u_r", "u_z", "u_theta"]):
        d = np.max(np.abs(u_cyl_pe[:, comp] - u_cyl_vom[:, comp]))
        r = d / (np.max(np.abs(u_cyl_pe[:, comp])) + 1e-30)
        print(f"   {name}: max |diff| = {d:.4e}  (rel = {r:.4e})")

    # ===================================================================
    # Layer B: Different 2D solutions (background_flow vs background_flow_differentiable)
    # ===================================================================
    bg_orig = background_flow(R_hat, H_hat, W_hat, Re)
    G_orig, _, u_bar_orig_2d, _ = bg_orig.solve_2D_background_flow()

    with stop_annotating():
        u_cyl_orig_2d = np.array(PointEvaluator(
            u_bar_orig_2d.function_space().mesh(), qu, tolerance=1e-8
        ).evaluate(u_bar_orig_2d))

    diff_B = np.max(np.abs(u_cyl_pe - u_cyl_orig_2d))
    rel_B = diff_B / (np.max(np.abs(u_cyl_pe)) + 1e-30)
    print(f"\nLayer B: Different 2D solutions (diff_bg vs orig_bg, both via PointEvaluator)")
    print(f"   max |u_cyl_diff2d - u_cyl_orig2d| = {diff_B:.4e}  (rel = {rel_B:.4e})")
    for comp, name in enumerate(["u_r", "u_z", "u_theta"]):
        d = np.max(np.abs(u_cyl_pe[:, comp] - u_cyl_orig_2d[:, comp]))
        r = d / (np.max(np.abs(u_cyl_pe[:, comp])) + 1e-30)
        print(f"   {name}: max |diff| = {d:.4e}  (rel = {r:.4e})")

    # ===================================================================
    # Layer C: CG2 Function vs UFL expression (commutation error)
    # ===================================================================
    # Build u_bar_3d two ways from the SAME u_cyl data (PointEvaluator result):
    #   (a) Cartesian at DOF nodes -> CG2 Function (like original)
    #   (b) u_cyl as CG2 Function + trig as UFL (like differentiable)

    with stop_annotating():
        cos_th = xyz[:, 0] / r_3d
        sin_th = xyz[:, 1] / r_3d
        u_r, u_z, u_th = u_cyl_pe[:, 0], u_cyl_pe[:, 1], u_cyl_pe[:, 2]

        # (a) Cartesian at DOF nodes
        u_cart_fn = Function(V_3d, name="u_cart")
        u_cart_fn.dat.data[:] = np.column_stack([
            u_r * cos_th - u_th * sin_th,
            u_r * sin_th + u_th * cos_th,
            u_z])
        bc_walls = DirichletBC(V_3d, Constant((0, 0, 0)), tags["walls"])
        bc_walls.apply(u_cart_fn)

        # (b) cylindrical as CG2 Function + trig UFL
        u_cyl_fn = Function(V_3d, name="u_cyl_fn")
        u_cyl_fn.dat.data[:] = u_cyl_pe  # [u_r, u_z, u_theta] at CG2 DOFs
        bc_walls.apply(u_cyl_fn)

    x_3d = SpatialCoordinate(mesh3d)
    r_xy = sqrt(x_3d[0]**2 + x_3d[1]**2)
    cos_ufl = x_3d[0] / r_xy
    sin_ufl = x_3d[1] / r_xy

    u_cart_ufl = as_vector([
        cos_ufl * u_cyl_fn[0] - sin_ufl * u_cyl_fn[2],
        sin_ufl * u_cyl_fn[0] + cos_ufl * u_cyl_fn[2],
        u_cyl_fn[1]])

    # Also interpolate the UFL into a CG2 Function to see DOF-level match
    with stop_annotating():
        u_cart_from_ufl = Function(V_3d, name="u_cart_from_ufl")
        u_cart_from_ufl.interpolate(u_cart_ufl)

    diff_C_dof = np.max(np.abs(u_cart_fn.dat.data_ro - u_cart_from_ufl.dat.data_ro))
    rel_C_dof = diff_C_dof / (np.max(np.abs(u_cart_fn.dat.data_ro)) + 1e-30)

    with stop_annotating():
        # L2 difference in quadrature (Function vs UFL, same cylindrical data)
        err_C = sqrt(abs(float(assemble(
            dot(u_cart_fn - u_cart_ufl, u_cart_fn - u_cart_ufl) * dx(degree=6)))))
        norm_C = sqrt(abs(float(assemble(dot(u_cart_fn, u_cart_fn) * dx(degree=6)))))

    print(f"\nLayer C: CG2 Function vs UFL expression (same u_cyl, commutation error)")
    print(f"   DOF-level:  max |diff| = {diff_C_dof:.4e}  (rel = {rel_C_dof:.4e})")
    print(f"   Quadrature: ||fn - ufl||_L2 = {err_C:.6e}  (rel = {err_C/(norm_C+1e-30):.4e})")