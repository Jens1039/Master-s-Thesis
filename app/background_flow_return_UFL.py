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
from pyadjoint import Block, get_working_tape, set_working_tape, Tape, ReducedFunctional, Control, taylor_test
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


# ---------------------------------------------------------------------------
#  Original VOM transfer (kept for pressure, which does not affect forces)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Wall-BC block (unchanged)
# ---------------------------------------------------------------------------

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

    def evaluate_adj_component(self, inputs, adj_inputs,
                               block_variable, idx, prepared=None):
        with stop_annotating():
            adj = Function(self.V)
            adj.dat.data[:] = self._apply_projection(adj_inputs[0].dat.data_ro)
        return adj

    def evaluate_tlm_component(self, inputs, tlm_inputs,
                               block_variable, idx, prepared=None):
        if tlm_inputs[0] is None:
            return None
        with stop_annotating():
            out = Function(block_variable.output.function_space())
            out.dat.data[:] = self._apply_projection(tlm_inputs[0].dat.data_ro)
        return out


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


# ---------------------------------------------------------------------------
#  CG1 → CG2 interpolation matrix  (precomputed, sparse, per-scalar-DOF)
# ---------------------------------------------------------------------------

def _build_CG1_to_CG2_map(mesh3d):
    """
    Build a sparse (n_CG2_scalar, n_CG1_scalar) interpolation matrix M
    such that  ``M @ f_CG1 == interpolate(f_CG1, CG2)``  for any scalar
    CG1 function.  For vector data stored as ``(n_scalar_dofs, 3)`` the
    same matrix applies independently to each column.

    The local element matrix is determined once (from cell 0) and then
    assembled globally using the cell-node maps.
    """
    from scipy.sparse import coo_matrix

    V1 = FunctionSpace(mesh3d, "CG", 1)
    V2 = FunctionSpace(mesh3d, "CG", 2)
    n1 = V1.dof_count
    n2 = V2.dof_count

    cmap1 = V1.cell_node_map().values                      # (n_cells, dpc1)
    cmap2 = V2.cell_node_map().values                      # (n_cells, dpc2)
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

    M_sum = coo_matrix((vals,             (rows, cols)), shape=(n2, n1)).tocsr()
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


# ---------------------------------------------------------------------------
#  NEW: Differentiable 2D → 3D field evaluation
#
#  The block takes *xi* (CG1 mesh displacement, on the tape) as its sole
#  dependency and internally
#    1. computes CG2 node positions via  M @ (X_ref + xi)
#    2. projects them to the 2D cross-section
#    3. evaluates field_2d via a fresh VOM
#  The adjoint propagates through  ∇field_2d ⊗ ∂(s,t)/∂(x,y,z) ⊗ M^T
#  back to adj_xi on CG1, avoiding any annotated cross-space interpolation.
# ---------------------------------------------------------------------------

class DifferentiableFieldEvalBlock(Block):
    """
    Evaluate a 2D field at 3D positions projected to the cross-section.

    Dependency:  xi  (CG1 vector Function – mesh displacement)
    Output:      field_3d  (CG2 vector or scalar Function)

    The CG1→CG2 DOF mapping is handled by a precomputed sparse matrix M
    so that no Firedrake cross-space interpolation appears on the tape.

    Projection:
        s = sqrt(x²+y²) − R + W/2     (clipped to [0, W])
        t = z + H/2                    (clipped to [0, H])

    Derivatives of the projection:
        ∂s/∂x = x/r,   ∂s/∂y = y/r,   ∂s/∂z = 0
        ∂t/∂x = 0,     ∂t/∂y = 0,     ∂t/∂z = 1
    (zero when clipping is active)
    """

    def __init__(self, xi_fn, out_fn, X_ref_data, field_2d,
                 R, W, H, mesh2d, V_def, field_dim, M):
        super().__init__()
        self.add_dependency(xi_fn)
        self.add_output(out_fn.create_block_variable())

        self.X_ref_data = X_ref_data        # numpy (n_CG1, 3)
        self.field_2d   = field_2d
        self.R          = R
        self.W          = W
        self.H          = H
        self.mesh2d     = mesh2d
        self.V_def      = V_def             # CG1 vector space (for adjoint)
        self.field_dim  = field_dim
        self.M          = M                 # sparse (n_CG2, n_CG1)
        self.MT         = M.T.tocsr()       # transpose for adjoint

        if field_dim > 1:
            comps = []
            for k in range(field_dim):
                for m in range(2):
                    comps.append(Dx(field_2d[k], m))
            self._grad_expr = as_vector(comps)
            self._grad_dim  = 2 * field_dim
        else:
            self._grad_expr = as_vector([Dx(field_2d, 0), Dx(field_2d, 1)])
            self._grad_dim  = 2

    # ------------------------------------------------------------------ #
    def _get_CG2_positions(self, xi_data):
        return self.M @ (self.X_ref_data + xi_data)      # (n_CG2, 3)

    # ------------------------------------------------------------------ #
    def _project_and_eval(self, xyz, need_grad=False):
        """Project 3D positions → 2D cross-section, evaluate via VOM."""
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

        return vals, grad_arr, r

    # ------------------------------------------------------------------ #
    def recompute_component(self, inputs, block_variable, idx, prepared):
        out = block_variable.output
        with stop_annotating():
            xyz = self._get_CG2_positions(inputs[0].dat.data_ro)
            vals, _, _ = self._project_and_eval(xyz)
            out.dat.data[:] = vals
        return out

    # ------------------------------------------------------------------ #
    def evaluate_adj_component(self, inputs, adj_inputs,
                               block_variable, idx, prepared=None):
        if adj_inputs[0] is None:
            return Cofunction(self.V_def.dual())

        with stop_annotating():
            xi_data = inputs[0].dat.data_ro
            xyz = self._get_CG2_positions(xi_data)
            _, grad_arr, r = self._project_and_eval(xyz, need_grad=True)

            adj_out = np.asarray(adj_inputs[0].dat.data_ro)
            if adj_out.ndim == 1:
                adj_out = adj_out[:, np.newaxis]

            du_ds = grad_arr[:, :, 0]                       # (n_CG2, d)
            du_dt = grad_arr[:, :, 1]
            s_term = np.sum(du_ds * adj_out, axis=1)        # (n_CG2,)
            t_term = np.sum(du_dt * adj_out, axis=1)

            cos_phi = xyz[:, 0] / r
            sin_phi = xyz[:, 1] / r

            adj_pos_CG2 = np.zeros_like(xyz)                # (n_CG2, 3)
            adj_pos_CG2[:, 0] = cos_phi * s_term
            adj_pos_CG2[:, 1] = sin_phi * s_term
            adj_pos_CG2[:, 2] = t_term

            adj_xi_CG1 = self.MT @ adj_pos_CG2              # (n_CG1, 3)

            adj = Cofunction(self.V_def.dual())
            adj.dat.data[:] = adj_xi_CG1
        return adj

    # ------------------------------------------------------------------ #
    def evaluate_tlm_component(self, inputs, tlm_inputs,
                               block_variable, idx, prepared=None):
        if tlm_inputs[0] is None:
            return None

        with stop_annotating():
            xi_data = inputs[0].dat.data_ro
            xyz = self._get_CG2_positions(xi_data)
            _, grad_arr, r = self._project_and_eval(xyz, need_grad=True)

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


def differentiable_field_eval(xi, X_ref, field_2d, R, W, H,
                              V_def, V_3d, field_dim, M):
    """
    Evaluate *field_2d* at 3D positions projected to the 2D cross-section.

    The returned Function is linked to *xi* (CG1 mesh displacement) on the
    pyadjoint tape.  The CG1→CG2 DOF mapping is done via the precomputed
    sparse matrix *M*, avoiding any annotated cross-space interpolation.
    """
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


# ---------------------------------------------------------------------------
#  Build 3D background flow  (MODIFIED – accepts X_ref, xi for diff. path)
# ---------------------------------------------------------------------------

def build_3d_background_flow_differentiable(
        R, H, W, G_val, mesh3d, tags, u_bar_2d, p_bar_2d,
        X_ref=None, xi=None):
    """
    Construct the 3D background flow from the 2D solution.

    When *X_ref* and *xi* are provided the velocity transfer uses the
    differentiable path: the custom Block depends directly on *xi* (CG1)
    and internally computes CG2 positions via a precomputed sparse matrix,
    so that pyadjoint can propagate derivatives through the 2D→3D sampling
    without any annotated cross-space interpolation.

    When they are ``None`` the original (non-differentiable) VOM transfer
    is used – suitable for forward-only evaluation.
    """
    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d = FunctionSpace(mesh3d, "CG", 1)

    x3d   = SpatialCoordinate(mesh3d)
    r_xy  = sqrt(x3d[0] ** 2 + x3d[1] ** 2)
    cos_ufl   = x3d[0] / r_xy
    sin_ufl   = x3d[1] / r_xy
    theta_ufl = atan2(x3d[1], x3d[0])

    mesh2d = u_bar_2d.function_space().mesh()

    # ── velocity ──────────────────────────────────────────────────────
    if X_ref is not None and xi is not None:
        V_def = xi.function_space()
        M = _build_CG1_to_CG2_map(mesh3d)
        u_cyl_3d = differentiable_field_eval(
            xi, X_ref, u_bar_2d, R, W, H, V_def, V_3d, 3, M)
    else:
        with stop_annotating():
            V_coords = VectorFunctionSpace(mesh3d, "CG", 2)
            coords_func_u = Function(V_coords).interpolate(
                SpatialCoordinate(mesh3d))
            xyz_nodes_u = coords_func_u.dat.data_ro.copy()

            r_3d = np.sqrt(xyz_nodes_u[:, 0] ** 2
                           + xyz_nodes_u[:, 1] ** 2)
            qu = np.column_stack([
                np.clip(r_3d - R + 0.5 * W, 0, W),
                np.clip(xyz_nodes_u[:, 2] + 0.5 * H, 0, H),
            ])

            vom_u      = VertexOnlyMesh(mesh2d, qu,
                                        missing_points_behaviour="error")
            V_vom_u    = VectorFunctionSpace(vom_u, "DG", 0, dim=3)
            inv_perm_u = _compute_inv_perm(vom_u, qu)

        u_vom    = Function(V_vom_u).interpolate(u_bar_2d)
        u_cyl_3d = vom_transfer(u_vom, V_3d, inv_perm_u, V_vom_u)

    # ── pressure (non-differentiable – not used in force computation) ─
    with stop_annotating():
        cp = Function(VectorFunctionSpace(mesh3d, "CG", 1))
        cp.interpolate(SpatialCoordinate(mesh3d))
        xyz_p = cp.dat.data_ro.copy()
        r_p   = np.sqrt(xyz_p[:, 0]**2 + xyz_p[:, 1]**2)

        qp = np.column_stack([
            np.clip(r_p - R + 0.5 * W, 0, W),
            np.clip(xyz_p[:, 2] + 0.5 * H, 0, H),
        ])
        vom_p      = VertexOnlyMesh(mesh2d, qp,
                                    missing_points_behaviour="error")
        V_vom_p    = FunctionSpace(vom_p, "DG", 0)
        inv_perm_p = _compute_inv_perm(vom_p, qp)

    p_vom    = Function(V_vom_p).interpolate(p_bar_2d)
    p_cyl_3d = vom_transfer(p_vom, Q_3d, inv_perm_p, V_vom_p)

    # ── wall BC + cylindrical → Cartesian rotation ────────────────────
    u_cyl_3d = apply_wall_bc_on_tape(u_cyl_3d, V_3d, tags["walls"])

    u_bar_3d = as_vector([
        cos_ufl * u_cyl_3d[0] - sin_ufl * u_cyl_3d[2],
        sin_ufl * u_cyl_3d[0] + cos_ufl * u_cyl_3d[2],
        u_cyl_3d[1],
    ])

    p_bar_3d = p_cyl_3d - Constant(G_val * R) * theta_ufl

    return u_bar_3d, p_bar_3d, u_cyl_3d


# ---------------------------------------------------------------------------
#  Self-test: taylor_test on ∫ |u_bar_3d|² dx  w.r.t.  delta_r / delta_z
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    R_hat = 5
    H_hat = 2
    W_hat = 2
    a_hat = 0.05
    Re = 1.0

    L_hat = 4 * max(H_hat, W_hat)
    particle_maxh = 0.2 * a_hat
    global_maxh = 0.2 * min(H_hat, W_hat)

    set_working_tape(Tape())
    continue_annotation()

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R_hat, H_hat, W_hat, L_hat, a_hat,
        particle_maxh, global_maxh, r_off=0.0, z_off=0.0)

    R_space = FunctionSpace(mesh3d, "R", 0)
    delta_r = Function(R_space, name="delta_r").assign(0.0)
    delta_z = Function(R_space, name="delta_z").assign(0.0)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    dist = sqrt((X_ref[0] - cx) ** 2 + (X_ref[1] - cy) ** 2
                + (X_ref[2] - cz) ** 2)

    r_cut = Constant(0.5 * min(H_hat, W_hat))
    bump = max_value(Constant(0.0), 1.0 - dist / r_cut)

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    xi = Function(V_def, name="xi")
    xi.interpolate(as_vector([
        delta_r * cos_th * bump,
        delta_r * sin_th * bump,
        delta_z * bump,
    ]))

    mesh3d.coordinates.assign(X_ref + xi)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_val, U_m_hat, u_bar, p_bar_tilde = bg.solve_2D_background_flow()

    u_bar_3d, p_bar_3d, _ = build_3d_background_flow_differentiable(
        R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_bar, p_bar_tilde,
        X_ref=X_ref, xi=xi)

    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    u_bar_3d_acc = Function(V_3d).interpolate(u_bar_3d)

    J = assemble(inner(u_bar_3d_acc, u_bar_3d_acc) * dx)

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    h_r = Function(R_space).assign(1.0)
    h_z = Function(R_space).assign(1.0)

    Jhat_r = ReducedFunctional(J, c_r)
    Jhat_z = ReducedFunctional(J, c_z)

    print("Taylor test w.r.t. delta_r (expect 2nd-order convergence):")
    taylor_test(Jhat_r, delta_r, h_r)
    print("\nTaylor test w.r.t. delta_z (expect 2nd-order convergence):")
    taylor_test(Jhat_z, delta_z, h_z)
