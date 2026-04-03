import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import sys
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import warnings

# Wildcard Import
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


class WallBCBlock(Block):

    def __init__(self, u_in, u_out, bc_node_indices, V):
        super().__init__()
        self.add_dependency(u_in)
        self.add_output(u_out.create_block_variable())
        self.bc_nodes = bc_node_indices      # 1-D int array of DOF rows to zero
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
        bc_node_indices = bc.nodes            # 1-D numpy int array

    u_out = Function(V, name=u_fn.name() + "_bc" if u_fn.name() else "u_bc")
    with stop_annotating():
        u_out.dat.data[:] = u_fn.dat.data_ro[:]
        u_out.dat.data[bc_node_indices] = 0.0

    if annotate_tape():
        block = WallBCBlock(u_fn, u_out, bc_node_indices, V)
        get_working_tape().add_block(block)

    return u_out


def build_3d_background_flow_differentiable(R, H, W, G_val, mesh3d, tags, u_bar_2d, p_bar_2d):

    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d = FunctionSpace(mesh3d, "CG", 1)

    x3d = SpatialCoordinate(mesh3d)
    r_xy = sqrt(x3d[0] ** 2 + x3d[1] ** 2)
    cos_ufl = x3d[0] / r_xy
    sin_ufl = x3d[1] / r_xy

    theta_ufl = atan2(x3d[1], x3d[0])

    with stop_annotating():

        V_coords = VectorFunctionSpace(mesh3d, "CG", 2)
        coords_func_u = Function(V_coords).interpolate(SpatialCoordinate(mesh3d))
        xyz_nodes_u = coords_func_u.dat.data_ro.copy()

        r_3d = np.sqrt(xyz_nodes_u[:, 0] ** 2 + xyz_nodes_u[:, 1] ** 2)

        qu = np.column_stack([
            np.clip(r_3d - R + 0.5 * W, 0, W),
            np.clip(xyz_nodes_u[:, 2] + 0.5 * H, 0, H),
        ])

        mesh2d = u_bar_2d.function_space().mesh()
        vom_u      = VertexOnlyMesh(mesh2d, qu, missing_points_behaviour="error")
        V_vom_u    = VectorFunctionSpace(vom_u, "DG", 0, dim=3)
        inv_perm_u = _compute_inv_perm(vom_u, qu)

        # cos_fn = Function(FunctionSpace(mesh3d, "CG", 2))
        # sin_fn = Function(FunctionSpace(mesh3d, "CG", 2))
        # cos_fn.dat.data[:] = xyz_nodes_u[:, 0] / r_3d
        # sin_fn.dat.data[:] = xyz_nodes_u[:, 1] / r_3d

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

        # theta_fn = Function(Q_3d)
        # theta_fn.dat.data[:] = np.arctan2(xyz_p[:, 1], xyz_p[:, 0])

    u_vom    = Function(V_vom_u).interpolate(u_bar_2d)
    u_cyl_3d = vom_transfer(u_vom, V_3d, inv_perm_u, V_vom_u)

    p_vom    = Function(V_vom_p).interpolate(p_bar_2d)
    p_cyl_3d = vom_transfer(p_vom, Q_3d, inv_perm_p, V_vom_p)

    u_cyl_3d = apply_wall_bc_on_tape(u_cyl_3d, V_3d, tags["walls"])

    u_bar_3d = as_vector([
        cos_ufl * u_cyl_3d[0] - sin_ufl * u_cyl_3d[2],
        sin_ufl * u_cyl_3d[0] + cos_ufl * u_cyl_3d[2],
        u_cyl_3d[1],
    ])

    p_bar_3d = p_cyl_3d - Constant(G_val * R) * theta_ufl

    '''
    u_bar_3d = as_vector([
        cos_fn * u_cyl_3d[0] - sin_fn * u_cyl_3d[2],
        sin_fn * u_cyl_3d[0] + cos_fn * u_cyl_3d[2],
        u_cyl_3d[1],
    ])

    p_bar_3d = p_cyl_3d - Constant(G_val * R) * theta_fn
    '''

    return u_bar_3d, p_bar_3d, u_cyl_3d


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

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R_hat, H_hat, W_hat, L_hat, a_hat,
            particle_maxh, global_maxh, r_off=0.0, z_off=0.0)

    R_space = FunctionSpace(mesh3d, "R", 0)
    delta_r = Function(R_space, name="delta_r").assign(0.0)
    delta_z = Function(R_space, name="delta_z").assign(0.0)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    x3d = SpatialCoordinate(mesh3d)
    dist = sqrt((x3d[0] - cx) ** 2 + (x3d[1] - cy) ** 2 + (x3d[2] - cz) ** 2)

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

    u_bar_3d, p_bar_3d, _ = build_3d_background_flow_differentiable(R_hat, H_hat, W_hat, G_val, mesh3d, tags, u_bar, p_bar_tilde)
    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    u_bar_3d_acc = Function(V_3d).interpolate(u_bar_3d)

    from background_flow import background_flow, build_3d_background_flow

    bg_2 = background_flow(R_hat, H_hat, W_hat, Re)
    G_val_2, U_m_hat_2, u_bar_2, p_bar_tilde_2 = bg_2.solve_2D_background_flow()

    with stop_annotating():
        u_bar_3d_2, p_bar_3d_2 = build_3d_background_flow(R_hat, H_hat, W_hat, G_val_2, mesh3d, tags, u_bar_2, p_bar_tilde_2)

    u_bar_3d_diff_data = u_bar_3d_acc.dat.data_ro - u_bar_3d_2.dat.data_ro

    print("max difference between tape-safe and verified background_flow:", np.max(np.abs(u_bar_3d_diff_data)))

    J = assemble(inner(u_bar_3d_acc, u_bar_3d_acc) * dx)

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    h_r = Function(R_space).assign(1.0)
    h_z = Function(R_space).assign(1.0)

    Jhat_r = ReducedFunctional(J, c_r)
    Jhat_z = ReducedFunctional(J, c_z)

    taylor_test(Jhat_r, delta_r, h_r)
    taylor_test(Jhat_z, delta_z, h_z)


