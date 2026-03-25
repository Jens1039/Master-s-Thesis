import os
from copy import deepcopy

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import firedrake as fd
from firedrake.adjoint import stop_annotating, annotate_tape, continue_annotation
from pyadjoint import Block, get_working_tape, set_working_tape, Tape, ReducedFunctional, Control, taylor_test

from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole


class background_flow:

    def __init__(self, R, H, W, Re, comm=None):

        self.R   = R
        self.H   = H
        self.W   = W
        self.Re_float  = Re
        actual_comm = comm if comm is not None else fd.COMM_WORLD
        self.mesh2d = fd.RectangleMesh(120, 120, self.W, self.H, quadrilateral=False, comm=actual_comm)

    def solve_2D_background_flow(self):

        V       = fd.VectorFunctionSpace(self.mesh2d, "CG", 2, dim=3)
        Q       = fd.FunctionSpace(self.mesh2d, "CG", 1)
        G_space = fd.FunctionSpace(self.mesh2d, "R", 0)
        W_mixed = V * Q * G_space

        w = fd.Function(W_mixed)
        self.Re = fd.Function(G_space).assign(self.Re_float)
        u, p, G = fd.split(w)
        v, q, g = fd.TestFunctions(W_mixed)

        u_r = u[0];  u_theta = u[2];  u_z = u[1]
        v_r = v[0];  v_theta = v[2];  v_z = v[1]

        x = fd.SpatialCoordinate(self.mesh2d)
        r = x[0] - 0.5 * self.W

        def del_r(f): return fd.Dx(f, 0)
        def del_z(f): return fd.Dx(f, 1)

        F_cont = (q * (del_r(u_r) + del_z(u_z) + u_r / (self.R + r))
                  * (self.R + r) * fd.dx)
        F_r    = ((u_r * del_r(u_r) + u_z * del_z(u_r)
                   - (u_theta**2) / (self.R + r)) * v_r
                  + del_r(p) * v_r
                  + (fd.Constant(1.0) / self.Re) * fd.dot(fd.grad(u_r), fd.grad(v_r))
                  + (1 / self.Re) * (u_r / (self.R + r)**2) * v_r
                  ) * (self.R + r) * fd.dx
        F_theta = ((u_r * del_r(u_theta) + u_z * del_z(u_theta)
                    + (u_r * u_theta) / (self.R + r)) * v_theta
                   - ((G * self.R) / (self.R + r)) * v_theta
                   + 1 / self.Re * fd.dot(fd.grad(u_theta), fd.grad(v_theta))
                   + 1 / self.Re * (u_theta / (self.R + r)**2) * v_theta
                   ) * (self.R + r) * fd.dx
        F_z     = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
                   + del_z(p) * v_z
                   + 1 / self.Re * fd.dot(fd.grad(u_z), fd.grad(v_z))
                   ) * (self.R + r) * fd.dx
        F_G     = (u_theta - 1.0) * g * fd.dx
        F       = F_r + F_theta + F_z + F_cont + F_G

        no_slip   = fd.DirichletBC(W_mixed.sub(0), fd.Constant((0.0, 0.0, 0.0)),
                                   "on_boundary")
        nullspace = fd.MixedVectorSpaceBasis(
            W_mixed,
            [W_mixed.sub(0),
             fd.VectorSpaceBasis(constant=True, comm=W_mixed.comm),
             W_mixed.sub(2)],
        )

        problem = fd.NonlinearVariationalProblem(F, w, bcs=[no_slip])
        solver  = fd.NonlinearVariationalSolver(
            problem, nullspace=nullspace,
            solver_parameters={
                "snes_type": "newtonls", "snes_linesearch_type": "l2",
                "mat_type": "matfree", "ksp_type": "fgmres",
                "pc_type": "fieldsplit", "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                "pc_fieldsplit_0_fields": "0,1", "pc_fieldsplit_1_fields": "2",
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

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, u_vom, u_cyl, inv_perm, V_vom):
        self._dependencies = []
        self._outputs      = []
        self.block_helper  = None
        self.tag           = None
        self.adj_state     = None
        # ---------------------------------------------------------------
        self.add_dependency(u_vom)
        self.add_output(u_cyl.create_block_variable())
        self.inv_perm = inv_perm
        self.perm     = np.argsort(inv_perm)
        self.V_vom    = V_vom

    def recompute_component(self, inputs, block_variable, idx, prepared):
        output = block_variable.output
        with stop_annotating():
            output.dat.data[:] = inputs[0].dat.data_ro[self.inv_perm]
        return output

    def evaluate_adj_component(self, inputs, adj_inputs,
                                block_variable, idx, prepared=None):
        with stop_annotating():
            adj_vom = fd.Function(self.V_vom)
            adj_vom.dat.data[:] = adj_inputs[0].dat.data_ro[self.perm]
        return adj_vom

    def evaluate_tlm_component(self, inputs, tlm_inputs,
                                block_variable, idx, prepared=None):
        tlm_in = tlm_inputs[0]
        if tlm_in is None:
            return None
        with stop_annotating():
            tlm_out = fd.Function(block_variable.output.function_space())
            tlm_out.dat.data[:] = tlm_in.dat.data_ro[self.inv_perm]
        return tlm_out


def _compute_inv_perm(vom, query_pts):
    vom_coords = vom.coordinates.dat.data_ro
    tree       = cKDTree(vom_coords)
    dist, inv_perm = tree.query(query_pts, workers=-1)
    if np.max(dist) > 1e-10:
        raise RuntimeError(
            f"VOM-Query mismatch! Max. dist: {np.max(dist):.2e}")
    return inv_perm.astype(np.int32)


def vom_transfer(u_vom, V_target, inv_perm, V_vom):
    u_out = fd.Function(V_target)
    with stop_annotating():
        u_out.dat.data[:] = u_vom.dat.data_ro[inv_perm]
    if annotate_tape():
        block = VOMTransferBlock(u_vom, u_out, inv_perm, V_vom)
        get_working_tape().add_block(block)
    return u_out


class VOMInterpolateTransferBlock(Block):

    def __init__(self, u_2d, u_3d_out, V_vom, inv_perm):
        super().__init__()
        self.add_dependency(u_2d)
        self.add_output(u_3d_out.create_block_variable())
        self.V_vom    = V_vom
        self.inv_perm = inv_perm
        self.perm     = np.argsort(inv_perm)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        output = block_variable.output
        with stop_annotating():
            u_vom = fd.Function(self.V_vom)
            u_vom.interpolate(inputs[0])                    # 2D → VOM
            output.dat.data[:] = u_vom.dat.data_ro[self.inv_perm]  # VOM → 3D
        return output

    def evaluate_adj_component(self, inputs, adj_inputs,
                                block_variable, idx, prepared=None):
        with stop_annotating():
            # adjoint of permutation: scatter back to VOM ordering
            adj_vom = fd.Function(self.V_vom)
            adj_vom.dat.data[:] = adj_inputs[0].dat.data_ro[self.perm]
            # adjoint of VOM interpolation: transpose interpolation
            adj_2d = fd.Function(inputs[0].function_space())
            # Firedrake's interpolation transpose via assemble
            v = fd.TestFunction(inputs[0].function_space())
            adj_2d.assign(fd.assemble(fd.action(
                fd.interpolate(v, self.V_vom).form, adj_vom)))
        return adj_2d

    def evaluate_tlm_component(self, inputs, tlm_inputs,
                                block_variable, idx, prepared=None):
        tlm_in = tlm_inputs[0]
        if tlm_in is None:
            return None
        with stop_annotating():
            tlm_vom = fd.Function(self.V_vom)
            tlm_vom.interpolate(tlm_in)
            tlm_out = fd.Function(block_variable.output.function_space())
            tlm_out.dat.data[:] = tlm_vom.dat.data_ro[self.inv_perm]
        return tlm_out


def vom_interpolate_and_transfer(u_2d, V_vom, V_target, inv_perm):

    u_out = fd.Function(V_target)
    with stop_annotating():
        u_vom = fd.Function(V_vom)
        u_vom.interpolate(u_2d)
        u_out.dat.data[:] = u_vom.dat.data_ro[inv_perm]
    if annotate_tape():
        block = VOMInterpolateTransferBlock(u_2d, u_out, V_vom, inv_perm)
        get_working_tape().add_block(block)
    return u_out


def build_3d_background_flow_differentiable(R, H, W, G, mesh3d, tags,
                                             u_bar_2d, p_bar_2d):

    mesh2d = u_bar_2d.function_space().mesh()
    V_3d   = fd.VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d   = fd.FunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():

        cf = fd.Function(fd.VectorFunctionSpace(mesh3d, "CG", 2))
        cf.interpolate(fd.SpatialCoordinate(mesh3d))
        xyz = cf.dat.data_ro.copy()

        r_dofs = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        cos_th = xyz[:, 0] / r_dofs
        sin_th = xyz[:, 1] / r_dofs

        x2d = np.clip(r_dofs      - R + 0.5 * W, 0.0, W)
        y2d = np.clip(xyz[:, 2]   + 0.5 * H,     0.0, H)
        qu  = np.column_stack([x2d, y2d])

        V_vom_u    = fd.VectorFunctionSpace(
            fd.VertexOnlyMesh(mesh2d, qu, missing_points_behaviour="error"),
            "DG", 0, dim=3)
        vom_u      = V_vom_u.mesh()
        inv_perm_u = _compute_inv_perm(vom_u, qu)

        cos_fn = fd.Function(fd.FunctionSpace(mesh3d, "CG", 2))
        sin_fn = fd.Function(fd.FunctionSpace(mesh3d, "CG", 2))
        cos_fn.dat.data[:] = cos_th
        sin_fn.dat.data[:] = sin_th

        cp = fd.Function(fd.VectorFunctionSpace(mesh3d, "CG", 1))
        cp.interpolate(fd.SpatialCoordinate(mesh3d))
        xyz_p = cp.dat.data_ro.copy()

        r_p    = np.sqrt(xyz_p[:, 0]**2 + xyz_p[:, 1]**2)
        x2d_p  = np.clip(r_p           - R + 0.5 * W, 0.0, W)
        y2d_p  = np.clip(xyz_p[:, 2]   + 0.5 * H,     0.0, H)
        qp     = np.column_stack([x2d_p, y2d_p])

        V_vom_p    = fd.FunctionSpace(
            fd.VertexOnlyMesh(mesh2d, qp, missing_points_behaviour="error"),
            "DG", 0)
        vom_p      = V_vom_p.mesh()
        inv_perm_p = _compute_inv_perm(vom_p, qp)

        theta_p    = np.arctan2(xyz_p[:, 1], xyz_p[:, 0])
        correction = fd.Function(Q_3d)
        correction.dat.data[:] = G * R * theta_p

    u_cyl_3d = vom_interpolate_and_transfer(u_bar_2d, V_vom_u, V_3d, inv_perm_u)

    u_bar_3d = fd.Function(V_3d)
    u_bar_3d.interpolate(fd.as_vector([
        cos_fn * u_cyl_3d[0] - sin_fn * u_cyl_3d[2],
        sin_fn * u_cyl_3d[0] + cos_fn * u_cyl_3d[2],
        u_cyl_3d[1],
    ]))

    p_vom = fd.Function(V_vom_p)
    p_vom.interpolate(p_bar_2d)
    p_cyl_3d = vom_interpolate_and_transfer(p_bar_2d, V_vom_p, Q_3d, inv_perm_p)

    p_bar_3d = fd.Function(Q_3d)
    p_bar_3d.assign(p_cyl_3d - correction)

    return u_bar_3d, p_bar_3d


if __name__ == "__main__":

    R_hat = 500
    H_hat = 2
    W_hat = 2
    L_hat = 4 * max(H_hat, W_hat)
    a_hat = 0.05
    Re = 1.0

    particle_maxh = 0.2 * a_hat
    global_maxh = 0.2 * min(H_hat, W_hat)

    # ------------------------------------------------------------------------------------------------------------------

    set_working_tape(Tape())
    while not annotate_tape():
        continue_annotation()

    with stop_annotating():
        mesh3d, tags = make_curved_channel_section_with_spherical_hole(
            R_hat, H_hat, W_hat, L_hat, a_hat,
            particle_maxh, global_maxh, r_off=0.0, z_off=0.0)

    set_working_tape(Tape())
    while not annotate_tape():
        continue_annotation()

    # ===== MINIMAL TEST: Laplace-Solve weglassen =====
    R_space = fd.FunctionSpace(mesh3d, "R", 0)
    delta_r = fd.Function(R_space, name="delta_r").assign(0.0)

    V_def = fd.VectorFunctionSpace(mesh3d, "CG", 1)

    import math

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    with stop_annotating():
        X_ref = fd.Function(V_def, name="X_ref")
        X_ref.interpolate(fd.SpatialCoordinate(mesh3d))

    xi = fd.Function(V_def, name="xi")
    xi.interpolate(fd.as_vector([
        delta_r * cos_th,
        delta_r * sin_th,
        fd.Constant(0.0),
    ]))

    mesh3d.coordinates.assign(X_ref + xi)

    x3d = fd.SpatialCoordinate(mesh3d)
    J_test = fd.assemble(x3d[0] ** 2 * fd.dx)
    print(f"J_test (volume) = {J_test}")

    c_r = Control(delta_r)
    J_hat_test = ReducedFunctional(J_test, c_r)

    tape = get_working_tape()
    print(f"\n=== TAPE: {len(tape.get_blocks())} blocks ===")
    for i, block in enumerate(tape.get_blocks()):
        deps = [
            f"{type(d.output).__name__}(name={getattr(d.output, 'name', lambda: '?')() if callable(getattr(d.output, 'name', None)) else getattr(d.output, 'name', '?')})"
            for d in block.get_dependencies()]
        print(f"  Block {i}: {type(block).__name__} deps={deps}")

    # Taylor-Test
    h_r = fd.Function(R_space).assign(0.1)
    taylor_test(J_hat_test, [fd.Function(R_space).assign(0.0)], [h_r])
    exit()

    R_space = fd.FunctionSpace(mesh3d, "R", 0)
    delta_r = fd.Function(R_space, name="delta_r").assign(0.0)
    delta_z = fd.Function(R_space, name="delta_z").assign(0.0)

    V_def = fd.VectorFunctionSpace(mesh3d, "CG", 1)
    xi = fd.Function(V_def, name="mesh_displacement")
    eta = fd.TestFunction(V_def)

    a_form = fd.inner(fd.grad(xi), fd.grad(eta)) * fd.dx
    L_form = fd.inner(fd.Constant((0, 0, 0)), eta) * fd.dx

    import math

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    bc_particle = fd.DirichletBC(V_def, fd.as_vector([
        delta_r * cos_th,
        delta_r * sin_th,
        delta_z,
    ]), tags["particle"])

    bc_walls = fd.DirichletBC(V_def, fd.Constant((0, 0, 0)), tags["walls"][0])
    bc_inlet = fd.DirichletBC(V_def, fd.Constant((0, 0, 0)), tags["inlet"])
    bc_outlet = fd.DirichletBC(V_def, fd.Constant((0, 0, 0)), tags["outlet"])

    F_mesh = fd.inner(fd.grad(xi), fd.grad(eta)) * fd.dx

    problem = fd.NonlinearVariationalProblem(
        F_mesh, xi,
        bcs=[bc_particle, bc_walls, bc_inlet, bc_outlet])

    solver = fd.NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_type": "ksponly",
            "ksp_type": "cg",
            "pc_type": "gamg",
        })

    X_ref = fd.Function(V_def, name="X_ref")
    X_ref.interpolate(fd.SpatialCoordinate(mesh3d))

    solver.solve()
    print(f"||xi|| = {fd.norm(xi)}")

    mesh3d.coordinates.assign(X_ref + xi)

    bg = background_flow(R_hat, H_hat, W_hat, Re)
    G_val, U_m_hat, u_bar, p_bar_tilde = bg.solve_2D_background_flow()

    u_bar_3d, p_bar_3d = build_3d_background_flow_differentiable(R_hat, H_hat, W_hat, G_val, mesh3d, tags,
                                             u_bar, p_bar_tilde)

    J = fd.assemble(fd.inner(u_bar_3d, u_bar_3d) * fd.dx)

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    J_hat = ReducedFunctional(J, [c_r, c_z])

    dJ = J_hat.derivative()

    h_r = fd.Function(R_space).assign(0.01)
    h_z = fd.Function(R_space).assign(0.01)

    # Statt des ganzen 2D-Solve + VOM-Transfer, teste NUR die Mesh-Deformation:
    J_test = fd.assemble(fd.Constant(1.0) * fd.dx(domain=mesh3d))
    print(f"J_test (volume) = {J_test}")

    c_r = Control(delta_r)
    J_hat_test = ReducedFunctional(J_test, c_r)
    dJ_test = J_hat_test.derivative()
    print(f"dJ_test/d(delta_r) = {float(dJ_test)}")

    tape = get_working_tape()
    for i, block in enumerate(tape.get_blocks()):
        print(f"Block {i}: {type(block).__name__}, "
              f"deps={[str(d) for d in block.get_dependencies()]}")

    taylor_test(J_hat,
                [fd.Function(R_space).assign(0.0),
                 fd.Function(R_space).assign(0.0)],
                [h_r, h_z])





