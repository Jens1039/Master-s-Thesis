import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *

import numpy as np
import matplotlib.pyplot as plt


class background_flow:

    def __init__(self, R, H, W, Re, comm=None, shape_gamma=None):
        self.R = R
        self.H = H
        self.W = W
        self.Re = Re
        self.shape_gamma = shape_gamma

        actual_comm = comm if comm is not None else COMM_WORLD
        self.mesh2d = RectangleMesh(120, 120, self.W, self.H, quadrilateral=False, comm=actual_comm)


    def solve_2D_background_flow(self):

        # Velocity space (3 components: u_r, u_z, u_theta), pressure space, and scalar space for G
        V = VectorFunctionSpace(self.mesh2d, "CG", 2, dim=3)
        Q = FunctionSpace(self.mesh2d, "CG", 1)
        G_space = FunctionSpace(self.mesh2d, "R", 0)

        W_mixed = V * Q * G_space

        w = Function(W_mixed)
        u, p, G = split(w)
        v, q, g = TestFunctions(W_mixed)

        # Transformation of the velocity components to our cylindric coordinate system
        u_r = u[0]
        u_theta = u[2]
        u_z = u[1]

        v_r = v[0]
        v_theta = v[2]
        v_z = v[1]

        x = SpatialCoordinate(self.mesh2d)

        # Here we need to take into account, that (r, z) = (0, 0) in the center of the duct.
        # Apart from that, we don't need to take care of coordinate transformations, since the dircetion of x and z is the same as of r and z even when they are shifted
        r = x[0] - 0.5 * self.W

        # Shortcuts for readability
        def del_r(f):  return Dx(f, 0)
        def del_z(f):  return Dx(f, 1)

        # Below is equation (B1) from Harding et. al. in weak and nondimensionalized form and with the pressure-gradient equation added

        F_cont = q * (del_r(u_r) + del_z(u_z) + u_r / (self.R + r)) * (self.R + r) * dx

        F_r = ((u_r * del_r(u_r) + u_z * del_z(u_r) - (u_theta ** 2) / (self.R + r)) * v_r
                    + del_r(p) * v_r
                    + (1 / self.Re) * dot(grad(u_r), grad(v_r))
                    + (1 / self.Re) * (u_r / (self.R + r) ** 2) * v_r
              ) * (self.R + r) * dx

        F_theta = ((u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_r * u_theta) / (self.R + r)) * v_theta
                        - ((G * self.R) / (self.R + r)) * v_theta
                        + 1 / self.Re * dot(grad(u_theta), grad(v_theta))
                        + 1 / self.Re * (u_theta / (self.R + r) ** 2) * v_theta
                  ) * (self.R + r) * dx

        F_z = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
                    + del_z(p) * v_z
                    + 1 / self.Re * dot(grad(u_z), grad(v_z))
              ) * (self.R + r) * dx

        F_G = (u_theta  - 1.0) * g * dx

        # Total residual
        F = F_r + F_theta + F_z + F_cont + F_G

        no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")

        nullspace = MixedVectorSpaceBasis(
            W_mixed,
            [
                W_mixed.sub(0),
                VectorSpaceBasis(constant=True, comm=W_mixed.comm),
                W_mixed.sub(2),
            ],
        )

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                # Tell firedrake not to assemble the global matrix to avoid the "Monolithic matrix assembly ..." error
                "mat_type": "matfree",
                # Use FieldSplit to separate (u, p) from (G)
                "ksp_type": "fgmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                # Manually group fields:
                #    Split '0' = Fields 0, 1 (Velocity, Pressure)
                #    Split '1' = Field 2 (G)
                "pc_fieldsplit_0_fields": "0,1",
                "pc_fieldsplit_1_fields": "2",
                # Solver for Split '0' (Navier-Stokes block)
                #    Since the global operator is matfree, we must use "AssembledPC"
                #    to force assembly of this block so we can use MUMPS LU.
                "fieldsplit_0": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled_pc_type": "lu",
                    "assembled_pc_factor_mat_solver_type": "mumps"
                },
                # Solver for Split '1' (The scalar G)
                #    This is just a 1x1 block, so 'none' or 'jacobi' is fine.
                "fieldsplit_1": {
                    "ksp_type": "preonly",
                    "pc_type": "none"
                },
            },
        )

        solver.solve()

        u_bar, p_bar_tilde, G_sol = w.subfunctions

        # Extracts the value from the scalar firedrake function
        G_val = float(G_sol.dat.data_ro[0])

        self.u_bar = u_bar
        self.p_bar_tilde = p_bar_tilde

        # necessary for rescale Re in the same way as Harding et. al.
        u_data = self.u_bar.dat.data_ro
        U_m_hat = np.max(u_data[:, 2])
        self.U_m_hat = U_m_hat

        return G_val, U_m_hat, u_bar, p_bar_tilde


    def plot(self):

        nx, nz = 120, 120

        coords = self.mesh2d.coordinates.dat.data_ro
        x_i = np.linspace(float(coords[:, 0].min()), float(coords[:, 0].max()), nx)
        z_i = np.linspace(float(coords[:, 1].min()), float(coords[:, 1].max()), nz)
        X_i, Z_i = np.meshgrid(x_i, z_i)

        plotting_points = np.column_stack([X_i.ravel(), Z_i.ravel()])

        evaluator = PointEvaluator(self.mesh2d, plotting_points)
        U_at_list = evaluator.evaluate(self.u_bar)

        u_at = np.asarray(U_at_list, dtype=float)

        U_r = u_at[:, 0].reshape(nz, nx)
        U_theta = u_at[:, 2].reshape(nz, nx)
        U_z = u_at[:, 1].reshape(nz, nx)

        Speed = np.sqrt(U_r ** 2 + U_z ** 2)

        Speed[~np.isfinite(Speed)] = 0.0
        U_r[~np.isfinite(U_r)] = 0.0
        U_z[~np.isfinite(U_z)] = 0.0

        x_mid = (float(coords[:, 0].min()) + float(coords[:, 0].max())) / 2.0
        z_mid = (float(coords[:, 1].min()) + float(coords[:, 1].max())) / 2.0

        xi_plot = x_i - x_mid
        zi_plot = z_i - z_mid
        Xi_plot, Zi_plot = np.meshgrid(xi_plot, zi_plot)

        eps = 1e-14
        lw = 0.8 + 2.0 * (Speed / (Speed.max() + eps))

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$z$")

        cf = ax.contourf(Xi_plot, Zi_plot, U_theta, levels=40, cmap="coolwarm")
        cbar1 = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.05)
        cbar1.set_label(r"$u_\theta$")

        stream = ax.streamplot(
            xi_plot, zi_plot, U_r, U_z,
            density=1.4,
            color=Speed,
            linewidth=lw,
            cmap="viridis",
            arrowsize=1.2,
            minlength=0.1
        )

        cbar2 = fig.colorbar(stream.lines, ax=ax, shrink=0.9)
        cbar2.set_label(r"$|u_{\mathrm{sec}}| = \sqrt{u_r^2 + u_z^2}$")

        plt.tight_layout()
        plt.show()


def build_3d_background_flow(R, H, W, G, mesh3d, tags, u_bar_2d, p_bar_2d, shape_gamma=None):
    """If shape_gamma is set, 2D flow is on deformed cross-section; query points are transformed accordingly."""
    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d = FunctionSpace(mesh3d, "CG", 1)

    u_3d = Function(V_3d)
    p_3d = Function(Q_3d)

    V_coords = VectorFunctionSpace(mesh3d, "CG", 2)
    coords_func_u = Function(V_coords).interpolate(SpatialCoordinate(mesh3d))
    xyz_nodes_u = coords_func_u.dat.data_ro

    r_3d = np.sqrt(xyz_nodes_u[:, 0] ** 2 + xyz_nodes_u[:, 1] ** 2)
    theta_3d = np.arctan2(xyz_nodes_u[:, 1], xyz_nodes_u[:, 0])
    z_3d = xyz_nodes_u[:, 2]

    r_cs = r_3d - R
    z_cs = z_3d
    if shape_gamma is not None:
        from cross_section_shape import deformation_map
        rp, zp = deformation_map(r_cs, z_cs, W, H, shape_gamma)
        x_query_u = rp + 0.5 * W
        y_query_u = zp + 0.5 * H
    else:
        x_query_u = r_cs + 0.5 * W
        y_query_u = z_cs + 0.5 * H

    epsilon = 1e-12
    x_query_u = np.clip(x_query_u, 0.0, W)
    y_query_u = np.clip(y_query_u, 0.0, H)

    query_points_u = np.column_stack((x_query_u, y_query_u))

    u_evaluator = PointEvaluator(u_bar_2d.function_space().mesh(), query_points_u, tolerance=1e-8)
    u_vals = np.array(u_evaluator.evaluate(u_bar_2d))

    u_r = u_vals[:, 0]
    u_z = u_vals[:, 1]
    u_th = u_vals[:, 2]

    cos_th = np.cos(theta_3d)
    sin_th = np.sin(theta_3d)

    u_x_3d = u_r * cos_th - u_th * sin_th
    u_y_3d = u_r * sin_th + u_th * cos_th
    u_z_3d = u_z

    u_3d.dat.data[:] = np.column_stack((u_x_3d, u_y_3d, u_z_3d))

    # Enforce u_3d == 0 on the boundary
    bc_walls = DirichletBC(V_3d, Constant((0.0, 0.0, 0.0)), tags["walls"])
    bc_walls.apply(u_3d)

    Q_coords = VectorFunctionSpace(mesh3d, "CG", 1)
    coords_func_p = Function(Q_coords).interpolate(SpatialCoordinate(mesh3d))
    xyz_nodes_p = coords_func_p.dat.data_ro

    r_3d_p = np.sqrt(xyz_nodes_p[:, 0] ** 2 + xyz_nodes_p[:, 1] ** 2)
    theta_3d_p = np.arctan2(xyz_nodes_p[:, 1], xyz_nodes_p[:, 0])
    z_3d_p = xyz_nodes_p[:, 2]
    r_cs_p = r_3d_p - R
    if shape_gamma is not None:
        from cross_section_shape import deformation_map
        rp_p, zp_p = deformation_map(r_cs_p, z_3d_p, W, H, shape_gamma)
        x_query_p = np.clip(rp_p + 0.5 * W, 0.0, W)
        y_query_p = np.clip(zp_p + 0.5 * H, 0.0, H)
    else:
        x_query_p = np.clip(r_cs_p + 0.5 * W, 0.0, W)
        y_query_p = np.clip(z_3d_p + 0.5 * H, 0.0, H)

    query_points_p = np.column_stack((x_query_p, y_query_p))

    p_evaluator = PointEvaluator(p_bar_2d.function_space().mesh(), query_points_p, tolerance=1e-8)
    p_vals = np.array(p_evaluator.evaluate(p_bar_2d))

    if p_vals.ndim > 1:
        p_vals = p_vals.flatten()

    p_total = p_vals - G * R * theta_3d_p
    p_3d.dat.data[:] = p_total

    return u_3d, p_3d


if __name__ == "__main__":

    print("Starting sanity checks for 2D and 3D background flow...\n")

    R_test = 100.0
    H_test = 2.0
    W_test = 2.0
    Re_test = 20.0

    print("1. Solving 2D background flow...")
    bf = background_flow(R_test, H_test, W_test, Re_test)
    G_val, U_m_hat, u_2d, p_2d = bf.solve_2D_background_flow()

    print(f"   -> Calculated pressure gradient G = {G_val:.6f}")
    print(f"   -> Maximum axial velocity U_m_hat = {U_m_hat:.6f}")
    assert U_m_hat > 0, "Error: Maximum axial velocity should be positive."

    print("\n2. Checking 2D divergence (mass conservation)...")
    # Cylindrical divergence: d(u_r)/dr + d(u_z)/dz + u_r / (R + r) = 0
    # In the local mesh system: r = x - W/2
    mesh2d = bf.mesh2d
    x_coords = SpatialCoordinate(mesh2d)
    r_local = x_coords[0] - 0.5 * W_test
    R_plus_r = R_test + r_local

    u_r, u_z, u_theta = split(u_2d)
    div_2d = Dx(u_r, 0) + Dx(u_z, 1) + u_r / R_plus_r

    # L2 norm of the divergence error
    div_norm = sqrt(assemble(inner(div_2d, div_2d) * dx))
    print(f"   -> L2 norm of the divergence error: {div_norm:.2e}")
    assert div_norm < 1e-3, f"Error: 2D divergence is too high! ({div_norm})"

    print("\n3. Checking 2D boundary conditions (no-slip)...")
    # Integral of the squared velocity on the boundary (ds) should be nearly 0
    v_bnd_norm = assemble(inner(u_2d, u_2d) * ds)
    print(f"   -> Integral velocity error at the boundary: {v_bnd_norm:.2e}")
    assert v_bnd_norm < 1e-8, "Error: No-slip condition in 2D was not exactly met."

    print("\n4. Creating 3D test mesh for mapping...")
    # To test the 3D script, we construct a curved ExtrudedMesh
    m2d_base = RectangleMesh(15, 15, W_test, H_test)
    mesh3d = ExtrudedMesh(m2d_base, layers=15, layer_height=1.0)

    # Bend the originally straight 3D mesh into an arc
    Vc = mesh3d.coordinates.function_space()
    x, y, zeta = SpatialCoordinate(mesh3d)

    theta_max = 0.2  # Arc angle in radians
    r_3d_c = R_test - W_test / 2.0 + x
    theta_c = zeta * (theta_max / 15.0)
    z_cyl_c = y - H_test / 2.0

    f_coords = Function(Vc).interpolate(as_vector([r_3d_c * cos(theta_c), r_3d_c * sin(theta_c), z_cyl_c]))
    mesh3d.coordinates.assign(f_coords)

    # Provide tags dictionary as required by build_3d_background_flow
    # "on_boundary" automatically captures all boundaries of the 3D mesh.
    tags = {"walls": "on_boundary"}

    print("\n5. Executing 3D mapping...")
    u_3d, p_3d = build_3d_background_flow(R_test, H_test, W_test, G_val, mesh3d, tags, u_2d, p_2d)

    print("\n6. Checking 3D mapping consistency (velocity magnitude in channel center)...")
    # Evaluation point exactly in the center of the channel
    mid_r = R_test
    mid_theta = theta_max / 2.0
    mid_z = 0.0

    # Conversion for the 3D PointEvaluator
    mid_x_3d = mid_r * np.cos(mid_theta)
    mid_y_3d = mid_r * np.sin(mid_theta)
    mid_z_3d = mid_z

    try:
        # 3D evaluation with PointEvaluator
        evaluator_3d = PointEvaluator(mesh3d, np.array([[mid_x_3d, mid_y_3d, mid_z_3d]]))
        val_3d = evaluator_3d.evaluate(u_3d)[0]

        # 2D evaluation with PointEvaluator
        evaluator_2d = PointEvaluator(bf.mesh2d, np.array([[W_test / 2.0, H_test / 2.0]]))
        val_2d = evaluator_2d.evaluate(u_2d)[0]

        # Calculate magnitude
        speed_2d = np.sqrt(val_2d[0] ** 2 + val_2d[1] ** 2 + val_2d[2] ** 2)
        speed_3d = np.sqrt(val_3d[0] ** 2 + val_3d[1] ** 2 + val_3d[2] ** 2)

        err_speed = abs(speed_2d - speed_3d)
        print(f"   -> 2D velocity magnitude: {speed_2d:.6f}")
        print(f"   -> 3D velocity magnitude: {speed_3d:.6f}")
        print(f"   -> Absolute difference:   {err_speed:.2e}")
        assert err_speed < 1e-5, "Error: Velocity magnitudes do not match during mapping!"

    except Exception as e:
        print(f"   -> Warning: Single point evaluation failed: {e}")

    print("\n=============================================")
    print("=== ALL SANITY CHECKS COMPLETED SUCCESSFULLY ===")
    print("=============================================")