import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt


class background_flow:

    def __init__(self, R, H, W, Re, comm=None):
        self.R = R
        self.H = H
        self.W = W
        self.Re = Re

        # Use COMM_SELF to force a local solve and prevent MPI deadlocks, as other ranks are not participating.
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
        U_at_list = self.u_bar.at(plotting_points)

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


def build_3d_background_flow(R, H, W, G, mesh3d, u_bar_2d, p_bar_2d):

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

    x_query_u = (r_3d - R) + 0.5 * W
    y_query_u = z_3d + 0.5 * H

    epsilon = 1e-12
    x_query_u = np.clip(x_query_u, 0.0, W)
    y_query_u = np.clip(y_query_u, 0.0, H)

    query_points_u = np.column_stack((x_query_u, y_query_u))

    u_vals = np.array(u_bar_2d.at(query_points_u, tolerance=1e-8))

    u_r = u_vals[:, 0]
    u_z = u_vals[:, 1]
    u_th = u_vals[:, 2]

    cos_th = np.cos(theta_3d)
    sin_th = np.sin(theta_3d)

    u_x_3d = u_r * cos_th - u_th * sin_th
    u_y_3d = u_r * sin_th + u_th * cos_th
    u_z_3d = u_z

    u_3d.dat.data[:] = np.column_stack((u_x_3d, u_y_3d, u_z_3d))

    Q_coords = VectorFunctionSpace(mesh3d, "CG", 1)
    coords_func_p = Function(Q_coords).interpolate(SpatialCoordinate(mesh3d))
    xyz_nodes_p = coords_func_p.dat.data_ro

    r_3d_p = np.sqrt(xyz_nodes_p[:, 0] ** 2 + xyz_nodes_p[:, 1] ** 2)
    theta_3d_p = np.arctan2(xyz_nodes_p[:, 1], xyz_nodes_p[:, 0])
    z_3d_p = xyz_nodes_p[:, 2]

    x_query_p = np.clip((r_3d_p - R) + 0.5 * W, 0.0, W)
    y_query_p = np.clip(z_3d_p + 0.5 * H, 0.0, H)

    query_points_p = np.column_stack((x_query_p, y_query_p))

    p_vals = np.array(p_bar_2d.at(query_points_p, tolerance=1e-8))

    if p_vals.ndim > 1:
        p_vals = p_vals.flatten()

    p_total = p_vals - G * R * theta_3d_p
    p_3d.dat.data[:] = p_total

    return u_3d, p_3d

