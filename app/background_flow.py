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

        from firedrake import COMM_WORLD
        actual_comm = comm if comm is not None else COMM_WORLD

        self.mesh2d = RectangleMesh(120, 120, self.W, self.H, quadrilateral=False, comm=actual_comm)


    def solve_2D_background_flow(self):

        # Velocity space (3 components: u_r, u_z, u_theta), pressure space, and scalar space for G
        V = VectorFunctionSpace(self.mesh2d, "CG", 2, dim=3)
        Q = FunctionSpace(self.mesh2d, "CG", 1)
        G_space = FunctionSpace(self.mesh2d, "R", 0)

        # Just the Cartesian product of V, Q and R
        W_mixed = V * Q * G_space

        w = Function(W_mixed)
        u, p, G = split(w)
        v, q, g = TestFunctions(W_mixed)

        # Transformation of the velocity components to our local coordinate system
        # u = (u_x, u_z, u_theta) in the mesh coordinates, we interpret them as:
        u_r = u[0]
        u_theta = u[2]
        u_z = u[1]

        v_r = v[0]
        v_theta = v[2]
        v_z = v[1]

        # extracts a UFL expression for the coordinates of the mesh
        x = SpatialCoordinate(self.mesh2d)

        # Here we need to take into account, that (r, z) = (0, 0) in the center of the duct.
        # Apart from that, we don't need to take care of coordinate transformations, since the dircetion of x and z is the same as of r and z
        # even when they are shifted
        r = x[0] - 0.5 * self.W

        # Shortcuts for readability
        def del_r(f):  return Dx(f, 0)
        def del_z(f):  return Dx(f, 1)

        # weak form of the continuity equation
        F_cont = q * (del_r(u_r) + del_z(u_z) + u_r / (self.R + r)) * (self.R + r) * dx

        F_r = (
                      (u_r * del_r(u_r) + u_z * del_z(u_r) - (u_theta ** 2) / (self.R + r)) * v_r
                      + del_r(p) * v_r
                      + (1 / self.Re) * dot(grad(u_r), grad(v_r))
                      + (1 / self.Re) * (u_r / (self.R + r) ** 2) * v_r
              ) * (self.R + r) * dx

        F_theta = (
                          (u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_r * u_theta) / (self.R + r)) * v_theta
                          - ((G * self.R) / (self.R + r)) * v_theta
                          + 1 / self.Re * dot(grad(u_theta), grad(v_theta))
                          + 1 / self.Re * (u_theta / (self.R + r) ** 2) * v_theta
                  ) * (self.R + r) * dx

        F_z = (
                      (u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
                      + del_z(p) * v_z
                      + 1 / self.Re * dot(grad(u_z), grad(v_z))
              ) * (self.R + r) * dx

        # weak form of the flow rate constraint
        F_G = (u_theta  - 1.0) * g * dx

        # Total residual
        F = F_r + F_theta + F_z + F_cont + F_G

        # When you define the problem the boundary conditions must apply to components of W_mixed, not to the stand-alone V,
        # thats why we're using W_mixed.sub(0) = "First component of W_mixed"
        no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")

        # In Navier-Stokes, the pressure just appears in a differentiated form and is therefore unique except for a constant.
        # A constant pressure is (because of a lack of boundary conditions for p) part of the kernel of the linearised operator.
        # If we give the solver the information where nullspaces are, it can ignore this direction while inverting the matrix
        # in our Newton method.
        nullspace = MixedVectorSpaceBasis(
            W_mixed,
            [
                W_mixed.sub(0),
                VectorSpaceBasis(constant=True, comm=W_mixed.comm),
                W_mixed.sub(2),
            ],
        )

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        # Firedrake calculates the Gateaux-Derivative and transfers it into the Jacobian matrix on the given FEM mesh.
        # The resulting LSE is solved by using LU - decomposition.
        solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters={
                # SNES = Scalable Nonlinear Equation Solver  here: Newton with Line Search as step range criteria
                "snes_type": "newtonls",
                # bt: test the descent criteria ||F(w_k + a_k*d_k)|| < (1 - ca_k)||F(w_k)|| with decreasing a_k until it is valid
                # l2: solves min ||F(x_k + alpha*d_k||_(L^2) over alpha
                "snes_linesearch_type": "l2",
                # Tests for ||F(w_k)||/||F(w_0)|| and stops algorithm once ratio has fallen below
                "snes_rtol": 1e-8,
                # Tests for ||F(w_k)|| and stops algorithm once absolute value has fallen below
                "snes_atol": 1e-10,
                # Maximal Newton steps
                "snes_max_it": 50,
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
        u_bar, p_bar_tilde, G_hat = w.subfunctions
        # Extracts the value from the scalar firedrake function
        G_hat_val = float(G_hat.dat.data_ro[0])
        self.u_bar = u_bar
        # We want to parse the pressure with
        self.p_bar_tilde = p_bar_tilde

        # necessary for rescale Re in the same way as in the paper
        u_data = self.u_bar.dat.data_ro
        U_m_hat = np.max(u_data[:, 2])
        self.U_m_hat = U_m_hat
        print("U_m_hat", self.U_m_hat)

        return u_bar, p_bar_tilde, G_hat_val, U_m_hat


    def plot_2D_background_flow(self):

        coords = self.mesh2d.coordinates.dat.data_ro
        xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
        zmin, zmax = float(coords[:, 1].min()), float(coords[:, 1].max())


        nxp, nzp = 160, 160
        xi = np.linspace(xmin, xmax, nxp)
        zi = np.linspace(zmin, zmax, nzp)
        Xi, Zi = np.meshgrid(xi, zi)


        pts = np.column_stack([Xi.ravel(), Zi.ravel()])
        U_at_list = self.u_bar.at(pts)

        try:
            U_at = np.asarray(U_at_list, dtype=float)
            if U_at.ndim != 2 or U_at.shape[1] != 3:
                raise ValueError
        except Exception:
            U_at = np.vstack([np.asarray(v, dtype=float).ravel() for v in U_at_list])

        Ur = U_at[:, 0].reshape(nzp, nxp)
        Uz = U_at[:, 1].reshape(nzp, nxp)
        Uth = U_at[:, 2].reshape(nzp, nxp)

        Speed = np.sqrt(Ur ** 2 + Uz ** 2)

        Speed[~np.isfinite(Speed)] = 0.0
        Ur[~np.isfinite(Ur)] = 0.0
        Uz[~np.isfinite(Uz)] = 0.0

        x_mid = (xmin + xmax) / 2.0
        z_mid = (zmin + zmax) / 2.0

        xi_plot = xi - x_mid
        zi_plot = zi - z_mid
        Xi_plot, Zi_plot = np.meshgrid(xi_plot, zi_plot)

        eps = 1e-14
        lw = 0.8 + 2.0 * (Speed / (Speed.max() + eps))

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$z$")

        cf = ax.contourf(Xi_plot, Zi_plot, Uth, levels=40, cmap="coolwarm")
        cbar1 = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
        cbar1.set_label(r"$u_\theta$")

        strm = ax.streamplot(
            xi_plot, zi_plot, Ur, Uz,
            density=1.4,
            color=Speed,
            linewidth=lw,
            cmap="viridis",
            arrowsize=1.2,
            minlength=0.1
        )
        cbar2 = fig.colorbar(strm.lines, ax=ax, shrink=0.9, pad=0.02)
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


    try:
        u_vals = np.array(u_bar_2d.at(query_points_u, tolerance=1e-8))
    except Exception as e:
        print(f"Interpolation Warning (Velocity): {e}")
        u_vals = np.zeros_like(xyz_nodes_u)

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

    try:
        p_vals = np.array(p_bar_2d.at(query_points_p, tolerance=1e-8))
    except Exception as e:
        print(f"Interpolation Warning (Pressure): {e}")
        p_vals = np.zeros(len(query_points_p))

    if p_vals.ndim > 1:
        p_vals = p_vals.flatten()

    p_total = p_vals - G * R * theta_3d_p
    p_3d.dat.data[:] = p_total

    return u_3d, p_3d

