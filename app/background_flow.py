import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole
from problem_setup import *


class background_flow:

    def __init__(self, R, H, W, Re, comm=None, mesh2d=None):
        self.R = R
        self.H = H
        self.W = W
        self.Re = Re

        # If an external 2D mesh is supplied (e.g. a cross-section deformed by
        # a shape-optimisation T_2d), solve the background flow on it instead of
        # the default rectangle. The mesh must still span [0, W] x [0, H] in its
        # reference frame so the cylindrical coordinate map (r = x - W/2) holds.
        if mesh2d is not None:
            self.mesh2d = mesh2d
        else:
            actual_comm = comm if comm is not None else COMM_WORLD
            self.mesh2d = RectangleMesh(128, 128, self.W, self.H, quadrilateral=False, diagonal="crossed", comm=actual_comm)


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
        U_m = np.max(u_data[:, 2])
        self.U_m = U_m

        return G_val, U_m, u_bar, p_bar_tilde


    def plot(self):

        nx, nz = 128, 128

        coords = self.mesh2d.coordinates.dat.data_ro
        x_i = np.linspace(float(coords[:, 0].min()), float(coords[:, 0].max()), nx)
        z_i = np.linspace(float(coords[:, 1].min()), float(coords[:, 1].max()), nz)
        X_i, Z_i = np.meshgrid(x_i, z_i)

        plotting_points = np.column_stack([X_i.ravel(), Z_i.ravel()])

        evaluator = PointEvaluator(self.mesh2d, plotting_points)
        U_at_list = evaluator.evaluate(self.u_bar)

        u_at = np.array(U_at_list, dtype=float, copy=True)

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

        stream = ax.streamplot(xi_plot, zi_plot, U_r, U_z, density=1.4, color=Speed, linewidth=lw, cmap="viridis", arrowsize=1.2, minlength=0.1)

        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="4%", pad=0.1)
        cbar1 = fig.colorbar(cf, cax=cax1)
        cbar1.set_label(r"$u_\theta$")

        cax2 = divider.append_axes("right", size="4%", pad=0.7)
        cbar2 = fig.colorbar(stream.lines, cax=cax2)
        cbar2.set_label(r"$\sqrt{u_r^2 + u_z^2}$")

        plt.tight_layout()
        plt.savefig("background_flow.png")
        plt.show()


    def plot_mesh(self, filename="mesh_2d_128.png", figsize=(6, 6), linewidth=0.15):
        # Underlying mesh lives at [0, W] x [0, H]; tick labels are shifted to
        # the centred [-W/2, W/2] x [-H/2, H/2] convention used elsewhere in
        # the case study, without modifying the mesh coordinates themselves.
        from firedrake.pyplot import triplot

        fig, ax = plt.subplots(figsize=figsize)
        triplot(
            self.mesh2d,
            axes=ax,
            interior_kw={"linewidths": linewidth, "edgecolors": "black", "facecolors": "none"},
            boundary_kw={"linewidths": 0.8, "colors": "black"},
        )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$z$")

        xticks = np.linspace(0, self.W, 5)
        yticks = np.linspace(0, self.H, 5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{x - self.W / 2:g}" for x in xticks])
        ax.set_yticklabels([f"{y - self.H / 2:g}" for y in yticks])

        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)


def build_3d_background_flow(R, H, W, G, mesh3d, tags, u_bar_2d, p_bar_2d):

    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d = FunctionSpace(mesh3d, "CG", 1)

    u_bar_3d = Function(V_3d)
    p_bar_3d = Function(Q_3d)

    V_coords = VectorFunctionSpace(mesh3d, "CG", 2)
    coords_func_u = Function(V_coords).interpolate(SpatialCoordinate(mesh3d))
    xyz_nodes_u = coords_func_u.dat.data_ro

    r_3d = np.sqrt(xyz_nodes_u[:, 0] ** 2 + xyz_nodes_u[:, 1] ** 2)
    theta_3d = np.arctan2(xyz_nodes_u[:, 1], xyz_nodes_u[:, 0])
    z_3d = xyz_nodes_u[:, 2]

    r_cs = r_3d - R
    z_cs = z_3d

    x_query_u = r_cs + 0.5 * W
    y_query_u = z_cs + 0.5 * H

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

    u_bar_3d.dat.data[:] = np.column_stack((u_x_3d, u_y_3d, u_z_3d))

    # Enforce u_3d == 0 on the boundary
    bc_walls = DirichletBC(V_3d, Constant((0.0, 0.0, 0.0)), tags["walls"])
    bc_walls.apply(u_bar_3d)

    Q_coords = VectorFunctionSpace(mesh3d, "CG", 1)
    coords_func_p = Function(Q_coords).interpolate(SpatialCoordinate(mesh3d))
    xyz_nodes_p = coords_func_p.dat.data_ro

    r_3d_p = np.sqrt(xyz_nodes_p[:, 0] ** 2 + xyz_nodes_p[:, 1] ** 2)
    theta_3d_p = np.arctan2(xyz_nodes_p[:, 1], xyz_nodes_p[:, 0])
    z_3d_p = xyz_nodes_p[:, 2]
    r_cs_p = r_3d_p - R

    x_query_p = np.clip(r_cs_p + 0.5 * W, 0.0, W)
    y_query_p = np.clip(z_3d_p + 0.5 * H, 0.0, H)

    query_points_p = np.column_stack((x_query_p, y_query_p))

    p_evaluator = PointEvaluator(p_bar_2d.function_space().mesh(), query_points_p, tolerance=1e-8)
    p_vals = np.array(p_evaluator.evaluate(p_bar_2d))

    if p_vals.ndim > 1:
        p_vals = p_vals.flatten()

    p_total = p_vals - G * R * theta_3d_p
    p_bar_3d.dat.data[:] = p_total

    return u_bar_3d, p_bar_3d


def run_mms_convergence(R, Re, W, H, Ns=(8, 16, 32, 64, 128)):
    """Method of Manufactured Solutions for the cylindrical background-flow
    operator solved in ``solve_2D_background_flow`` (the G-driving term and the
    flow-rate constraint are dropped; the pressure is fixed up to a constant via
    the constant null space). A smooth field is substituted into the weak form to
    produce a consistent body force; the problem is re-solved with that force and
    the matching Dirichlet data, and the velocity/pressure errors are measured
    under uniform refinement. Prints L2/H1 velocity and L2 pressure errors and
    the observed convergence rates (expected 3, 2, 2 for Taylor-Hood P2/P1).
    Fills Table~\\ref{tab:mms_bg} of the thesis."""
    import math

    def _residual(u, p, v, q, r):
        Rr = R + r
        u_r, u_z, u_th = u[0], u[1], u[2]
        v_r, v_z, v_th = v[0], v[1], v[2]
        F_cont = q * (Dx(u_r, 0) + Dx(u_z, 1) + u_r / Rr) * Rr * dx
        F_r = ((u_r * Dx(u_r, 0) + u_z * Dx(u_r, 1) - (u_th ** 2) / Rr) * v_r
               + Dx(p, 0) * v_r
               + (1.0 / Re) * dot(grad(u_r), grad(v_r))
               + (1.0 / Re) * (u_r / Rr ** 2) * v_r) * Rr * dx
        F_th = ((u_r * Dx(u_th, 0) + u_z * Dx(u_th, 1) + (u_r * u_th) / Rr) * v_th
                + (1.0 / Re) * dot(grad(u_th), grad(v_th))
                + (1.0 / Re) * (u_th / Rr ** 2) * v_th) * Rr * dx
        F_z = ((u_r * Dx(u_z, 0) + u_z * Dx(u_z, 1)) * v_z
               + Dx(p, 1) * v_z
               + (1.0 / Re) * dot(grad(u_z), grad(v_z))) * Rr * dx
        return F_cont + F_r + F_th + F_z

    print(f"\nMMS convergence (background flow): R={R}, Re={Re}, W={W}, H={H}")
    header = f"{'N':>5} {'h':>11} {'|e_u|_L2':>12} {'rate':>6} {'|e_u|_H1':>12} {'rate':>6} {'|e_p|_L2':>12} {'rate':>6}"
    print(header)
    prev = None
    for N in Ns:
        mesh = RectangleMesh(N, N, W, H, quadrilateral=False, diagonal="crossed")
        V = VectorFunctionSpace(mesh, "CG", 2, dim=3)
        Q = FunctionSpace(mesh, "CG", 1)
        Z = V * Q
        w = Function(Z)
        u, p = split(w)
        v, q = TestFunctions(Z)
        x = SpatialCoordinate(mesh)
        r = x[0] - 0.5 * W

        kx, kz = np.pi / W, np.pi / H
        ur_m = sin(kx * x[0]) * cos(kz * x[1])
        uz_m = -cos(kx * x[0]) * sin(kz * x[1])
        uth_m = sin(kx * x[0]) * sin(kz * x[1]) + 1.0
        p_m = cos(kx * x[0]) * cos(kz * x[1])
        u_m = as_vector([ur_m, uz_m, uth_m])

        # forcing = weak residual evaluated at the manufactured solution
        F = _residual(u, p, v, q, r) - _residual(u_m, p_m, v, q, r)

        bc = DirichletBC(Z.sub(0), u_m, "on_boundary")
        nullspace = MixedVectorSpaceBasis(
            Z, [Z.sub(0), VectorSpaceBasis(constant=True, comm=mesh.comm)])
        w.sub(0).interpolate(u_m)  # initial guess for Newton
        problem = NonlinearVariationalProblem(F, w, bcs=[bc])
        solver = NonlinearVariationalSolver(
            problem, nullspace=nullspace,
            solver_parameters={"snes_type": "newtonls",
                               "ksp_type": "preonly",
                               "pc_type": "lu",
                               "pc_factor_mat_solver_type": "mumps"})
        solver.solve()
        uh, ph = w.subfunctions

        e_u_L2 = math.sqrt(assemble(inner(uh - u_m, uh - u_m) * dx))
        e_u_H1 = math.sqrt(assemble((inner(uh - u_m, uh - u_m)
                                     + inner(grad(uh - u_m), grad(uh - u_m))) * dx))
        area = assemble(Constant(1.0) * dx(domain=mesh))
        shift = assemble((ph - p_m) * dx) / area
        e_p_L2 = math.sqrt(assemble((ph - p_m - shift) ** 2 * dx))

        h = max(W, H) / N
        if prev is None:
            print(f"{N:5d} {h:11.4e} {e_u_L2:12.4e} {'--':>6} {e_u_H1:12.4e} {'--':>6} {e_p_L2:12.4e} {'--':>6}")
        else:
            hp, eu2p, euhp, ep2p = prev
            ru2 = math.log(eu2p / e_u_L2) / math.log(2.0)
            ruh = math.log(euhp / e_u_H1) / math.log(2.0)
            rp2 = math.log(ep2p / e_p_L2) / math.log(2.0)
            print(f"{N:5d} {h:11.4e} {e_u_L2:12.4e} {ru2:6.2f} {e_u_H1:12.4e} {ruh:6.2f} {e_p_L2:12.4e} {rp2:6.2f}")
        prev = (h, e_u_L2, e_u_H1, e_p_L2)


if __name__ == "__main__":

    bf = background_flow(R, H, W, Re)
    G_val, U_m, u_2d, p_2d = bf.solve_2D_background_flow()

    bf.plot()
    bf.plot_mesh()

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, H, W, L_rel * max(H, W), a, particle_maxh_rel * a, global_maxh_rel * min(H, W))

    u_3d, p_3d = build_3d_background_flow(R, H, W, G_val, mesh3d, tags, u_2d, p_2d)

    print("3D mapping consistency at multiple points:")
    theta_max = L_rel * max(H, W) / R
    test_points_2d = [
        (0.5 * W, 0.5 * H),
        (0.5 * W, 0.25 * H),
        (0.25 * W, 0.5 * H),
        (0.75 * W, 0.25 * H),
    ]
    theta_samples = [0.1 * theta_max, 0.3 * theta_max, 0.7 * theta_max, 0.9 * theta_max]

    eval_2d = PointEvaluator(bf.mesh2d, np.array(test_points_2d))
    vals_2d = np.array(eval_2d.evaluate(u_2d))

    print(f"{'(r, z)':>18} {'θ':>10} {'|u|_2D':>12} {'|u|_3D':>12} {'|Δ|':>10}")
    max_err = 0.0
    for i, (xx, yy) in enumerate(test_points_2d):
        r_local = xx - 0.5 * W
        z_local = yy - 0.5 * H
        speed_2d = float(np.linalg.norm(vals_2d[i]))
        for theta in theta_samples:
            X = (R + r_local) * np.cos(theta)
            Y = (R + r_local) * np.sin(theta)
            Z = z_local
            eval_3d = PointEvaluator(mesh3d, np.array([[X, Y, Z]]))
            val_3d = np.array(eval_3d.evaluate(u_3d))[0]
            speed_3d = float(np.linalg.norm(val_3d))
            diff = abs(speed_2d - speed_3d)
            max_err = max(max_err, diff)
            print(f"({r_local:+.3f}, {z_local:+.3f})  {theta:10.4f}  {speed_2d:12.6f}  {speed_3d:12.6f}  {diff:10.2e}")
    print(f"-> max |Δ|u|| = {max_err:.2e}")

    # --- Verification: MMS convergence study (fills Table tab:mms_bg) ---
    run_mms_convergence(R, Re, W, H)
    print(f"-> max |Δ|u|| = {max_err:.2e}")