from tqdm import tqdm
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
import numpy as np

from background_flow import *
from build_3d_geometry import *
from perturbed_flow import *




class F_p_grid:

    def __init__(self, R, H, W, a, G, Re_p, L, particle_maxh, global_maxh, eps):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.G = G
        self.Re_p = Re_p
        self.L = L

        self.particle_maxh = particle_maxh
        self.global_maxh = global_maxh
        self.eps = eps

        self.r_min = -W / 2 + a + eps
        self.r_max = W / 2 - a - eps
        self.z_min = -H / 2 + a + eps
        self.z_max = H / 2 - a - eps

        self.mesh_nx = 120
        self.mesh_ny = 120

        self._current_mesh = None
        self._original_coords = None
        self._current_r = None
        self._current_z = None


    def compute_F_p_grid_ensemble(self, N_r, N_z, u_bg_data_np, p_bg_data_np):

        ensemble = Ensemble(COMM_WORLD, 1)
        my_comm = ensemble.comm

        global_rank = COMM_WORLD.rank
        global_size = COMM_WORLD.size

        mesh2d_local = RectangleMesh(self.mesh_nx, self.mesh_ny, self.W, self.H, quadrilateral=False, comm=my_comm)

        V_local = VectorFunctionSpace(mesh2d_local, "CG", 2, dim=3)
        Q_local = FunctionSpace(mesh2d_local, "CG", 1)

        u_bg_local = Function(V_local)
        p_bg_local = Function(Q_local)

        u_bg_local.dat.data[:] = u_bg_data_np
        p_bg_local.dat.data[:] = p_bg_data_np

        r_vals = np.linspace(self.r_min, self.r_max, N_r)
        z_vals = np.linspace(self.z_min, self.z_max, N_z)

        all_tasks = []
        for i in range(N_r):
            for j in range(N_z):
                all_tasks.append((i, j, r_vals[i], z_vals[j]))

        my_tasks = all_tasks[global_rank::global_size]
        local_results = []

        if global_rank == 0:
            print(f"Start ensemble grid: {len(all_tasks)} points on {global_size} cores.")

        for task in tqdm(my_tasks, disable=(global_rank != 0)):
            (i, j, r_loc, z_loc) = task

            try:
                mesh3d, tags = make_curved_channel_section_with_spherical_hole(
                    self.R, self.H, self.W, self.L, self.a,
                    self.particle_maxh, self.global_maxh,
                    r_off=r_loc, z_off=z_loc,
                    comm=my_comm
                )

                u_3d, p_3d = build_3d_background_flow(self.R, self.H, self.W, self.G, mesh3d, u_bg_local, p_bg_local)

                pf = perturbed_flow(self.R, self.H, self.W, self.a, self.Re_p, mesh3d, tags, u_3d, p_3d)
                F_r, F_z = pf.F_p()

                local_results.append((i, j, F_r, F_z))

            except Exception as e:
                print(f"[Rank {global_rank}] Error at {i},{j}: {e}")
                local_results.append((i, j, 0.0, 0.0))


        all_data = COMM_WORLD.gather(local_results, root=0)


        if global_rank == 0:
            Fr_grid = np.zeros((N_r, N_z))
            Fz_grid = np.zeros((N_r, N_z))

            for rank_result_list in all_data:
                for (i, j, Fr, Fz) in rank_result_list:
                    Fr_grid[i, j] = Fr
                    Fz_grid[i, j] = Fz

            self.Fr_grid = Fr_grid
            self.Fz_grid = Fz_grid

            phi = np.sqrt(Fr_grid ** 2 + Fz_grid ** 2)
            self.phi = phi

            self.r_vals = r_vals
            self.z_vals = z_vals

            return r_vals, z_vals, phi, Fr_grid, Fz_grid
        else:
            return None, None, None, None, None


    def generate_initial_guesses(self, n_grid_search=50, tol_unique=1e-3, tol_residual=1e-5):

        self.interp_Fr = RectBivariateSpline(self.r_vals, self.z_vals, self.Fr_grid)
        self.interp_Fz = RectBivariateSpline(self.r_vals, self.z_vals, self.Fz_grid)

        def _interpolated_coarse_grid(x):
            r, z = x
            return [self.interp_Fr(r, z)[0, 0], self.interp_Fz(r, z)[0, 0]]

        r_starts = np.linspace(self.r_min, self.r_max, n_grid_search)
        z_starts = np.linspace(self.z_min, self.z_max, n_grid_search)

        initial_guesses = []

        for r0 in r_starts:
            for z0 in z_starts:

                # finds a root on the interpolated function. It uses a hybrid method between GD and newton and starts with (r0, z0).
                solution = root(_interpolated_coarse_grid, [r0, z0], method='hybr')

                # returns a boolean expression depending on the success of the root search
                if solution.success:
                    r_sol, z_sol = solution.x

                    if not (self.r_min <= r_sol <= self.r_max and self.z_min <= z_sol <= self.z_max):
                        continue

                    if np.linalg.norm(_interpolated_coarse_grid((r_sol, z_sol))) > tol_residual:
                        continue

                    is_new = True
                    for existing in initial_guesses:
                        dist = np.linalg.norm(np.array([r_sol, z_sol]) - existing)
                        if dist < tol_unique:
                            is_new = False
                            break

                    if is_new:
                        initial_guesses.append(np.array([r_sol, z_sol]))

        self.initial_guesses = initial_guesses

        return initial_guesses


    def classify_equilibria_on_grid(self, equilibria):

        classified_equilibria = []

        for k, x_eq in enumerate(equilibria, start=1):
            r, z = x_eq[0], x_eq[1]

            dFr_dr = self.interp_Fr(r, z, dx=1, dy=0)[0, 0]
            dFr_dz = self.interp_Fr(r, z, dx=0, dy=1)[0, 0]
            dFz_dr = self.interp_Fz(r, z, dx=1, dy=0)[0, 0]
            dFz_dz = self.interp_Fz(r, z, dx=0, dy=1)[0, 0]

            J = np.array([
                [dFr_dr, dFr_dz],
                [dFz_dr, dFz_dz]
            ])

            eigvals, eigvecs = np.linalg.eig(J)
            real_parts = eigvals.real

            if real_parts[0] * real_parts[1] < 0:
                eq_type = "saddle"
                color = "yellow"
            elif np.all(real_parts < 0):
                eq_type = "stable"
                color = "green"
            elif np.all(real_parts > 0):
                eq_type = "unstable"
                color = "red"

            info = {
                "x_eq": x_eq,
                "type": eq_type,
                "color": color
            }

            classified_equilibria.append(info)

        return classified_equilibria


    def plot(self, L_c_p, L_c, initial_guesses=None, classified_equilibria=None):

        R_grid, Z_grid = np.meshgrid(self.r_vals, self.z_vals, indexing='ij')

        exclusion_dist = self.a + self.eps

        fig, ax = plt.subplots(figsize=(8, 6))

        levels = np.linspace(np.nanmin(self.phi), np.nanmax(self.phi), 40)

        cs = ax.contourf(R_grid, Z_grid, self.phi, levels=levels, cmap="viridis", alpha=1)

        cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Force Magnitude $\|\mathbf{F}\|$")

        ax.contour(R_grid, Z_grid, self.Fr_grid, levels=[0], colors="black", linestyles="-", linewidths=2)
        ax.contour(R_grid, Z_grid, self.Fz_grid, levels=[0], colors="white", linestyles="-", linewidths=2)

        wall_rect = patches.Rectangle((-self.W / 2, -self.H / 2), self.W, self.H,
                                      linewidth=3, edgecolor='black', facecolor='none', zorder=10)
        ax.add_patch(wall_rect)

        ax.add_patch(patches.Rectangle((-self.W / 2, -self.H / 2), exclusion_dist, self.H,
                                       facecolor='gray', alpha=0.3, hatch='///'))

        ax.add_patch(patches.Rectangle((self.W / 2 - exclusion_dist, -self.H / 2), exclusion_dist, self.H,
                                       facecolor='gray', alpha=0.3, hatch='///'))

        ax.add_patch(patches.Rectangle((-self.W / 2, -self.H / 2), self.W, exclusion_dist,
                                       facecolor='gray', alpha=0.3, hatch='///'))

        ax.add_patch(patches.Rectangle((-self.W / 2, self.H / 2 - exclusion_dist), self.W, exclusion_dist,
                                       facecolor='gray', alpha=0.3, hatch='///'))

        if initial_guesses is not None:
            guesses = np.array(initial_guesses)
            if guesses.ndim == 1 and len(guesses) == 2:
                ax.scatter(guesses[0], guesses[1], c='red', marker='x', s=100, linewidths=2, label='Initial Guess', zorder=20)
            elif guesses.ndim == 2:
                ax.scatter(guesses[:, 0], guesses[:, 1], c='red', marker='x', s=100, linewidths=2, label='Initial Guesses', zorder=20)


        if classified_equilibria is not None:

            plotted_types = set()

            for eq in classified_equilibria:
                r_eq, z_eq = eq["x_eq"]
                color = eq["color"]
                eq_type = eq["type"]

                label = None
                if eq_type not in plotted_types:
                    label = f"Equilibrium ({eq_type})"
                    plotted_types.add(eq_type)

                ax.scatter(r_eq, z_eq, c=color, marker='o', s=150, edgecolors='black', linewidths=1.5, label=label, zorder=30)

        ax.legend(loc='upper right', framealpha=1.0, fontsize='small')

        margin = self.eps
        ax.set_xlim(-self.W / 2 - margin, self.W / 2 + margin)
        ax.set_ylim(-self.H / 2 - margin, self.H / 2 + margin)
        ax.set_aspect('equal')
        ax.set_xlabel("r")
        ax.set_ylabel("z")

        # Just for visualisation purposes
        a = (L_c_p/L_c) * self.a * 2
        R = (L_c_p/L_c) * self.R * 2

        ax.set_title(f"Force_Map_a={a:.3f}_R={R:.1f}_with_H=W=2")
        plt.tight_layout()
        filename = f"images/Force_Map_a={a:.3f}_R={R:.1f}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()



class F_p_exact_roots:

    def __init__(self, R, W, H, G, L, a, particle_maxh, global_maxh, Re, Re_p, u_bar, p_bar, eps):
        self.R, self.W, self.H, self.L = R, W, H, L
        self.a, self.particle_maxh, self.global_maxh = a, particle_maxh, global_maxh
        self.Re, self.Re_p = Re, Re_p
        self.u_bar, self.p_bar = u_bar, p_bar
        self.G = G

        self.eps = eps

        self.r_min = -0.5 * self.W + self.a + self.eps
        self.r_max = 0.5 * self.W - self.a - self.eps
        self.z_min = -0.5 * self.H + self.a + self.eps
        self.z_max = 0.5 * self.H - self.a - self.eps

        self.mesh2d_static = RectangleMesh(120, 120, self.W, self.H, quadrilateral=False, comm=COMM_SELF)

        V_2d = VectorFunctionSpace(self.mesh2d_static, "CG", 2, dim=3)
        Q_2d = FunctionSpace(self.mesh2d_static, "CG", 1)

        self.u_2d_func = Function(V_2d)
        self.p_2d_func = Function(Q_2d)

        self.u_2d_func.dat.data[:] = u_bar
        self.p_2d_func.dat.data[:] = p_bar

        self._current_mesh = None
        self._current_tags = None
        self._current_r = None
        self._current_z = None
        self._original_coords = None


    @staticmethod
    def move_mesh_elasticity(mesh3d, tags, displacement_vector):

        # We set up a new function space to solve the elasticity equation for the mesh deformation function u
        V_disp = VectorFunctionSpace(mesh3d, "CG", 1)
        u = TrialFunction(V_disp)
        v = TestFunction(V_disp)

        x = SpatialCoordinate(mesh3d)
        cx, cy, cz = tags["particle_center"]
        r_dist = sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2)

        stiffness = 1.0 / (r_dist ** 2 + 0.02)

        mu = Constant(1.0) * stiffness
        lmbda = Constant(1.0) * stiffness

        # We want to view the mesh as if the edges would be made of rubber or as if they would be springs
        # Therefore they obey Hooks Law div(sigma) = 0
        # In the weak formulation we have grad(v) = 0.5 * (grad(v) + grad(v).T) + 0.5 * (grad(v) - grad(v).T)
        # Since (sigma : (grad(v) - grad(v).T)) = 0 always, we can eleviate this term.

        def epsilon(v):
            return 0.5 * (grad(v) + grad(v).T)

        def sigma(u):
            return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)

        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(Constant((0, 0, 0)), v) * dx

        bcs_walls = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["walls"])
        bcs_in = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["inlet"])
        bcs_out = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["outlet"])

        disp_const = Constant(displacement_vector)
        bc_part = DirichletBC(V_disp, disp_const, tags["particle"])

        bcs = [bcs_walls, bcs_in, bcs_out, bc_part]

        displacement_sol = Function(V_disp)
        solve(a == L, displacement_sol, bcs=bcs,
              solver_parameters=
              {
                  "ksp_type": "preonly",
                  "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"
              }
            )

        V_coords = mesh3d.coordinates.function_space()

        displacement_high_order = Function(V_coords)

        displacement_high_order.interpolate(displacement_sol)

        mesh3d.coordinates.assign(mesh3d.coordinates + displacement_high_order)

        return displacement_sol

    @staticmethod
    def approx_jacobian(F, x, Fx=None, eps_rel=1e-3, eps_abs=1e-4, r_min=None, r_max=None, z_min=None, z_max=None):

        x = np.asarray(x, float)

        if Fx is None:
            Fx = F(x)

        clamp = not (r_min is None or r_max is None or z_min is None or z_max is None)
        J = np.zeros((2, 2), float)

        for i in range(2):
            h = max(eps_rel * (1.0 + abs(x[i])), eps_abs)

            dx = np.zeros(2)
            dx[i] = h

            xp = x + dx
            xm = x - dx

            if clamp:
                xp[0] = np.clip(xp[0], r_min, r_max)
                xp[1] = np.clip(xp[1], z_min, z_max)
                xm[0] = np.clip(xm[0], r_min, r_max)
                xm[1] = np.clip(xm[1], z_min, z_max)

            fp = F(xp)
            fm = F(xm)

            denom = xp[i] - xm[i]
            if abs(denom) < 1e-14:
                denom = h
                fp = F(x + dx)
                J[:, i] = (fp - Fx) / denom
            else:
                J[:, i] = (fp - fm) / denom

        return J


    def _get_mesh_at(self, r, z):

        if self._current_mesh is None:
            if self._check_inside_box(r, z):
                print(f"Generating BASE mesh at r={r:.4f}, z={z:.4f}")
                mesh, tags = make_curved_channel_section_with_spherical_hole(
                    self.R, self.H, self.W, self.L, self.a,
                    self.particle_maxh, self.global_maxh,
                    r_off=r, z_off=z
                )
                self._current_mesh = mesh
                self._current_tags = tags
                self._current_r = r
                self._current_z = z

                self._original_coords = Function(mesh.coordinates)
                return mesh, tags
            else:
                raise ValueError("Initial guess outside box")

        self._current_mesh.coordinates.assign(self._original_coords)

        dr = r - self._current_r
        dz = z - self._current_z

        dist = sqrt(dr ** 2 + dz ** 2)
        if dist > 2.0 * self.a:
            print(f"Deformation too large ({dist:.3f} > {2 * self.a:.3f}). Remeshing base...")
            self._current_mesh = None
            return self._get_mesh_at(r, z)

        theta_p = (self.L / self.R) * 0.5

        dx = dr * cos(theta_p)
        dy = dr * sin(theta_p)
        dz_disp = dz

        disp_vec = [dx, dy, dz_disp]

        self.move_mesh_elasticity(self._current_mesh, self._current_tags, disp_vec)

        return self._current_mesh, self._current_tags


    def evaluate_F(self, x):
        r, z = float(x[0]), float(x[1])

        if not self._check_inside_box(r, z):
            raise ValueError(f"Out of bounds: {r}, {z}")

        mesh3d, tags = self._get_mesh_at(r, z)



        u_3d, p_3d = build_3d_background_flow(self.R, self.H, self.W, self.G,
                                                  mesh3d, self.u_2d_func, self.p_2d_func)


        pf = perturbed_flow(self.R, self.H, self.W, self.a, self.Re_p, mesh3d, tags, u_3d, p_3d)

        Fr, Fz = pf.F_p()

        return np.array([Fr, Fz], dtype=float)


    def _check_inside_box(self, r, z):
        return (self.r_min <= r <= self.r_max) and (self.z_min <= z <= self.z_max)


    def classify_single_equilibrium(self, x_eq, eps_rel=1e-2, eps_abs=1e-3, ode_sign=1.0, tol_eig=1e-6):

        x_eq = np.asarray(x_eq, dtype=float)

        def G(x):
            F = np.asarray(self.evaluate_F(x), dtype=float)
            return ode_sign * F

        Gx = G(x_eq)
        J = self.approx_jacobian(G, x_eq, Fx=Gx, eps_rel=eps_rel, eps_abs=eps_abs)

        eigvals, eigvecs = np.linalg.eig(J)
        real_parts = eigvals.real

        tr = np.trace(J)
        det = np.linalg.det(J)

        if np.any(np.isnan(real_parts)):
            eq_type = "unclear (NaN in eigenvalues)"
        else:
            if real_parts[0] * real_parts[1] < -tol_eig ** 2:
                eq_type = "saddle"
            elif np.all(real_parts < -tol_eig):
                eq_type = "stable"
            elif np.all(real_parts > tol_eig):
                eq_type = "unstable"
            else:
                eq_type = "unclear"

        return {
            "x_eq": x_eq,
            "J": J,
            "eigvals": eigvals,
            "trace": tr,
            "det": det,
            "type": eq_type,
        }


    def classify_equilibria(self, equilibria, ode_sign=-1.0, tol_eig=1e-6, eps_rel=1e-2, eps_abs=1e-3, verbose=True):

        equilibria = np.asarray(equilibria, dtype=float)
        if equilibria.ndim == 1:
            equilibria = equilibria[None, :]

        classified_equilibria = []

        for k, x_eq in enumerate(equilibria, start=1):

            self._current_mesh = None
            self._current_r = None
            self._current_z = None

            info = self.classify_single_equilibrium(x_eq, ode_sign=ode_sign, tol_eig=tol_eig,
                                                    eps_rel=eps_rel, eps_abs=eps_abs)

            if info["type"] == "stable":
                info["color"] = "green"
            elif info["type"] == "unstable":
                info["color"] = "red"
            elif info["type"] == "saddle":
                info["color"] = "yellow"
            else:
                info["color"] = "gray"

            classified_equilibria.append(info)

            if verbose:
                ev = info["eigvals"]
                print(f"EQ #{k}: x_eq = ({x_eq[0]:.6f}, {x_eq[1]:.6f})")
                print(f"        eigenvalues(J_dyn): "
                      f"{ev[0].real:+.3e}{ev[0].imag:+.3e}i, "
                      f"{ev[1].real:+.3e}{ev[1].imag:+.3e}i")
                print(f"        Typ: {info['type']}\n")

        return classified_equilibria