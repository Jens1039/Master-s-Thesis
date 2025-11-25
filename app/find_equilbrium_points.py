import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import os
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
from firedrake import *


from background_flow import background_flow
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow


class F_p_grid:

    def __init__(self, R, W, H, L, a, Re_nominal, Re_p, particle_maxh, global_maxh, eps, bg_flow=None, Q=1.0):

        self.R = float(R)
        self.W = float(W)
        self.H = float(H)
        self.L = float(L)
        self.a = float(a)
        self.Re_nominal = float(Re_nominal)
        self.Re_p = float(Re_p)
        self.particle_maxh = float(particle_maxh)
        self.global_maxh = float(global_maxh)
        self.eps = eps

        self.r_min = -W/2 + a + eps
        self.r_max = W/2 - a - eps
        self.z_min = -H/2 + a + eps
        self.z_max = H/2 - a - eps

        self.Q = float(Q)

        if bg_flow is None:
            self.bg_flow = background_flow(self.R, self.H, self.W, self.Q, self.Re_nominal)
            self.bg_flow.solve_2D_background_flow()
        else:
            self.bg_flow = bg_flow

    _BG_WORKER = None

    @staticmethod
    def _init_bg_worker(R, H, W, Q, Re_bg_input):
        global _BG_WORKER
        _BG_WORKER = background_flow(R, H, W, Q, Re_bg_input)
        _BG_WORKER.solve_2D_background_flow()

    @staticmethod
    def _compute_single_F_p(task):

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NETGEN_NUM_THREADS"] = "1"

        (i, j, r_loc, z_loc,
         R, H, W, Q, L, a,
         particle_maxh, global_maxh, Re_p_correct) = task

        mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, W, H, L, a, particle_maxh, global_maxh, r_off=r_loc, z_off=z_loc)

        global _BG_WORKER
        pf = perturbed_flow(mesh3d, tags, a, Re_p_correct, _BG_WORKER)

        F_vec = pf.F_p()

        cx, cy, cz = tags["particle_center"]
        r0 = float(np.hypot(cx, cy))
        if r0 < 1e-14:
            ex0 = np.array([1., 0., 0.])
        else:
            ex0 = np.array([cx / r0, cy / r0, 0.])
        ez0 = np.array([0., 0., 1.])

        Fr = float(ex0 @ F_vec)
        Fz = float(ez0 @ F_vec)
        return (i, j, Fr, Fz)


    def compute_F_p_grid(self, N_r, N_z, nproc=mp.cpu_count()):

        r_vals = np.linspace(self.r_min, self.r_max, N_r)
        z_vals = np.linspace(self.z_min, self.z_max, N_z)

        Fr_grid = np.zeros((N_r, N_z))
        Fz_grid = np.zeros((N_r, N_z))

        tasks = [
            (i, j, r_vals[i], z_vals[j],
             self.R, self.H, self.W, self.Q, self.L, self.a,
             self.particle_maxh, self.global_maxh, self.Re_p)
            for i in range(N_r) for j in range(N_z)
        ]

        print(f"Start parallel coarse grid.")
        with mp.Pool(
                processes=nproc,
                initializer=self._init_bg_worker,
                initargs=(self.R, self.H, self.W, self.Q, self.Re_nominal)
        ) as pool:

            results = []
            for res in tqdm(pool.imap_unordered(self._compute_single_F_p, tasks), total=len(tasks)):
                results.append(res)

        for (i, j, Fr, Fz) in results:
            Fr_grid[i, j] = Fr
            Fz_grid[i, j] = Fz

        phi = np.sqrt(Fr_grid ** 2 + Fz_grid ** 2)

        self.r_vals = r_vals
        self.z_vals = z_vals
        self.phi = phi
        self.Fr_grid = Fr_grid
        self.Fz_grid = Fz_grid

        return r_vals, z_vals, phi, Fr_grid, Fz_grid


    def plot_paper_reproduction(self, r_vals, z_vals, phi, Fr_grid, Fz_grid, invert_xaxis=True):

        R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

        exclusion_dist = self.a + self.eps

        fig, ax = plt.subplots(figsize=(8, 8))

        levels = np.linspace(phi.min(), phi.max(), 40)
        cs = ax.contourf(R_grid, Z_grid, phi, levels=levels, cmap="viridis", alpha=0.9)
        cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Force Magnitude $\|\mathbf{F}\|$")

        ax.contour(R_grid, Z_grid, Fr_grid, levels=[0], colors="cyan", linestyles="--", linewidths=2)
        ax.contour(R_grid, Z_grid, Fz_grid, levels=[0], colors="magenta", linestyles="-", linewidths=2)

        wall_rect = patches.Rectangle((-self.W / 2, -self.H / 2), self.W, self.H,
                                      linewidth=3, edgecolor='black', facecolor='none', zorder=10)
        ax.add_patch(wall_rect)

        for xy, w, h in [
            ((-self.W / 2, -self.H / 2), exclusion_dist, self.H),
            ((self.W / 2 - exclusion_dist, -self.H / 2), exclusion_dist, self.H),
            ((-self.W / 2, -self.H / 2), self.W, exclusion_dist),
            ((-self.W / 2, self.H / 2 - exclusion_dist), self.W, exclusion_dist)
        ]:
            ax.add_patch(patches.Rectangle(xy, w, h, facecolor='gray', alpha=0.3, hatch='///'))

        margin = 0.1
        ax.set_xlim(-self.W / 2 - margin, self.W / 2 + margin)
        ax.set_ylim(-self.H / 2 - margin, self.H / 2 + margin)
        ax.set_aspect('equal')
        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title("Figure 2 Reproduction")

        if invert_xaxis:
            ax.invert_xaxis()

        plt.tight_layout()
        plt.show()

    # Uses a hybrid GD Newton method to find the roots on the interpolated force grid
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
                    # solution.x stores the solution coordinates r and z while "solution" stores all the information
                    # about the executed solution attempt
                    r_sol, z_sol = solution.x

                    if not (self.r_min <= r_sol <= self.r_max and self.z_min <= z_sol <= self.z_max):
                        # The remainder of this iteration is skipped and we go right into the next iteration
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


    def plot_guesses_and_roots_on_grid(self, roots=None, stability_info=None, invert_xaxis=True):

        R_grid, Z_grid = np.meshgrid(self.r_vals, self.z_vals, indexing='ij')
        fig, ax = plt.subplots(figsize=(7, 6))

        levels = np.linspace(self.phi.min(), self.phi.max(), 30)
        cs = ax.contourf(R_grid, Z_grid, self.phi, levels=levels, cmap="viridis", alpha=0.8)
        plt.colorbar(cs, ax=ax, label=r"Force Magnitude $\|\mathbf{F}\|$")

        ax.contour(R_grid, Z_grid, self.Fr_grid, levels=[0], colors="cyan", linestyles="--", linewidths=1)
        ax.contour(R_grid, Z_grid, self.Fz_grid, levels=[0], colors="magenta", linestyles="-", linewidths=1)

        if hasattr(self, "initial_guesses") and len(self.initial_guesses) > 0:
            ig = np.asarray(self.initial_guesses)
            ax.scatter(ig[:, 0], ig[:, 1], c="black", s=80, marker="x", label="Initial guesses", zorder=10)

        if roots is not None:
            roots_arr = np.asarray(roots, dtype=float)
            if roots_arr.ndim == 1:
                roots_arr = roots_arr.reshape(1, -1)

            if stability_info is not None:

                for k, info in enumerate(stability_info):
                    r, z = info["x_eq"]
                    ax.scatter(r, z,
                               s=140, marker="o",
                               facecolors=info["color"],
                               edgecolors="black",
                               linewidths=1.5,
                               label=f"{info['type']}" if k == 0 else None,
                               zorder=11)
            else:
                # fallback: all red
                ax.scatter(roots_arr[:, 0], roots_arr[:, 1],
                           s=120, marker="o",
                           facecolors="red", edgecolors="black",
                           label="Exact roots", zorder=11)

        ax.set_xlabel("r")
        ax.set_ylabel("z")
        ax.set_title("Initial guesses and exact roots on coarse force grid")
        ax.set_aspect("equal")

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if invert_xaxis:
            ax.invert_xaxis()

        plt.tight_layout()
        plt.show()


class F_p_roots:

    def __init__(self, R, W, H, L, a, particle_maxh, global_maxh, Re, Re_p, bg_flow=None, Q=1.0, eps=0.025):
        self.R, self.W, self.H, self.L = R, W, H, L
        self.a, self.particle_maxh, self.global_maxh = a, particle_maxh, global_maxh
        self.Re, self.Re_p, self.Q, self.eps = Re, Re_p, Q, eps

        self.r_min = -0.5 * self.W + self.a + self.eps
        self.r_max = 0.5 * self.W - self.a - self.eps
        self.z_min = -0.5 * self.H + self.a + self.eps
        self.z_max = 0.5 * self.H - self.a - self.eps

        if bg_flow is None:
            self.bg_flow = background_flow(self.R, self.H, self.W, self.Q, self.Re)
            self.bg_flow.solve_2D_background_flow()
        else:
            self.bg_flow = bg_flow

        self._current_mesh = None
        self._current_tags = None
        self._current_r = None
        self._current_z = None
        self._original_coords = None

    @staticmethod
    def move_mesh_elasticity(mesh, tags, displacement_vector, verbose=False):

        V_disp = VectorFunctionSpace(mesh, "CG", 1)

        u = TrialFunction(V_disp)
        v = TestFunction(V_disp)

        # 2. Define Material Properties
        x = SpatialCoordinate(mesh)
        cx, cy, cz = tags["particle_center"]
        r_dist = sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2)

        # Stiffening factor: stiff near particle
        stiff = 1.0 / (r_dist ** 2 + 0.01)

        mu = Constant(1.0) * stiff
        lmbda = Constant(1.0) * stiff

        def epsilon(u):
            return 0.5 * (grad(u) + grad(u).T)

        def sigma(u):
            return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)

        # 3. Variational Form
        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(Constant((0, 0, 0)), v) * dx

        # 4. Boundary Conditions
        bc_walls = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["walls"])
        bc_in = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["inlet"])
        bc_out = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["outlet"])

        disp_const = Constant(displacement_vector)
        bc_part = DirichletBC(V_disp, disp_const, tags["particle"])

        bcs = [bc_walls, bc_in, bc_out, bc_part]

        # 5. Solve
        displacement_sol = Function(V_disp)
        solve(a == L, displacement_sol, bcs=bcs,
              solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', "pc_factor_mat_solver_type": "mumps"})

        # 6. Actually move the mesh coordinates

        # --- FIX START ---
        # Get the function space of the ACTUAL mesh coordinates (which is Degree 3)
        V_coords = mesh.coordinates.function_space()

        # Create a function in that high-order space
        displacement_high_order = Function(V_coords)

        # Interpolate the linear solution onto the high-order space
        displacement_high_order.interpolate(displacement_sol)

        # Now we can add them because they are in the same function space
        mesh.coordinates.assign(mesh.coordinates + displacement_high_order)
        # --- FIX END ---

        return displacement_sol

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

        pf = perturbed_flow(mesh3d, tags, self.a, self.Re_p, self.bg_flow)
        F_cart = pf.F_p()

        cx, cy, cz = tags["particle_center"]
        r0 = np.hypot(cx, cy)
        if r0 < 1e-14:
            ex0 = np.array([1.0, 0.0, 0.0])
        else:
            ex0 = np.array([cx / r0, cy / r0, 0.0])

        ez0 = np.array([0.0, 0.0, 1.0])

        Fr = float(ex0 @ F_cart)
        Fz = float(ez0 @ F_cart)

        return np.array([Fr, Fz], dtype=float)

    def _check_inside_box(self, r, z):
        return (self.r_min <= r <= self.r_max) and (self.z_min <= z <= self.z_max)

    @staticmethod
    def approx_jacobian(F, x, Fx=None, eps_rel=1e-3, eps_abs=1e-4,
                        r_min=None, r_max=None, z_min=None, z_max=None):

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

    @staticmethod
    def newton_monitor(iter, x, Fx, delta):
        print(f"[Newton iter {iter:02d}] " f"x = ({x[0]:.5f}, {x[1]:.5f}) | " f"|F| = {np.linalg.norm(Fx):.3e} | "
              f"|dx| = {np.linalg.norm(delta):.3e}")

    def newton_deflated_2d(self, x0, known_roots,
                           alpha=1e-2, p=2.0,
                           tol_F=1e-5, tol_x=1e-6, max_iter=15,
                           ls_max_steps=8, ls_reduction=0.5, ls_eta=1e-3,
                           defl_r_cut=0.01):
        """
        Newton mit Deflation:
        - Deflation wirkt über F_defl = d(x) * F_orig(x)
        - Jacobian auf F_defl
        - Linesearch + Akzeptanzbedingungen NUR auf F_orig
        - Erfolgskriterium: ||F_orig|| < tol_F
        """

        x = np.asarray(x0, dtype=float)

        def F_orig(x_vec):
            return np.asarray(self.evaluate_F(x_vec), dtype=float)

        roots = [np.asarray(rz, dtype=float) for rz in known_roots]

        def defl_factor_local(x_vec):
            """Lokaler, entschärfter Deflationsfaktor um bekannte Roots."""
            if not roots:
                return 1.0
            fac = 1.0
            for rstar in roots:
                dist = np.linalg.norm(x_vec - rstar)

                # Singularität entschärfen: dist_eff nie kleiner als defl_r_cut
                # => Plateau hoher, aber nicht explodierender Deflation
                if defl_r_cut is not None:
                    dist_eff = max(dist, defl_r_cut)
                else:
                    dist_eff = max(dist, 1e-12)

                fac *= (1.0 / (dist_eff ** p) + alpha)
            return fac

        def F_defl(x_vec):
            return defl_factor_local(x_vec) * F_orig(x_vec)

        def compute_alpha_max(x_vec, delta):
            alpha_max = 1.0
            r_min, r_max = self.r_min, self.r_max
            z_min, z_max = self.z_min, self.z_max

            for (xi, di, xmin, xmax) in [
                (x_vec[0], delta[0], r_min, r_max),
                (x_vec[1], delta[1], z_min, z_max),
            ]:
                if abs(di) < 1e-14:
                    continue
                if di > 0:
                    a_k = (xmax - xi) / di
                else:
                    a_k = (xmin - xi) / di
                alpha_max = min(alpha_max, a_k)

            return max(min(alpha_max, 1.0), 0.0)

        # Startwerte
        F0 = F_orig(x)
        F0_norm = np.linalg.norm(F0)
        F0_defl = defl_factor_local(x) * F0

        self.newton_monitor(0, x, F0, np.zeros_like(x))

        # Einziges sofortiges Erfolgskriterium
        if F0_norm < tol_F:
            return x, True

        for k in range(1, max_iter + 1):
            # Jacobian von F_defl
            J = self.approx_jacobian(
                F_defl,
                x,
                Fx=F0_defl,
                r_min=self.r_min,
                r_max=self.r_max,
                z_min=self.z_min,
                z_max=self.z_max,
            )

            try:
                delta = np.linalg.solve(J, -F0_defl)
            except np.linalg.LinAlgError:
                print(f"[Abort defl] LinAlgError (singular Jacobian) at iter {k}, x={x}")
                return x, False

            alpha_max = compute_alpha_max(x, delta)
            if alpha_max <= 0:
                print(f"[Abort defl] alpha_max <= 0 (boundary hit) at iter {k}, x={x}, delta={delta}")
                return x, False

            alpha_ls = alpha_max
            F_trial = None
            F_trial_norm = None

            # Linesearch NUR auf F_orig
            for _ in range(ls_max_steps):
                x_trial = x + alpha_ls * delta

                if not (self.r_min <= x_trial[0] <= self.r_max and
                        self.z_min <= x_trial[1] <= self.z_max):
                    alpha_ls *= ls_reduction
                    continue

                F_trial = F_orig(x_trial)
                F_trial_norm = np.linalg.norm(F_trial)

                # Erfolg in der Linesearch:
                # 1) Genauigkeit erreicht
                if F_trial_norm <= tol_F:
                    break

                # 2) Sufficient decrease in ||F_orig||
                if F_trial_norm <= F0_norm * (1.0 - ls_eta):
                    break

                # sonst Schritt verkleinern
                alpha_ls *= ls_reduction

            if F_trial is None:
                print(f"[Abort defl] Linesearch stagnation (no valid step) at iter {k}, x={x}, |F|={F0_norm:.3e}")
                return x, False

            if F_trial_norm is None or F_trial_norm >= F0_norm:
                print(f"[Abort defl] No improvement at iter {k}, x={x}, |F|={F0_norm:.3e}")
                return x, False

            # akzeptierter Schritt
            step = alpha_ls * delta
            x = x + step

            F0 = F_trial
            F0_norm = F_trial_norm
            F0_defl = defl_factor_local(x) * F0

            self.newton_monitor(k, x, F0, step)

            # EINZIGE Erfolgsbedingung während der Iteration
            if F0_norm < tol_F:
                return x, True

            if np.linalg.norm(step) < tol_x:
                print(f"[Stop defl] Step small at iter {k}, |dx|={np.linalg.norm(step):.3e}, "
                      f"|F|={F0_norm:.3e}, rejecting")
                return x, False

        # max_iter erreicht: Erfolg NUR wenn F klein ist
        if F0_norm < tol_F:
            print(f"[Stop defl] Max iters reached, but |F|={F0_norm:.3e} < tol_F, accepting")
            return x, True

        print(f"[Stop defl] Max iters reached, |F|={F0_norm:.3e}, rejecting")
        return x, False

    def newton_plain_2d(self, x0,
                        tol_F=1e-5, tol_x=1e-6,
                        max_iter=15,
                        ls_max_steps=8, ls_reduction=0.5, ls_eta=1e-3):

        x = np.asarray(x0, dtype=float)

        def F_orig(x_vec):
            return np.asarray(self.evaluate_F(x_vec), dtype=float)

        def compute_alpha_max(x_vec, delta):
            alpha_max = 1.0
            r_min, r_max = self.r_min, self.r_max
            z_min, z_max = self.z_min, self.z_max

            for (xi, di, xmin, xmax) in [
                (x_vec[0], delta[0], r_min, r_max),
                (x_vec[1], delta[1], z_min, z_max),
            ]:
                if abs(di) < 1e-14:
                    continue
                if di > 0:
                    a_k = (xmax - xi) / di
                else:
                    a_k = (xmin - xi) / di
                alpha_max = min(alpha_max, a_k)

            return max(min(alpha_max, 1.0), 0.0)

        F0 = F_orig(x)
        F0_norm = np.linalg.norm(F0)
        self.newton_monitor(0, x, F0, np.zeros_like(x))

        if F0_norm < tol_F:
            return x, True

        for k in range(1, max_iter + 1):
            J = self.approx_jacobian(
                F_orig,
                x,
                Fx=F0,
                r_min=self.r_min,
                r_max=self.r_max,
                z_min=self.z_min,
                z_max=self.z_max,
            )

            try:
                delta = np.linalg.solve(J, -F0)
            except np.linalg.LinAlgError:
                print(f"[Abort plain] LinAlgError at iter {k}, x={x}")
                return x, False

            alpha_max = compute_alpha_max(x, delta)
            if alpha_max <= 0:
                print(f"[Abort plain] alpha_max <= 0 at iter {k}, x={x}, delta={delta}")
                return x, False

            alpha_ls = alpha_max
            F_trial = None
            F_trial_norm = None

            for _ in range(ls_max_steps):
                x_trial = x + alpha_ls * delta

                if not (self.r_min <= x_trial[0] <= self.r_max and
                        self.z_min <= x_trial[1] <= self.z_max):
                    alpha_ls *= ls_reduction
                    continue

                F_trial = F_orig(x_trial)
                F_trial_norm = np.linalg.norm(F_trial)

                if F_trial_norm <= tol_F:
                    break

                if F_trial_norm <= F0_norm * (1.0 - ls_eta):
                    break

                alpha_ls *= ls_reduction

            if F_trial is None:
                print(f"[Abort plain] Linesearch stagnation at iter {k}, x={x}, |F|={F0_norm:.3e}")
                return x, False

            if F_trial_norm is None or F_trial_norm >= F0_norm:
                print(f"[Abort plain] No improvement at iter {k}, x={x}, |F|={F0_norm:.3e}")
                return x, False

            step = alpha_ls * delta
            x = x + step
            F0 = F_trial
            F0_norm = F_trial_norm

            self.newton_monitor(k, x, F0, step)

            if F0_norm < tol_F:
                return x, True

            if np.linalg.norm(step) < tol_x:
                print(f"[Stop plain] Step small at iter {k}, |dx|={np.linalg.norm(step):.3e}, |F|={F0_norm:.3e}")
                return x, (F0_norm < tol_F)

        print(f"[Stop plain] Max iters reached, |F|={F0_norm:.3e}")
        return x, (F0_norm < tol_F)

    def find_equilibria_with_deflation(self, initial_guesses, max_roots=20, skip_radius=0.02, newton_kwargs=None, verbose=True,
                                        coarse_data=None, max_candidates=None, boundary_tol=5e-3,):
        if newton_kwargs is None:
            newton_kwargs = {}

        if coarse_data is None:
            if verbose:
                print("=== Coarse grid (zero level sets) ===")

        candidates = initial_guesses.copy()

        candidates = np.asarray(candidates, dtype=float)
        if candidates.size == 0:
            if verbose:
                print("No zero level set candidates on coarse grid.")
            return np.empty((0, 2))

        if max_candidates is not None and len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        filtered_candidates = []
        for x0 in candidates:
            r0, z0 = x0
            if (
                abs(r0 - self.r_min) < boundary_tol
                or abs(r0 - self.r_max) < boundary_tol
                or abs(z0 - self.z_min) < boundary_tol
                or abs(z0 - self.z_max) < boundary_tol
            ):
                if verbose:
                    print(f"[Skip] x0={x0} too close to boundary.")
                continue
            if any(np.linalg.norm(x0 - y) < skip_radius for y in filtered_candidates):
                if verbose:
                    print(f"[Skip] x0={x0} too close to another candidate.")
                continue
            filtered_candidates.append(x0)

        if verbose:
            print("\n=== Start candidates for Newton (from zero level sets) ===")
            for x in filtered_candidates:
                print(f"  x0 = ({x[0]:.4f}, {x[1]:.4f})")

        roots = []
        if verbose:
            print("\n=== Newton + deflation ===")

        for x0 in filtered_candidates:
            if len(roots) >= max_roots:
                break

            if any(np.linalg.norm(x0 - r) < skip_radius for r in roots):
                if verbose:
                    print(f"[Skip] x0={x0} already covered by existing root.")
                continue

            if verbose:
                print(f"[OK] Starting Newton at x0={x0}")

            x_root, ok_newton = self.newton_plain_2d(
                x0,
                # known_roots=roots,
                # alpha=newton_kwargs.get("alpha", 1e-2),
                # p=newton_kwargs.get("p", 2.0),
                tol_F=newton_kwargs.get("tol_F", 1e-5),
                tol_x=newton_kwargs.get("tol_x", 1e-6),
                max_iter=newton_kwargs.get("max_iter", 20),
                ls_max_steps=newton_kwargs.get("ls_max_steps", 8),
                ls_reduction=newton_kwargs.get("ls_reduction", 0.5),
            )

            if not ok_newton:
                if verbose:
                    print(f"[Fail] Newton did not converge for x0={x0}")
                continue

            if any(np.linalg.norm(x_root - r) < skip_radius for r in roots):
                if verbose:
                    print(f"[Dup] x_root={x_root} is a duplicate.")
                continue

            roots.append(x_root)

            if verbose:
                Fvec = self.evaluate_F(x_root)
                print(f"  New equilibrium #{len(roots)}:")
                print(f"    r = {x_root[0]:.5f}, z = {x_root[1]:.5f}, |F| = {np.linalg.norm(Fvec):.3e}")

        return np.array(roots)


    def classify_single_equilibrium(fp_eval, x_eq, eps_rel=1e-4, ode_sign=-1.0, tol_eig=1e-6):

        x_eq = np.asarray(x_eq, dtype=float)

        def G(x):
            F = np.asarray(fp_eval.evaluate_F(x), dtype=float)
            return ode_sign * F

        Gx = G(x_eq)
        J = fp_eval.approx_jacobian(G, x_eq, Fx=Gx, eps_rel=eps_rel)

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


    def classify_equilibria(self, equilibria, ode_sign=-1.0, tol_eig=1e-6, verbose=True):

        equilibria = np.asarray(equilibria, dtype=float)
        if equilibria.ndim == 1:
            equilibria = equilibria[None, :]

        results = []

        for k, x_eq in enumerate(equilibria, start=1):
            info = self.classify_single_equilibrium(x_eq, ode_sign=ode_sign, tol_eig=tol_eig)

            # assign colors
            if info["type"] == "stable":
                info["color"] = "green"
            elif info["type"] == "unstable":
                info["color"] = "red"
            elif info["type"] == "saddle":
                info["color"] = "yellow"
            else:
                info["color"] = "gray"

            results.append(info)

            if verbose:
                ev = info["eigvals"]
                print(f"EQ #{k}: x_eq = ({x_eq[0]:.6f}, {x_eq[1]:.6f})")
                print(f"        eigenvalues(J_dyn): "
                      f"{ev[0].real:+.3e}{ev[0].imag:+.3e}i, "
                      f"{ev[1].real:+.3e}{ev[1].imag:+.3e}i")
                print(f"        Typ: {info['type']}\n")

        return results



