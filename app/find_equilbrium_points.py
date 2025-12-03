from tqdm import tqdm
import os
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
from firedrake import *


from background_flow import *
from build_3d_geometry import *
from perturbed_flow import *




class F_p_grid:

    def __init__(self, R, H, W, a, Re_p, L, particle_maxh, global_maxh, eps):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.Re_p = Re_p
        self.L = L

        self.particle_maxh = particle_maxh
        self.global_maxh = global_maxh
        self.eps = eps

        self.r_min = -W / 2 + 1.0 + eps
        self.r_max = W / 2 - 1.0 - eps
        self.z_min = -H / 2 + 1.0 + eps
        self.z_max = H / 2 - 1.0 - eps

        self.mesh_nx = 120
        self.mesh_ny = 120


    def compute_F_p_grid_ensemble(self, N_r, N_z, u_bg_data_np, p_bg_data_np):
        from mpi4py import MPI

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
            print(f"Start Ensemble Grid: {len(all_tasks)} points on {global_size} cores.")

        for task in tqdm(my_tasks, disable=(global_rank != 0)):
            (i, j, r_loc, z_loc) = task

            try:
                mesh3d, tags = make_curved_channel_section_with_spherical_hole(
                    self.R, self.H, self.W, self.L, self.a,
                    self.particle_maxh, self.global_maxh,
                    r_off=r_loc, z_off=z_loc,
                    comm=my_comm
                )

                u_3d, p_3d = build_3d_background_flow(self.R, self.H, self.W, mesh3d, u_bg_local, p_bg_local)

                pf = perturbed_flow(self.R, self.H, self.W, self.a, self.Re_p, mesh3d, tags, u_3d, p_3d)
                F_vec = pf.F_p()

                cx, cy, cz = tags["particle_center"]
                r0 = float(np.hypot(cx, cy))
                ex0 = np.array([1., 0., 0.]) if r0 < 1e-14 else np.array([cx / r0, cy / r0, 0.])
                ez0 = np.array([0., 0., 1.])

                Fr = float(np.dot(ex0, F_vec))
                Fz = float(np.dot(ez0, F_vec))

                local_results.append((i, j, Fr, Fz))

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

            phi = np.sqrt(Fr_grid ** 2 + Fz_grid ** 2)
            self.r_vals = r_vals
            self.z_vals = z_vals
            self.phi = phi
            self.Fr_grid = Fr_grid
            self.Fz_grid = Fz_grid

            return r_vals, z_vals, phi, Fr_grid, Fz_grid
        else:
            return None, None, None, None, None


    def plot_paper_reproduction(self, L_c_p, L_c, r_vals, z_vals, phi, Fr_grid, Fz_grid):

        R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

        exclusion_dist = 1.0 + self.eps

        fig, ax = plt.subplots(figsize=(8, 6))

        levels = np.linspace(np.nanmin(phi), np.nanmax(phi), 40)
        cs = ax.contourf(R_grid, Z_grid, phi, levels=levels, cmap="viridis", alpha=0.9)
        cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Force Magnitude $\|\mathbf{F}\|$")

        ax.contour(R_grid, Z_grid, Fr_grid, levels=[0], colors="cyan", linestyles="--", linewidths=2)
        ax.contour(R_grid, Z_grid, Fz_grid, levels=[0], colors="magenta", linestyles="-", linewidths=2)

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

        margin = self.eps
        ax.set_xlim(-self.W / 2 - margin, self.W / 2 + margin)
        ax.set_ylim(-self.H / 2 - margin, self.H / 2 + margin)
        ax.set_aspect('equal')
        ax.set_xlabel("r (local)")
        ax.set_ylabel("z (local)")

        a = (L_c_p/L_c) * self.a
        R = (L_c_p/L_c) * self.R

        ax.set_title(f"Force_Map_a={a:.3f}_R={R:.1f}")
        plt.tight_layout()
        filename = f"cache/Force_Map_a={a:.3f}_R={R:.1f}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()


'''
class F_p_grid:

    def __init__(self, R, H, W, a, Re_p, u_bar_2d, L, particle_maxh, global_maxh, eps):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.Re_p = Re_p
        self.L = L

        self.particle_maxh = particle_maxh
        self.global_maxh = global_maxh
        self.eps = eps

        self.r_min = -W/2 + 1 + eps
        self.r_max = W/2 - 1 - eps
        self.z_min = -H/2 + 1 + eps
        self.z_max = H/2 - 1 - eps

    # Global variables for the worker processes
    _U_2D_GLOBAL = None
    _P_2D_GLOBAL = None
    _MESH_PARAMS = None


    # Called once per process, to reconstruct the firedrake function from the numpy-data
    @staticmethod
    def _init_bg_worker(u_data_np, p_data_np, H, W):

        global _U_2D_GLOBAL, _P_2D_GLOBAL, _MESH_PARAMS

        mesh2d = RectangleMesh(120, 120, W, H, quadrilateral=False)

        V = VectorFunctionSpace(mesh2d, "CG", 2, dim=3)
        Q = FunctionSpace(mesh2d, "CG", 1)

        _U_2D_GLOBAL = Function(V)
        _P_2D_GLOBAL = Function(Q)

        # Copy data (Firedrake order is deterministic for similar mesh construction)
        _U_2D_GLOBAL.dat.data[:] = u_data_np
        _P_2D_GLOBAL.dat.data[:] = p_data_np
        _MESH_PARAMS = (H, W)

    # Computes the force F_p for one single point
    @staticmethod
    def _compute_single_F_p(task):

        global _U_2D_GLOBAL, _P_2D_GLOBAL, _MESH_PARAMS
        R_bg, W_bg, H_bg, Re_bg = _MESH_PARAMS

        (R, H, W, a, Re_p, i, j, r_loc, z_loc,
         L, a, particle_maxh, global_maxh, Re_p) = task

        mesh3d, tags = make_curved_channel_section_with_spherical_hole(
            R_bg, W_bg, H_bg, L, a, particle_maxh, global_maxh,
            r_off=r_loc, z_off=z_loc
        )

        u_3d, p_3d = build_3d_background_flow(mesh3d, _U_2D_GLOBAL, _P_2D_GLOBAL, R_bg, W_bg, H_bg)

        bg_flow_dummy = MockBackgroundObject(u_3d, p_3d, Re_bg)

        pf = perturbed_flow(R, H, W, a, Re_p, mesh3d, tags, u_bar, p_bar)
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


    def plot_paper_reproduction(self, r_vals, z_vals, phi, Fr_grid, Fz_grid, invert_xaxis=False):

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
        plt.savefig(f"a_{a}_R_{R}.png")







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


    def plot_guesses_and_roots_on_grid(self, roots=None, stability_info=None, invert_xaxis=False):

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
        plt.savefig(f"a_{a}_R_{R}_with_roots.png")
'''
'''
class F_p_grid:

    def __init__(self, R, H, W, a, Re_p, u_bar_2d, p_bar_2d, L, particle_maxh, global_maxh, eps):

        # Geometrie- und Physikparameter (bereits entdimensionalisiert nach Skalierung 2)
        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.Re_p = Re_p
        self.L = L

        self.particle_maxh = particle_maxh
        self.global_maxh = global_maxh
        self.eps = eps

        self.r_min = -W / 2 + 1.0 + eps
        self.r_max = W / 2 - 1.0 - eps
        self.z_min = -H / 2 + 1.0 + eps
        self.z_max = H / 2 - 1.0 - eps

        self.u_data_np = u_bar_2d
        self.p_data_np = p_bar_2d

        self.mesh_params = (120, 120, W, H)


    _U_2D_GLOBAL = None
    _P_2D_GLOBAL = None
    _MESH_PARAMS_GLOBAL = None

    @staticmethod
    def _init_bg_worker(u_data_np, p_data_np, mesh_dims):
        global _U_2D_GLOBAL, _P_2D_GLOBAL, _MESH_PARAMS_GLOBAL

        nx, ny, W, H = mesh_dims

        # Rekonstruktion des 2D Meshes (muss exakt übereinstimmen mit background_flow)
        mesh2d = RectangleMesh(nx, ny, W, H, quadrilateral=False)

        V = VectorFunctionSpace(mesh2d, "CG", 2, dim=3)
        Q = FunctionSpace(mesh2d, "CG", 1)

        _U_2D_GLOBAL = Function(V)
        _P_2D_GLOBAL = Function(Q)

        # Daten kopieren
        _U_2D_GLOBAL.dat.data[:] = u_data_np
        _P_2D_GLOBAL.dat.data[:] = p_data_np

        # Speichern von H und W für spätere Nutzung im Worker
        _MESH_PARAMS_GLOBAL = (W, H)

    @staticmethod
    def _compute_single_F_p(task):
        global _U_2D_GLOBAL, _P_2D_GLOBAL, _MESH_PARAMS_GLOBAL

        # Entpacken der Task-Parameter
        (i, j, r_loc, z_loc,
         R, H, W, L, a, Re_p,
         particle_maxh, global_maxh) = task

        # Parameter des globalen Meshes
        W_bg, H_bg = _MESH_PARAMS_GLOBAL

        # ACHTUNG: R ist hier der Radius der Kanalmitte aus Skalierung 2.
        # r_loc ist die radiale Verschiebung von der Kanalmitte.

        # 1. Erstelle das lokale 3D Mesh um das Partikel
        try:
            mesh3d, tags = make_curved_channel_section_with_spherical_hole(
                R, H, W, L, a, particle_maxh, global_maxh,
                r_off=r_loc, z_off=z_loc
            )
        except Exception as e:
            print(f"Mesh generation failed at {r_loc}, {z_loc}: {e}")
            return (i, j, np.nan, np.nan)

        # 2. Interpoliere den Hintergrundfluss auf das 3D Mesh
        # Hier nutzen wir die globalen 2D Funktionen und projizieren sie auf das gekrümmte Segment
        u_3d, p_3d = build_3d_background_flow(R, H, W, mesh3d, _U_2D_GLOBAL, _P_2D_GLOBAL)

        # 3. Löse das gestörte Flussproblem (Perturbed Flow)
        # Die Klasse perturbed_flow erwartet u_bar und p_bar als Funktionen auf mesh3d
        pf = perturbed_flow(R, H, W, a, Re_p, mesh3d, tags, u_3d, p_3d)

        # 4. Berechne die Kraft
        # F_vec ist ein Numpy Array [Fx, Fy, Fz] (in lokalen kartesischen Koordinaten des Partikels)
        F_vec = pf.F_p()

        # Projektion auf lokales Koordinatensystem (r, z)
        # r-Richtung: Vektor vom Krümmungsmittelpunkt zum Partikelzentrum (in xy Ebene)
        cx, cy, cz = tags["particle_center"]
        r0 = float(np.hypot(cx, cy))

        if r0 < 1e-14:
            ex0 = np.array([1., 0., 0.])
        else:
            ex0 = np.array([cx / r0, cy / r0, 0.])

        ez0 = np.array([0., 0., 1.])

        Fr = float(np.dot(ex0, F_vec))
        Fz = float(np.dot(ez0, F_vec))

        return (i, j, Fr, Fz)

    def compute_F_p_grid(self, N_r, N_z, nproc=mp.cpu_count()):

        r_vals = np.linspace(self.r_min, self.r_max, N_r)
        z_vals = np.linspace(self.z_min, self.z_max, N_z)

        Fr_grid = np.zeros((N_r, N_z))
        Fz_grid = np.zeros((N_r, N_z))

        # Erstellen der Aufgabenliste
        tasks = [
            (i, j, r_vals[i], z_vals[j],
             self.R, self.H, self.W, self.L, self.a, self.Re_p,
             self.particle_maxh, self.global_maxh)
            for i in range(N_r) for j in range(N_z)
        ]

        if nproc == 1:
            print(f"DEBUG MODE: Running sequentially to find errors...")
            self._init_bg_worker(self.u_data_np, self.p_data_np, self.mesh_params)

            results = []
            for task in tqdm(tasks):
                res = self._compute_single_F_p(task)
                results.append(res)

        print(f"Start parallel grid computation with {nproc} processes.")

        with mp.Pool(
                processes=nproc,
                initializer=self._init_bg_worker,
                initargs=(self.u_data_np, self.p_data_np, self.mesh_params)
        ) as pool:

            results = []
            # imap_unordered ist oft speichereffizienter und schneller bei ungleichmäßiger Last
            for res in tqdm(pool.imap_unordered(self._compute_single_F_p, tasks), total=len(tasks)):
                results.append(res)

        # Ergebnisse einsortieren
        for (i, j, Fr, Fz) in results:
            Fr_grid[i, j] = Fr
            Fz_grid[i, j] = Fz

        # Kraftbetrag für Konturplot
        phi = np.sqrt(Fr_grid ** 2 + Fz_grid ** 2)

        # Daten speichern
        self.r_vals = r_vals
        self.z_vals = z_vals
        self.phi = phi
        self.Fr_grid = Fr_grid
        self.Fz_grid = Fz_grid

        return r_vals, z_vals, phi, Fr_grid, Fz_grid

    def plot_paper_reproduction(self, r_vals, z_vals, phi, Fr_grid, Fz_grid, invert_xaxis=False):

        # Meshgrid für Matplotlib (indexing='ij' passt zu unserer Matrixstruktur [i, j])
        R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

        # Bereich, in dem das Partikel die Wand berühren würde (physikalisch verboten)
        # Da wir in Skalierung 2 sind, ist der Partikelradius = 1.0
        # self.a ist im Konstruktor bereits der skalierte Wert (also 1.0),
        # aber wir nutzen hier sicherheitshalber 1.0 + eps
        exclusion_dist = 1.0 + self.eps

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot Magnitude
        levels = np.linspace(np.nanmin(phi), np.nanmax(phi), 40)
        cs = ax.contourf(R_grid, Z_grid, phi, levels=levels, cmap="viridis", alpha=0.9)
        cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Force Magnitude $\|\mathbf{F}\|$")

        # Null-Isolinien (Gleichgewichtspunkte sind Schnittpunkte)
        # Fr = 0 (cyan, gestrichelt)
        ax.contour(R_grid, Z_grid, Fr_grid, levels=[0], colors="cyan", linestyles="--", linewidths=2)
        # Fz = 0 (magenta, durchgezogen)
        ax.contour(R_grid, Z_grid, Fz_grid, levels=[0], colors="magenta", linestyles="-", linewidths=2)

        # Wände zeichnen
        wall_rect = patches.Rectangle((-self.W / 2, -self.H / 2), self.W, self.H,
                                      linewidth=3, edgecolor='black', facecolor='none', zorder=10)
        ax.add_patch(wall_rect)

        # Verbotene Zonen schraffieren
        # Links
        ax.add_patch(patches.Rectangle((-self.W / 2, -self.H / 2), exclusion_dist, self.H,
                                       facecolor='gray', alpha=0.3, hatch='///'))
        # Rechts
        ax.add_patch(patches.Rectangle((self.W / 2 - exclusion_dist, -self.H / 2), exclusion_dist, self.H,
                                       facecolor='gray', alpha=0.3, hatch='///'))
        # Unten
        ax.add_patch(patches.Rectangle((-self.W / 2, -self.H / 2), self.W, exclusion_dist,
                                       facecolor='gray', alpha=0.3, hatch='///'))
        # Oben
        ax.add_patch(patches.Rectangle((-self.W / 2, self.H / 2 - exclusion_dist), self.W, exclusion_dist,
                                       facecolor='gray', alpha=0.3, hatch='///'))

        margin = 0.5
        ax.set_xlim(-self.W / 2 - margin, self.W / 2 + margin)
        ax.set_ylim(-self.H / 2 - margin, self.H / 2 + margin)
        ax.set_aspect('equal')
        ax.set_xlabel("r (local)")
        ax.set_ylabel("z (local)")
        ax.set_title(f"Force Map (Re_p={self.Re_p:.2f}, R={self.R:.1f})")

        if invert_xaxis:
            ax.invert_xaxis()  # Innenwand (Links im Paper) ist oft links, aber manchmal wird r invertiert

        plt.tight_layout()
        filename = f"ForceMap_R_{self.R:.1f}_Rep_{self.Re_p:.2f}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()

    def generate_initial_guesses(self, n_grid_search=50, tol_unique=1e-3, tol_residual=1e-5):
        # Überprüfen ob Grid berechnet wurde
        if not hasattr(self, 'Fr_grid'):
            print("Error: Compute Grid first.")
            return []

        self.interp_Fr = RectBivariateSpline(self.r_vals, self.z_vals, self.Fr_grid)
        self.interp_Fz = RectBivariateSpline(self.r_vals, self.z_vals, self.Fz_grid)

        def _interpolated_force(x):
            r, z = x
            # RectBivariateSpline gibt arrays zurück, wir brauchen scalars
            return [self.interp_Fr(r, z)[0, 0], self.interp_Fz(r, z)[0, 0]]

        r_starts = np.linspace(self.r_min, self.r_max, n_grid_search)
        z_starts = np.linspace(self.z_min, self.z_max, n_grid_search)

        initial_guesses = []

        for r0 in r_starts:
            for z0 in z_starts:

                # Hybrid-Solver sucht Nullstelle der interpolierten Kraftfelder
                solution = root(_interpolated_force, [r0, z0], method='hybr')

                if solution.success:
                    r_sol, z_sol = solution.x

                    # Check ob innerhalb der Domain
                    if not (self.r_min <= r_sol <= self.r_max and self.z_min <= z_sol <= self.z_max):
                        continue

                    # Check Residuum (manchmal konvergiert hybr fälschlicherweise)
                    if np.linalg.norm(_interpolated_force((r_sol, z_sol))) > tol_residual:
                        continue

                    # Duplikat-Check
                    is_new = True
                    for existing in initial_guesses:
                        dist = np.linalg.norm(np.array([r_sol, z_sol]) - existing)
                        if dist < tol_unique:
                            is_new = False
                            break

                    if is_new:
                        initial_guesses.append(np.array([r_sol, z_sol]))

        self.initial_guesses = initial_guesses
        print(f"Found {len(initial_guesses)} candidate equilibrium points.")
        return initial_guesses





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

        x = SpatialCoordinate(mesh)
        cx, cy, cz = tags["particle_center"]
        r_dist = sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2 + (x[2] - cz) ** 2)

        stiff = 1.0 / (r_dist ** 2 + 0.01)

        mu = Constant(1.0) * stiff
        lmbda = Constant(1.0) * stiff

        def epsilon(u):
            return 0.5 * (grad(u) + grad(u).T)

        def sigma(u):
            return lmbda * div(u) * Identity(3) + 2 * mu * epsilon(u)

        a = inner(sigma(u), epsilon(v)) * dx
        L = inner(Constant((0, 0, 0)), v) * dx

        bc_walls = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["walls"])
        bc_in = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["inlet"])
        bc_out = DirichletBC(V_disp, Constant((0., 0., 0.)), tags["outlet"])

        disp_const = Constant(displacement_vector)
        bc_part = DirichletBC(V_disp, disp_const, tags["particle"])

        bcs = [bc_walls, bc_in, bc_out, bc_part]

        displacement_sol = Function(V_disp)
        solve(a == L, displacement_sol, bcs=bcs,
              solver_parameters={'ksp_type': 'preonly', 'pc_type': 'lu', "pc_factor_mat_solver_type": "mumps"})

        V_coords = mesh.coordinates.function_space()

        displacement_high_order = Function(V_coords)

        displacement_high_order.interpolate(displacement_sol)

        mesh.coordinates.assign(mesh.coordinates + displacement_high_order)


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

        x = np.asarray(x0, dtype=float)

        def F_orig(x_vec):
            return np.asarray(self.evaluate_F(x_vec), dtype=float)

        roots = [np.asarray(rz, dtype=float) for rz in known_roots]

        def defl_factor_local(x_vec):

            if not roots:
                return 1.0
            fac = 1.0
            for rstar in roots:
                dist = np.linalg.norm(x_vec - rstar)

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

        F0 = F_orig(x)
        F0_norm = np.linalg.norm(F0)
        F0_defl = defl_factor_local(x) * F0

        self.newton_monitor(0, x, F0, np.zeros_like(x))

        if F0_norm < tol_F:
            return x, True

        for k in range(1, max_iter + 1):
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
                print(f"[Abort defl] Linesearch stagnation (no valid step) at iter {k}, x={x}, |F|={F0_norm:.3e}")
                return x, False

            if F_trial_norm is None or F_trial_norm >= F0_norm:
                print(f"[Abort defl] No improvement at iter {k}, x={x}, |F|={F0_norm:.3e}")
                return x, False

            step = alpha_ls * delta
            x = x + step

            F0 = F_trial
            F0_norm = F_trial_norm
            F0_defl = defl_factor_local(x) * F0

            self.newton_monitor(k, x, F0, step)

            if F0_norm < tol_F:
                return x, True

            if np.linalg.norm(step) < tol_x:
                print(f"[Stop defl] Step small at iter {k}, |dx|={np.linalg.norm(step):.3e}, "
                      f"|F|={F0_norm:.3e}, rejecting")
                return x, False

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
'''