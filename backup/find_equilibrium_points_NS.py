from tqdm import tqdm
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
import pickle
from mpi4py import MPI
import gc

from nondimensionalization import *
from perturbed_flow_full_navier_stokes import *


class F_p_grid_NS:

    def __init__(self, R, H, W, a, Re, L, particle_maxh, global_maxh, eps):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.Re = Re
        self.L = L

        self.particle_maxh = particle_maxh
        self.global_maxh = global_maxh
        self.eps = eps

        self.r_min = -W / 2 + a + eps
        self.r_max = W / 2 - a - eps
        self.z_min = -H / 2 + a + eps
        self.z_max = H / 2 - a - eps


    def compute_F_p_grid_ensemble(self, N_grid):

        self.N_grid = N_grid

        ensemble = Ensemble(COMM_WORLD, 1)
        my_comm = ensemble.comm

        global_rank = COMM_WORLD.rank
        global_size = COMM_WORLD.size

        r_vals = np.linspace(self.r_min, self.r_max, N_grid)
        z_vals = np.linspace(self.z_min, self.z_max, N_grid)

        all_tasks = []
        for i in range(N_grid):
            for j in range(N_grid):
                all_tasks.append((i, j, r_vals[i], z_vals[j]))

        my_tasks = all_tasks[global_rank::global_size]
        local_results = []

        if global_rank == 0:
            print(f"Start NS ensemble grid: {len(all_tasks)} points on {global_size} cores.")

        for task in tqdm(my_tasks, disable=(global_rank != 0)):
            (i, j, r_loc, z_loc) = task

            mesh3d, tags = make_curved_channel_section_with_spherical_hole_periodic(
                    self.R, self.H, self.W, self.L, self.a,
                    self.particle_maxh, self.global_maxh,
                    r_off=r_loc, z_off=z_loc,
                    comm=my_comm)

            ns = FullNavierStokesSolver(self.R, self.H, self.W, self.L, self.a, self.Re, mesh3d, tags)
            ns.solve_flow()
            F_r, F_theta, F_z = ns.compute_particle_force()

            local_results.append((i, j, F_r, F_z))

            del ns, mesh3d, tags
            gc.collect()


        all_data = COMM_WORLD.gather(local_results, root=0)


        if global_rank == 0:
            Fr_grid = np.zeros((N_grid, N_grid))
            Fz_grid = np.zeros((N_grid, N_grid))

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


    def generate_initial_guesses(self, n_grid_search=100, tol_unique=1e-3, tol_residual=1e-14):

        self.interp_Fr = RectBivariateSpline(self.r_vals, self.z_vals, self.Fr_grid, kx=3, ky=3)
        self.interp_Fz = RectBivariateSpline(self.r_vals, self.z_vals, self.Fz_grid, kx=3, ky=3)

        def F(x):
            r, z = x
            return np.array([
                self.interp_Fr(r, z)[0, 0],
                self.interp_Fz(r, z)[0, 0]
            ], dtype=float)

        def J(x):
            r, z = x
            return np.array([
                [self.interp_Fr(r, z, dx=1, dy=0)[0, 0], self.interp_Fr(r, z, dx=0, dy=1)[0, 0]],
                [self.interp_Fz(r, z, dx=1, dy=0)[0, 0], self.interp_Fz(r, z, dx=0, dy=1)[0, 0]]
            ], dtype=float)

        def _interpolated_coarse_grid(x):
            r, z = x
            return [self.interp_Fr(r, z)[0, 0], self.interp_Fz(r, z)[0, 0]]

        r_starts = np.linspace(self.r_min, self.r_max, n_grid_search)
        z_starts = np.linspace(self.z_min, self.z_max, n_grid_search)

        initial_guesses = []

        for r0 in r_starts:
            for z0 in z_starts:

                solution = root(F, [r0, z0], jac=J, method="hybr")

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
            else:
                eq_type = "unclassified"
                color = "black"

            info = {
                "x_eq": x_eq,
                "type": eq_type,
                "color": color
            }

            classified_equilibria.append(info)

        return classified_equilibria


    def plot(self, L_c, initial_guesses=None, classified_equilibria=None):

        R_grid, Z_grid = np.meshgrid(self.r_vals, self.z_vals, indexing='ij')

        exclusion_dist = self.a + self.eps

        fig, ax = plt.subplots(figsize=(8, 6))

        levels = np.linspace(np.nanmin(self.phi), np.nanmax(self.phi), 40)

        cs = ax.contourf(R_grid, Z_grid, self.phi, levels=levels, cmap="viridis", alpha=1)

        cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(r"Force Magnitude $\|\mathbf{F}\|$")

        rf = np.linspace(self.r_min, self.r_max, 400)
        zf = np.linspace(self.z_min, self.z_max, 400)
        Fr_f = self.interp_Fr(rf, zf)
        Fz_f = self.interp_Fz(rf, zf)

        Rf, Zf = np.meshgrid(rf, zf, indexing="ij")
        ax.contour(Rf, Zf, Fr_f, levels=[0], colors="black", linewidths=2)
        ax.contour(Rf, Zf, Fz_f, levels=[0], colors="white", linewidths=2)

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

        a_display = self.a
        R_display = self.R
        H_display = self.H
        W_display = self.W

        ax.set_title(f"Force_Map_NS_a={a_display:.3f}_R={R_display:.0f}_W={W_display:.0f}_H={H_display:.0f}_Re{self.Re:.1f}_N_grid={self.N_grid}")
        plt.tight_layout()
        os.makedirs("../app/images", exist_ok=True)
        filename = f"images/Force_Map_NS_a={a_display:.3f}_R={R_display:.0f}_W={W_display:.0f}_H={H_display:.0f}_Re{self.Re:.1f}_N_grid={self.N_grid}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()


def force_grid(R, H, W, Q, rho, mu, a, N_grid, particle_maxh_rel, global_maxh_rel, eps_rel):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    cache_dir = "cache"
    Re = (rho*Q/(W*H)*(H/2))/mu
    cache_filename = f"{cache_dir}/force_grid_NS_R{R:.1f}_H{H:.1f}_W{W:.1f}_a{a:.3f}_Re{Re:.1f}_N{N_grid}.pkl"

    cache_exists = False
    if rank == 0:
        os.makedirs(cache_dir, exist_ok=True)
        if os.path.exists(cache_filename):
            cache_exists = True

    cache_exists = comm.bcast(cache_exists, root=0)

    # ==========================================
    # PATH A: cache gets read
    # ==========================================
    if cache_exists:
        if rank == 0:
            print(f"Load raw data from cache {cache_filename}...")
            with open(cache_filename, 'rb') as f:
                cached_data = pickle.load(f)

            params = cached_data['params']
            r_vals = cached_data['r_vals']
            z_vals = cached_data['z_vals']
            Fr_grid = cached_data['Fr_grid']
            Fz_grid = cached_data['Fz_grid']
            phi = cached_data['phi']

        else:
            return None

    # ==========================================
    # PATH B: data needs to be computed
    # ==========================================
    else:
        if rank == 0:
            print("Computing nondimensionalized parameters for NS solver...")
            R_hat, H_hat, W_hat, L_c, U_c, Re_calc = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

            a_hat = a / L_c

            params = {
                'R_hat': R_hat, 'H_hat': H_hat, 'W_hat': W_hat,
                'a_hat': a_hat, 'Re': Re_calc,
                'L_c': L_c,
                'L': 30 * max(H_hat, W_hat),
                'particle_maxh': particle_maxh_rel * a_hat,
                'global_maxh': global_maxh_rel * min(H_hat, W_hat),
                'eps': eps_rel * a_hat
            }
            print("Parameter computation done. Broadcasting data...")
        else:
            params = None

        params = comm.bcast(params, root=0)

        if rank == 0:
            print("Starting parallel NS force grid calculation...")

        f_grid_calc = F_p_grid_NS(
            params['R_hat'], params['H_hat'], params['W_hat'],
            params['a_hat'], params['Re'],
            params['L'], params['particle_maxh'], params['global_maxh'], params['eps']
        )

        grid_values = f_grid_calc.compute_F_p_grid_ensemble(N_grid=N_grid)

        if rank == 0:
            r_vals, z_vals, phi, Fr_grid, Fz_grid = grid_values

            data_to_save = {
                'params': params,
                'r_vals': r_vals,
                'z_vals': z_vals,
                'Fr_grid': Fr_grid,
                'Fz_grid': Fz_grid,
                'phi': phi
            }

            with open(cache_filename, 'wb') as f:
                pickle.dump(data_to_save, f)

            print(f"Parallel computation finished. Raw data saved to {cache_filename}")
        else:
            return None

    # ==========================================
    # POST-PROCESSING (Rank 0 only)
    # ==========================================
    if rank == 0:
        print("Running analysis (interpolation, equilibria, plot)...")

        f_grid = F_p_grid_NS(
            params['R_hat'], params['H_hat'], params['W_hat'],
            params['a_hat'], params['Re'],
            params['L'], params['particle_maxh'], params['global_maxh'], params['eps']
        )

        f_grid.N_grid = N_grid
        f_grid.r_vals = r_vals
        f_grid.z_vals = z_vals
        f_grid.Fr_grid = Fr_grid
        f_grid.Fz_grid = Fz_grid
        f_grid.phi = phi

        initial_guesses = f_grid.generate_initial_guesses()
        classified_equilibria = f_grid.classify_equilibria_on_grid(initial_guesses)

        f_grid.plot(params['L_c'],
                    classified_equilibria=classified_equilibria)

        R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

        print("Analysis complete. Returning requested objects.")

        return {
            "grid_points": {"R": R_grid, "Z": Z_grid},
            "F_grid": {"Fr": Fr_grid, "Fz": Fz_grid},
            "interpolators": {"interp_Fr": f_grid.interp_Fr, "interp_Fz": f_grid.interp_Fz},
            "equilibria": classified_equilibria
        }


def auto_start_mpi(n_procs=5):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
    os.environ["MPICC"] = "/opt/homebrew/bin/mpicc"
    os.environ["MPICXX"] = "/opt/homebrew/bin/mpicxx"
    os.environ["CC"] = "/opt/homebrew/bin/mpicc"
    os.environ["CXX"] = "/opt/homebrew/bin/mpicxx"

    is_mpi = "OMPI_COMM_WORLD_RANK" in os.environ or \
             "PMI_RANK" in os.environ or \
             "SLURM_PROCID" in os.environ

    if not is_mpi:

        cmd = ["mpiexec", "-n", str(n_procs), sys.executable, "-u"] + sys.argv
        print(f"Executing: {' '.join(cmd)}\n")

        os.execv("/opt/homebrew/bin/mpiexec", ["/opt/homebrew/bin/mpiexec"] + cmd[1:])


if __name__ == "__main__":

    auto_start_mpi()

    from config_paper_parameters import *

    force_grid(R, H, W, Q, rho, mu, a, N_grid, particle_maxh_rel, global_maxh_rel, eps_rel)
