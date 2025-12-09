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

        self.r_min = -W / 2 + 1.0 + eps
        self.r_max = W / 2 - 1.0 - eps
        self.z_min = -H / 2 + 1.0 + eps
        self.z_max = H / 2 - 1.0 - eps

        self.mesh_nx = 120
        self.mesh_ny = 120


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

            self.Fr_grid = Fr_grid.T
            self.Fz_grid = Fz_grid.T

            phi = np.sqrt(Fr_grid.T ** 2 + Fz_grid.T ** 2)
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


    def plot_paper_reproduction(self, L_c_p, L_c, initial_guesses=None):

        # np.meshgrid takes in 2 (or more 1d arrays, which define the coordinates along the axis)
        # It returns two matrices which have the x coordinates in every row and the y coordinates in every column
        # indexing="xy" (Cartesian Indexing) means that the first argument defines the numbers on the horizontal axis and the second on the vertical axis
        R_grid, Z_grid = np.meshgrid(self.r_vals, self.z_vals, indexing='xy')

        exclusion_dist = 1.0 + self.eps

        # This creates the "canvas" (Leinwand) for the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Since phi = np.sqrt(Fr_grid ** 2 + Fz_grid ** 2) this returns the force magnitude to colour the plot later
        # np.linespace produces num equdistant values between the first and second argument
        # The default would be to choose nice round numbers for the colour transition values, which leads to the max and min being shifted
        levels = np.linspace(np.nanmin(self.phi), np.nanmax(self.phi), 40)

        # ax.contourf takes in the R_grid and Z_grid (matrices from above) and plots the function phi on theese gridpoints levels
        # It also uses cartesian indexing
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
            ax.legend(loc='upper right', framealpha=1.0)

        margin = self.eps
        ax.set_xlim(-self.W / 2 - margin, self.W / 2 + margin)
        ax.set_ylim(-self.H / 2 - margin, self.H / 2 + margin)
        ax.set_aspect('equal')
        ax.set_xlabel("r")
        ax.set_ylabel("z")

        a = (L_c_p/L_c) * self.a * 2
        R = (L_c_p/L_c) * self.R * 2

        ax.set_title(f"Force_Map_a={a:.3f}_R={R:.1f}_with_H=W=2")
        plt.tight_layout()
        filename = f"Force_Map_a={a:.3f}_R={R:.1f}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()

