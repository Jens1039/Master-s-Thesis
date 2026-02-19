from tqdm import tqdm
import matplotlib.patches as patches
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt

from InertialLiftCalculation_firedrake import *

import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import warnings
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()





class F_p_grid:

    def __init__(self, R, H, W, a, Re, L, eps):

        self.R = R
        self.H = H
        self.W = W
        self.Re = Re
        self.a = a
        self.L = L
        self.eps = eps

        self.r_min = -W / 2 + a + eps
        self.r_max = W / 2 - a - eps
        self.z_min = -H / 2 + a + eps
        self.z_max = H / 2 - a - eps

        self._current_mesh = None
        self._original_coords = None
        self._current_r = None
        self._current_z = None


    def compute_F_p_grid_ensemble(self, N_r, N_z, procs_per_ensemble=1):
        """
        Compute force grid using Firedrake's Ensemble parallelization.
        
        Parameters:
        -----------
        N_r : int
            Number of grid points in r direction
        N_z : int
            Number of grid points in z direction
        procs_per_ensemble : int
            Number of MPI processes to use per ensemble member (per grid point calculation).
            Default is 1 (each process computes different grid points sequentially).
            If > 1, each grid point calculation will be parallelized across procs_per_ensemble processes.
        """

        global_comm = COMM_WORLD
        global_rank = global_comm.rank
        global_size = global_comm.size
        
        # Create ensemble - each ensemble member uses procs_per_ensemble processes
        ensemble = Ensemble(global_comm, procs_per_ensemble)
        ensemble_comm = ensemble.ensemble_comm  # communicator between ensemble members
        spatial_comm = ensemble.comm            # communicator within each ensemble member
        
        ensemble_rank = ensemble_comm.Get_rank() if ensemble_comm != MPI.COMM_NULL else 0
        ensemble_size = ensemble_comm.Get_size() if ensemble_comm != MPI.COMM_NULL else 1
        
        spatial_rank = spatial_comm.Get_rank()
        spatial_size = spatial_comm.Get_size()

        r_vals = np.linspace(self.r_min, self.r_max, N_r)
        z_vals = np.linspace(self.z_min, self.z_max, N_z)

        # Create all tasks
        all_tasks = []
        for i in range(N_r):
            for j in range(N_z):
                all_tasks.append((i, j, r_vals[i], z_vals[j]))

        # Distribute tasks across ensemble members
        my_tasks = all_tasks[ensemble_rank::ensemble_size]
        local_results = []

        if global_rank == 0:
            print(f"Start ensemble grid: {len(all_tasks)} points")
            print(f"  Global MPI processes: {global_size}")
            print(f"  Ensemble members: {ensemble_size}")
            print(f"  Processes per member: {spatial_size}")
            print(f"  Tasks per member: ~{len(all_tasks)//ensemble_size}")

        # Each ensemble member computes its assigned tasks
        for task in tqdm(my_tasks, disable=(spatial_rank != 0)):
            (i, j, r_loc, z_loc) = task

            try:
                # Call InertialLiftCalculation with the spatial communicator
                # This allows parallel solves if procs_per_ensemble > 1
                F_r, F_z = InertialLiftCalculation(
                    R=self.R,
                    H=self.H,
                    W=self.W,
                    Re=self.Re,
                    DL=self.L,
                    px=r_loc,
                    pz=z_loc,
                    py=0,
                    pr=self.a,
                    comm=spatial_comm
                )

                local_results.append((i, j, F_r, F_z))

            except Exception as e:
                if spatial_rank == 0:
                    print(f"[Ensemble {ensemble_rank}] Error at {i},{j}: {e}")
                local_results.append((i, j, 0.0, 0.0))

        # Gather results from all ensemble members
        if ensemble_comm != MPI.COMM_NULL:
            all_data = ensemble_comm.gather(local_results, root=0)
        else:
            all_data = [local_results]

        # Process results on rank 0
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
        
        # Create Force_grids directory if it doesn't exist
        os.makedirs("images", exist_ok=True)
        
        filename = f"images/Force_Map_a={a:.3f}_R={R:.1f}.png"
        plt.savefig(filename)
        print(f"Plot saved to {filename}")
        plt.show()


def auto_start_mpi(n_procs=5):
    """
        Restart the script using mpiexec if not already running under MPI.
        Convenience function for local execution.
    """

    is_mpi = "OMPI_COMM_WORLD_RANK" in os.environ or \
             "PMI_RANK" in os.environ or \
             "SLURM_PROCID" in os.environ

    if not is_mpi:

        cmd = ["mpiexec", "-n", str(n_procs), sys.executable, "-u"] + sys.argv
        print(f"Executing: {' '.join(cmd)}\n")

        os.execvp("mpiexec", cmd)


if __name__ == "__main__":

    # Start with more processes for task parallelism
    # procs_per_ensemble=1 means each process computes different grid points
    # procs_per_ensemble=2 means grid points are computed in parallel using 2 processes each
    auto_start_mpi(n_procs=7)

    W = 2.0
    H = 2.0
    R = 160
    Re = 1.0
    DL = 8.0
    px = 0.0
    py = 0.0  # always 0.0
    pz = 0.25
    a = 0.05

    Grid = F_p_grid(R, H, W, a, Re, DL, 0.5*a)

    # Use procs_per_ensemble=1 for maximum task parallelism (default)
    # Each of the 5 processes will compute different grid points sequentially
    Grid.compute_F_p_grid_ensemble(7, 7, procs_per_ensemble=1)

    # Only rank 0 has the grid data, so only rank 0 should proceed with analysis
    if rank == 0:
        equilibria = Grid.generate_initial_guesses()

        Grid.classify_equilibria_on_grid(equilibria)

        Grid.plot(a, H)