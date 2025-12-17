import os
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import warnings
from mpi4py import MPI

from config_paper_parameters import *
from nondimensionalization import *
from find_equilbrium_points import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()


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

    auto_start_mpi()

    params = None
    u_bar_2d_hat_hat_np = None
    p_bar_2d_hat_hat_np = None

    if rank == 0:
        print("Calculating 2D background flow...")

        R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu)

        bg = background_flow(R_hat, H_hat, W_hat, Re, comm=MPI.COMM_SELF)
        G_hat, U_m_hat, u_bar_2d_hat, p_bar_2d_hat = bg.solve_2D_background_flow()
        bg.plot()

        R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, L_c_p, U_c_p, u_bar_2d_hat_hat, p_bar_2d_hat_hat, Re_p \
            = second_nondimensionalisation(R_hat, H_hat, W_hat, a, L_c, U_c, G_hat, Re, u_bar_2d_hat, p_bar_2d_hat, U_m_hat)

        # Extract raw arrays for MPI broadcast (avoids pickling issues with complex FEM objects)
        u_bar_2d_hat_hat_np = u_bar_2d_hat_hat.dat.data_ro.copy()
        p_bar_2d_hat_hat_np = p_bar_2d_hat_hat.dat.data_ro.copy()

        params = (R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, L_c, L_c_p, Re_p)

        print("Background flow calculation done. Broadcasting data...")

    params = comm.bcast(params, root=0)
    u_data_np = comm.bcast(u_bar_2d_hat_hat_np, root=0)
    p_data_np = comm.bcast(p_bar_2d_hat_hat_np, root=0)

    (R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, L_c, L_c_p, Re_p) = params

    if rank == 0:
        print("Starting parallel force grid calculation...")

    force_grid = F_p_grid(R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, Re_p,
                          L=4*max(H_hat_hat, W_hat_hat),
                          particle_maxh=0.2*a_hat_hat,
                          global_maxh=0.2*min(H_hat_hat, W_hat_hat),
                          eps=0.2*a_hat_hat)

    grid_values = force_grid.compute_F_p_grid_ensemble(N_r=10, N_z=10, u_bg_data_np=u_data_np, p_bg_data_np=p_data_np)

    if rank == 0:
        print("Finished parallel force grid calculation. Finding and classifying equilibria and visualizing...")
        r_vals, z_vals, phi, Fr_grid, Fz_grid = grid_values
        initial_guesses = force_grid.generate_initial_guesses()
        classified_equilibria = force_grid.classify_equilibria_on_grid(initial_guesses)
        force_grid.plot(L_c_p, L_c, classified_equilibria=classified_equilibria)