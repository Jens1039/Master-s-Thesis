import warnings
import sys

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

from config_paper_parameters import *
from nondimensionalization import *
from find_equilbrium_points import *


def auto_start_mpi(n_procs=10):

    is_mpi = "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ

    if not is_mpi:
        print(f"--- Auto-Starting with MPI (n={n_procs}) ---")

        cmd = ["mpiexec", "-n", str(n_procs), sys.executable, "-u"] + sys.argv

        print(f"Executing: {' '.join(cmd)}")
        print("-" * 40)

        try:
            os.execvp("mpiexec", cmd)
        except FileNotFoundError:
            print("Error: 'mpiexec' not found. Please install MPI or check your PATH.")
            sys.exit(1)


if __name__ == "__main__":

    auto_start_mpi()

    u_bar_2d_hat_hat_np = None
    p_bar_2d_hat_hat_np = None
    params = None
    L_c = None
    L_c_p = None

    if comm.rank == 0:
        print("Rank 0: Calculating 2D Background Flow...")

        R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=False)


        bg = background_flow(R_hat, H_hat, W_hat, Re, comm=MPI.COMM_SELF)
        u_bar_2d_hat, p_bar_2d_hat, G_hat, U_m_hat = bg.solve_2D_background_flow()
        # bg.plot_2D_background_flow()

        R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, L_c_p, U_c_p, u_bar_2d_hat_hat, p_bar_2d_hat_hat, G_hat_hat, Re_p \
            = second_nondimensionalisation(R_hat, H_hat, W_hat,  a, L_c, U_c, G_hat, Re, u_bar_2d_hat, p_bar_2d_hat, U_m_hat, print_values=False)

        u_bar_2d_hat_hat_np = u_bar_2d_hat_hat.dat.data_ro.copy()
        p_bar_2d_hat_hat_np = p_bar_2d_hat_hat.dat.data_ro.copy()

        params = (R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, Re_p)

    params = comm.bcast(params, root=0)
    u_data_np = comm.bcast(u_bar_2d_hat_hat_np, root=0)
    p_data_np = comm.bcast(p_bar_2d_hat_hat_np, root=0)

    (R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, Re_p) = params

    force_grid = F_p_grid(R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, Re_p, L=4*max(H_hat_hat, W_hat_hat),
                          particle_maxh=0.2*a_hat_hat, global_maxh=0.2*min(H_hat_hat, W_hat_hat), eps=0.2*a_hat_hat)

    grid_values = force_grid.compute_F_p_grid_ensemble(N_r=10, N_z=10, u_bg_data_np=u_data_np, p_bg_data_np=p_data_np)

    if comm.rank == 0:
        r_vals, z_vals, phi, Fr_grid, Fz_grid = grid_values
        force_grid.plot_paper_reproduction(L_c_p, L_c)