import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

from config_paper_parameters import *
from nondimensionalization import *
from find_equilbrium_points import *


def auto_start_mpi(n_procs=4):
    """
    Startet das Skript neu mit mpiexec, falls es nicht bereits unter MPI läuft.
    """
    # 1. Prüfen, ob wir schon unter MPI laufen
    # OpenMPI setzt OMPI_COMM_WORLD_RANK, MPICH setzt PMI_RANK
    is_mpi = "OMPI_COMM_WORLD_RANK" in os.environ or "PMI_RANK" in os.environ

    # Optional: Man kann auch prüfen, ob mpi4py Größe > 1 meldet,
    # aber Umgebungsvariablen sind sicherer bevor mpi4py initialisiert wird.

    if not is_mpi:
        print(f"--- Auto-Starting with MPI (n={n_procs}) ---")

        # Der Befehl, der ausgeführt werden soll:
        # mpiexec -n 4 /pfad/zu/python /pfad/zu/deinem/skript.py
        cmd = ["mpiexec", "-n", str(n_procs), sys.executable, "-u"] + sys.argv

        # Debug Info
        print(f"Executing: {' '.join(cmd)}")
        print("-" * 40)

        # Ersetzt den aktuellen Prozess durch den mpiexec Prozess
        try:
            os.execvp("mpiexec", cmd)
        except FileNotFoundError:
            print("Error: 'mpiexec' not found. Please install MPI or check your PATH.")
            sys.exit(1)


if __name__ == "__main__":

    auto_start_mpi(n_procs=10)

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    u_data_np = None
    p_data_np = None
    params = None


    if comm.rank == 0:
        print("Rank 0: Calculating 2D Background Flow...")

        R, H, W, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

        bg = background_flow(R, H, W, L_c, Re, comm=MPI.COMM_SELF)

        u_bar_2d, p_bar_2d, U_m = bg.solve_2D_background_flow()
        bg.plot_2D_background_flow()

        R_s2, H_s2, W_s2, a_s2, U_c_p, u_bar_s2, p_bar_s2, Re_p = second_nondimensionalisation(R, H, W,  a, L_c, U_c, Re, u_bar_2d, p_bar_2d, U_m, print_values=True)

        u_data_np = u_bar_s2.copy()
        p_data_np = p_bar_s2.copy()

        params = (R_s2, H_s2, W_s2, a_s2, Re_p)

    if comm.rank == 0:
        print("Broadcasting data to workers...")

    params = comm.bcast(params, root=0)
    u_data_np = comm.bcast(u_data_np, root=0)
    p_data_np = comm.bcast(p_data_np, root=0)

    (R, H, W, a, Re_p) = params

    force_grid = F_p_grid(R, H, W, a, Re_p, L=4*max(H, W), particle_maxh=0.3*a, global_maxh=0.3*min(H, W), eps=0.2*a)

    ret = force_grid.compute_F_p_grid_ensemble(
        N_r=10, N_z=10,
        u_bg_data_np=u_data_np,
        p_bg_data_np=p_data_np
    )

    if comm.rank == 0:
        r_vals, z_vals, phi, Fr_grid, Fz_grid = ret
        force_grid.plot_paper_reproduction(r_vals, z_vals, phi, Fr_grid, Fz_grid)