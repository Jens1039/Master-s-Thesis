import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os

os.environ["OMP_NUM_THREADS"] = "1"

from config_paper_parameters import *
from find_equilbrium_points import *

if __name__ == "__main__":
    bg = background_flow(R, H, W, Q, Re)
    bg.solve_2D_background_flow()

    Re_new = Re * (bg.U_m / U_c)
    Re_p = Re_new * ((a / D_h) ** 2)

    force_grid = F_p_grid(R, W, H, L, a, Re, Re_p, particle_maxh, global_maxh, eps, bg)
    r_vals, z_vals, phi, Fr_grid, Fz_grid = force_grid.compute_F_p_grid(N_r, N_z)
    force_grid.plot_paper_reproduction(r_vals, z_vals, phi, Fr_grid, Fz_grid)

    initial_guesses = force_grid.generate_initial_guesses()
    force_grid.plot_guesses_and_roots_on_grid()


    exit()
    F_p_roots = F_p_roots(R, W, H, L, a, particle_maxh, global_maxh, Re, Re_p)
    roots = F_p_roots.find_equilibria_with_deflation(initial_guesses)
    info = F_p_roots.classify_equilibria(roots)
    force_grid.plot_guesses_and_roots_on_grid(roots, stability_info=info)