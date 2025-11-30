import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

from config_paper_parameters import *
from nondimensionalization import *
from find_equilbrium_points import *

if __name__ == "__main__":

    R_nd, H_nd, W_nd, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

    bg = background_flow(R, H, W, L_c, Re)
    bg.solve_2D_background_flow()
    bg.plot_2D_background_flow()

    R, H, W, a, u_bar, p_bar, Re_p = second_nondimensionalisation(R, H, W, a, L_c, U_c, Re, bg, print_values=True)

    force_grid = F_p_grid(a, Re_p, bg, particle_maxh=0.01, global_maxh=0.3*max(W,H), eps=0.5*a)
    r_vals, z_vals, phi, Fr_grid, Fz_grid = force_grid.compute_F_p_grid(N_r=10, N_z=10)
    force_grid.plot_paper_reproduction(r_vals, z_vals, phi, Fr_grid, Fz_grid)

    initial_guesses = force_grid.generate_initial_guesses()
    force_grid.plot_guesses_and_roots_on_grid()

    exit()
    F_p_roots = F_p_roots(R, W, H, L, a, particle_maxh, global_maxh, Re, Re_p)
    roots = F_p_roots.find_equilibria_with_deflation(initial_guesses)
    info = F_p_roots.classify_equilibria(roots)
    force_grid.plot_guesses_and_roots_on_grid(roots, stability_info=info)