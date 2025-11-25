import warnings

from app.with_patricks_tipps import force_grid

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

    fp_eval = FpEvaluatorALE(R, W, H, L, a, particle_maxh, global_maxh, Re, Re_p)

    roots = find_equilibria_with_deflation(fp_eval, initial_guesses)

    force_grid.plot_guesses_and_roots_on_grid(roots)



    '''
    root_finder = Find_equilibria_with_deflated_newton(R, W, H, L, a, Re, Re_p, particle_maxh, global_maxh, eps,
                                                       bg_flow=bg, Q=Q)

    roots = root_finder.deflated_newton(initial_guesses, verbose=True)
    '''
