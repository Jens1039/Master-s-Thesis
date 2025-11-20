import warnings
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

    Re_new = Re * (bg.U_m/U_c)

    Re_p = Re_new*((a/H)**2)

    fp_eval = FpEvaluator(R, W, H, L, a, particle_maxh, global_maxh, Re_new, Re_p, bg_flow=bg)

    coarse_data = coarse_candidates_parallel_deflated(fp_eval, n_r=N_r, n_z=N_z, verbose=True, Re_bg_input=Re, nproc=nproc)

    candidates, r_vals, z_vals, phi, Fr_grid, Fz_grid = coarse_data

    plot_paper_reproduction(fp_eval, r_vals, z_vals, phi, Fr_grid, Fz_grid)
    exit()

    equilibria = find_equilibria_with_deflation(fp_eval,
        n_r=10,
        n_z=10,
        max_roots=10,
        skip_radius=0.02,
        newton_kwargs=dict(
            alpha=1e-2,
            p=2.0,
            tol_F=2e-2,
            tol_x=1e-6,
            max_iter=30,
            monitor=newton_monitor,
            ls_max_steps=8,
            ls_reduction=0.5,
        ),
        verbose=True,
        coarse_data=coarse_data,
        max_candidates=10,
        refine_factor=6,
    )

    print("\n=== Found equilibria ===")
    for i, (r_eq, z_eq) in enumerate(equilibria, start=1):
        print(f"{i}: r = {r_eq:.6f}, z = {z_eq:.6f}")

    stability_info = []

    if equilibria.size > 0:
        print("\n=== Equilibria clasification (x_dot = F_p) ===")
        stability_info = classify_equilibria(
            fp_eval,
            equilibria,
            eps_rel=1e-4,
            ode_sign=-1.0,
            tol_eig=1e-6,
            verbose=True,
        )
    else:
        print("\nNo equilibria found, nothing to classify")


