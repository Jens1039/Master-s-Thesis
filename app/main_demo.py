import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

from config_paper_parameters import *
from find_equilbrium_points import *





if __name__ == "__main__":

    Re_input = (D_h * U_c)

    bg = background_flow(R, H, W, Q, Re_input)
    bg.solve_2D_background_flow()

    Re_paper = Re_input * bg.Um
    Re_p_correct = Re_paper * (a / D_h) ** 2

    print(f"Re_input: {Re_input:.2f}")
    print(f"Re_paper: {Re_paper:.2f}")
    print(f"Re_p:     {Re_p_correct:.4f}")


    fp_eval = FpEvaluator(
            R, W, H, L, a,
            particle_maxh, global_maxh,
            Re_paper,
            Re_p_correct,
            bg_flow=bg
        )


    coarse_data = coarse_candidates_parallel_deflated(
            fp_eval,
            n_r=N_r,
            n_z=N_z,
            verbose=True,
            Re_bg_input=Re_input,
            nproc=nproc
        )

    candidates, r_vals, z_vals, phi, Fr_grid, Fz_grid = coarse_data

    plot_coarse_grid_with_zero_level_sets(fp_eval, r_vals, z_vals, phi, Fr_grid, Fz_grid)


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


