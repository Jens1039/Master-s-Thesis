import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from matplotlib import pyplot as plt

from config_paper_parameters import *
from find_equilbrium_points import *


# --------------------------------------------------------
# Monitoring-Funktion (optional)
# --------------------------------------------------------
def newton_monitor(iter, x, Fx, delta):
    print(
        f"[Newton iter {iter:02d}] "
        f"x = ({x[0]:.5f}, {x[1]:.5f}) | "
        f"|F| = {np.linalg.norm(Fx):.3e} | "
        f"|dx| = {np.linalg.norm(delta):.3e}"
    )


# --------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------
if __name__ == "__main__":

    print("=== Setup FpEvaluator + Background Flow ===")

    fp_eval = FpEvaluator(
        R, W, H, L, a,
        particle_maxh, global_maxh,
        Re, Re_p
    )

    print("\n=== Coarse Grid (nur ein Run!) ===")

    # coarse grid EINMAL berechnen
    coarse_data = coarse_candidates_parallel_deflated(
        fp_eval,
        n_r=9,
        n_z=9,
        verbose=True
    )

    candidates, r_vals, z_vals, phi = coarse_data

    print("\n=== Newton + Deflation ===")

    equilibria = find_equilibria_with_deflation(
        fp_eval,
        n_r=9,
        n_z=9,
        max_roots=10,
        skip_radius=0.02,
        newton_kwargs=dict(
            alpha=1e-2,
            p=2.0,
            tol_F=1e-2,  # <-- WICHTIG: realistischer Wert
            tol_x=1e-6,
            max_iter=15,
            monitor=newton_monitor,
            ls_max_steps=8,
            ls_reduction=0.5,
        ),
        verbose=True,
        coarse_data=coarse_data,
        max_candidates=10,
        refine_factor=4,
    )

    print("\n=== Gefundene Gleichgewichtslagen ===")
    for i, (r_eq, z_eq) in enumerate(equilibria, start=1):
        print(f"{i}: r = {r_eq:.6f}, z = {z_eq:.6f}")

    if equilibria.size > 0:
        print("\n=== Stabilitätsklassifikation (x_dot = -F_p) ===")
        stability_info = classify_equilibria(
            fp_eval,
            equilibria,
            eps_rel=1e-4,
            ode_sign=-1.0,
            tol_eig=1e-6,
            verbose=True,
        )
    else:
        print("\nKeine Gleichgewichtslagen gefunden – nichts zu klassifizieren.")

    print("\n=== Visualisierung ===")
    plot_equilibria_contour(
        fp_eval,
        r_vals,
        z_vals,
        phi,
        equilibria,
        stability_info=stability_info
    )
