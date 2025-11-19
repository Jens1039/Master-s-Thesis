import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
from matplotlib import pyplot as plt

from config_paper_parameters import *
from find_equilbrium_points import *


def newton_monitor(iter, x, Fx, delta):
    print(
        f"[Newton iter {iter:02d}] "
        f"x = ({x[0]:.5f}, {x[1]:.5f}) | "
        f"|F| = {np.linalg.norm(Fx):.3e} | "
        f"|dx| = {np.linalg.norm(delta):.3e}"
    )


if __name__ == "__main__":

    print(f"a == {a}")
    print("Re = ", Re)
    print("K = ", D_h / (2 * R) * (Re ** 2) / 4)
    print("De = ", De)
    print("kappa = ", kappa)

    bg = background_flow(R, H, W, Q, Re)
    bg.solve_2D_background_flow()
    Re_paper = Re * bg.Um

    print(f"Computed max axial velocity (dimensionless): Um = {bg.Um:.6f}")

    print(f"Your input Re (based on Q): Re_input = {Re:.6f}")


    print(f"Effective Re based on U_max (paper-style): Re_eff = {Re_paper:.6f}")

    De_input = Re * np.sqrt(D_h / (2 * R))
    De_paper = Re_paper * np.sqrt(D_h / (2 * R))

    print(f"Dean number with your Re: De_input = {De_input:.6f}")
    print(f"Dean number with U_max-based Re: De_eff = {De_paper:.6f}")

    fp_eval = FpEvaluator(
        R, W, H, L, a,
        particle_maxh, global_maxh,
        Re, Re_p
    )

    # coarse_data = coarse_candidates_parallel_deflated(fp_eval, n_r=N_r, n_z=N_z, verbose=True)

    coarse_data = np.load("app/coarse_data.npz")

    r_vals = coarse_data["r_vals"]
    z_vals = coarse_data["z_vals"]
    phi = coarse_data["phi"]
    Fr_grid = coarse_data["Fr_grid"]
    Fz_grid = coarse_data["Fz_grid"]





    def plot_coarse_grid_with_zero_level_sets(fp_eval,
                                              r_vals, z_vals,
                                              phi, Fr_grid, Fz_grid,
                                              title="Coarse grid with zero level sets",
                                              cmap="viridis", levels=20,
                                              figsize=(7, 5)):

        R, Z = np.meshgrid(r_vals, z_vals, indexing='ij')

        plt.figure(figsize=figsize)

        cs = plt.contourf(R, Z, phi, levels=levels, cmap=cmap)
        plt.colorbar(cs, label=r"$\|\mathbf{F}\|$")

        # Zero sets from stored grids
        plt.contour(R, Z, Fr_grid, levels=[0],
                    colors="cyan", linestyles="--", linewidths=2)

        plt.contour(R, Z, Fz_grid, levels=[0],
                    colors="magenta", linestyles="-", linewidths=2)

        plt.xlabel("r")
        plt.ylabel("z")
        plt.title(title)
        plt.tight_layout()
        plt.show()


    # candidates, r_vals, z_vals, phi, Fr_grid, Fz_grid = coarse_data

    plot_coarse_grid_with_zero_level_sets(fp_eval, r_vals, z_vals, phi, Fr_grid, Fz_grid)

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

    plot_equilibria_contour(fp_eval, r_vals, z_vals, phi, equilibria, stability_info=stability_info)
