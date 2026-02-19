import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from config_paper_parameters import *
from nondimensionalization import *
from find_equilbrium_points import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


ALPHA_VALUES = np.round(np.arange(0.01, 0.151, 0.01), 2)

RESULTS_FILE = "bifurcation_results.json"


def auto_start_mpi(n_procs=5):

    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")
    os.environ["MPICC"]  = "/opt/homebrew/bin/mpicc"
    os.environ["MPICXX"] = "/opt/homebrew/bin/mpicxx"
    os.environ["CC"]     = "/opt/homebrew/bin/mpicc"
    os.environ["CXX"]    = "/opt/homebrew/bin/mpicxx"

    is_mpi = (
        "OMPI_COMM_WORLD_RANK" in os.environ
        or "PMI_RANK" in os.environ
        or "SLURM_PROCID" in os.environ
    )
    if not is_mpi:
        cmd = ["mpiexec", "-n", str(n_procs), sys.executable, "-u"] + sys.argv
        print(f"Executing: {' '.join(cmd)}\n")
        os.execv("/opt/homebrew/bin/mpiexec", ["/opt/homebrew/bin/mpiexec"] + cmd[1:])


def plot_bifurcation_diagram(data, output_file="Force_grids/bifurcation_diagram.png"):
    """
    Bifurcation diagram:
      x-axis: relative particle size  a_hat = a / (H/2)
      y-axis: equilibrium r-position  r_norm = r / (H/2)  (singly nondimensional)
    r_norm lies in [-W_hat/2, W_hat/2]; for W = H this is [-1, 1].
    """
    type_styles = {
        "stable":       {"marker": "o", "color": "green",  "label": "Stable equilibrium"},
        "unstable":     {"marker": "o", "color": "red",    "label": "Unstable equilibrium"},
        "saddle":       {"marker": "X", "color": "orange", "label": "Saddle point"},
        "unclassified": {"marker": "s", "color": "gray",   "label": "Unclassified"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    plotted_types = set()
    for entry in data:
        a_hat  = entry["a_hat"]
        r_norm = entry["r_norm"]
        eq_type = entry["type"]

        style = type_styles.get(eq_type, type_styles["unclassified"])
        label = style["label"] if eq_type not in plotted_types else None
        plotted_types.add(eq_type)

        ax.scatter(
            a_hat, r_norm,
            color=style["color"], marker=style["marker"],
            s=90, edgecolors="black", linewidths=0.8,
            label=label, zorder=5,
        )

    W_hat = W / (H / 2)
    ax.axhline(y= W_hat / 2, color="black", linestyle="--", linewidth=1.5, label="Channel wall (±W/2)")
    ax.axhline(y=-W_hat / 2, color="black", linestyle="--", linewidth=1.5)
    ax.axhline(y=0,           color="gray",  linestyle=":",  linewidth=1.0, alpha=0.5)

    ax.set_xlabel(r"Relative particle size $\alpha = a\,/\,(H/2)$", fontsize=13)
    ax.set_ylabel(r"Equilibrium position $r\,/\,(H/2)$",            fontsize=13)
    ax.set_title("Bifurcation Diagram — equilibrium r-positions vs. particle size", fontsize=14)

    ax.set_xlim(ALPHA_VALUES[0] - 0.005, ALPHA_VALUES[-1] + 0.005)
    ax.set_xticks(ALPHA_VALUES)
    ax.legend(loc="upper right", framealpha=1.0)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=150)
    print(f"Bifurcation diagram saved to {output_file}")
    plt.show()


if __name__ == "__main__":

    auto_start_mpi()

    # ------------------------------------------------------------------ #
    #  Phase 0: Load cached data if existent                             #
    # ------------------------------------------------------------------ #
    use_cache = False
    if rank == 0:
        use_cache = os.path.exists(RESULTS_FILE)
    use_cache = comm.bcast(use_cache, root=0)

    if use_cache:
        if rank == 0:
            print(f"Loading cached results from {RESULTS_FILE} ...")
            with open(RESULTS_FILE, "r") as f:
                bifurcation_data = json.load(f)
            plot_bifurcation_diagram(bifurcation_data)
        comm.Barrier()
        sys.exit(0)

    # ------------------------------------------------------------------ #
    #  Phase 1: background flow (independent of a — compute once)        #
    # ------------------------------------------------------------------ #
    if rank == 0:
        print("Computing background flow (shared across all particle sizes)...")
        R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(
            R, H, W, Q, rho, mu, print_values=True
        )
        bg = background_flow(R_hat, H_hat, W_hat, Re, comm=MPI.COMM_SELF)
        G_hat, U_m_hat, u_bar_2d_hat, p_bar_2d_hat = bg.solve_2D_background_flow()

        scalar_bg = (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat)
        print("Background flow done.")
    else:
        scalar_bg = None

    scalar_bg = comm.bcast(scalar_bg, root=0)
    (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat) = scalar_bg

    # ------------------------------------------------------------------ #
    #  Phase 2: sweep over alpha values                                   #
    # ------------------------------------------------------------------ #
    bifurcation_data = []

    for alpha in ALPHA_VALUES:
        a_phys = float(alpha) * (H / 2)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  alpha = {alpha:.2f}  |  a = {a_phys * 1e6:.2f} µm")
            print(f"{'='*60}")

        # --- second nondimensionalisation on rank 0 (needs Firedrake objects) ---
        iter_params   = None
        u_hh_np = None
        p_hh_np = None

        if rank == 0:
            (R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, U_c_p,
             u_hh, p_hh, Re_p) = second_nondimensionalisation(
                R_hat, H_hat, W_hat, a_phys,
                L_c, U_c, G_hat, Re,
                u_bar_2d_hat, p_bar_2d_hat, U_m_hat,
            )
            u_hh_np = u_hh.dat.data_ro.copy()
            p_hh_np = p_hh.dat.data_ro.copy()
            iter_params = (R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, Re_p)

        iter_params = comm.bcast(iter_params, root=0)
        u_hh_np     = comm.bcast(u_hh_np,    root=0)
        p_hh_np     = comm.bcast(p_hh_np,    root=0)

        (R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, Re_p) = iter_params

        # --- force grid (parallel over all ranks) ---
        if rank == 0:
            print("  Computing force grid ...")

        force_grid = F_p_grid(
            R_hh, H_hh, W_hh, a_hh, G_hh, Re_p,
            L=4 * max(H_hh, W_hh),
            particle_maxh=0.2 * a_hh,
            global_maxh=0.2 * min(H_hh, W_hh),
            eps=0.2 * a_hh,
        )

        grid_values = force_grid.compute_F_p_grid_ensemble(
            N_r=10, N_z=10,
            u_bg_data_np=u_hh_np,
            p_bg_data_np=p_hh_np,
        )

        # --- find & classify equilibria (rank 0 only) ---
        if rank == 0:
            r_vals, z_vals, phi, Fr_grid, Fz_grid = grid_values
            initial_guesses      = force_grid.generate_initial_guesses()
            classified_equilibria = force_grid.classify_equilibria_on_grid(initial_guesses)

            print(f"  Found {len(classified_equilibria)} equilibrium/a.")

            for eq in classified_equilibria:
                r_eq, z_eq = eq["x_eq"]
                # convert doubly-nondim → singly-nondim (r / (H/2))
                r_norm = float(r_eq) * (L_c_p / L_c)
                z_norm = float(z_eq) * (L_c_p / L_c)

                bifurcation_data.append({
                    "a_hat":  float(alpha),
                    "r_raw":  float(r_eq),
                    "z_raw":  float(z_eq),
                    "r_norm": r_norm,
                    "z_norm": z_norm,
                    "type":   eq["type"],
                    "color":  eq["color"],
                })

    # ------------------------------------------------------------------ #
    #  Phase 3: save results and plot                                     #
    # ------------------------------------------------------------------ #
    if rank == 0:
        with open(RESULTS_FILE, "w") as f:
            json.dump(bifurcation_data, f, indent=2)
        print(f"\nResults saved to {RESULTS_FILE}")
        plot_bifurcation_diagram(bifurcation_data)
