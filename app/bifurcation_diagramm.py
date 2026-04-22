import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import plotly.graph_objects as go


from nondimensionalization import *
from find_equilbrium_points import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a_hat_values = np.round(np.arange(0.1375, 0.1376, 0.0025), 7)

RESULTS_FILE = "../images/Sweep_a=0.01_to_0.20_R=500_H=W=2_ss=0.0025/bifurcation_results.json"
PLOT_MODE = "3d"  # allowed: "3d", "2d_r", "2d_z"


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


def plot_bifurcation_diagram(data, plot_mode="3d", save=True, show=True):

    valid_modes = {"3d", "2d_r", "2d_z"}
    if plot_mode not in valid_modes:
        raise ValueError(f"Unsupported plot_mode '{plot_mode}'. Allowed: {sorted(valid_modes)}")

    if go is None:
        raise ImportError("plotly is required for this plot. Install with: pip install plotly")

    if len(data) == 0:
        print("No bifurcation data available to plot.")
        return

    type_styles = {
        "stable":       {"symbol": "circle", "color": "green",  "label": "Stable equilibrium"},
        "unstable":     {"symbol": "circle", "color": "red",    "label": "Unstable equilibrium"},
        "saddle":       {"symbol": "circle", "color": "orange", "label": "Saddle point"},
        "unclassified": {"symbol": "x",      "color": "gray",   "label": "Unclassified"},
    }

    fig = go.Figure()

    for eq_type in ["stable", "unstable", "saddle", "unclassified"]:
        points = [entry for entry in data if entry.get("type", "unclassified") == eq_type]
        if len(points) == 0:
            continue

        style = type_styles.get(eq_type, type_styles["unclassified"])
        x_vals = [entry["a_hat"] for entry in points]
        r_vals = [entry["r_norm"] for entry in points]
        z_vals = [entry["z_norm"] for entry in points]

        if plot_mode == "3d":
            fig.add_trace(go.Scatter3d(
                x=x_vals,
                y=r_vals,
                z=z_vals,
                mode="markers",
                name=style["label"],
                marker=dict(
                    size=6,
                    color=style["color"],
                    symbol=style["symbol"],
                    line=dict(color="black", width=1),
                    opacity=0.9,
                ),
                hovertemplate=(
                    "a/(H/2): %{x:.4f}<br>"
                    "r/(H/2): %{y:.4f}<br>"
                    "z/(H/2): %{z:.4f}<br>"
                    f"type: {eq_type}<extra></extra>"
                ),
            ))
        elif plot_mode == "2d_r":
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=r_vals,
                mode="markers",
                name=style["label"],
                marker=dict(
                    size=9,
                    color=style["color"],
                    symbol=style["symbol"],
                    line=dict(color="black", width=1),
                    opacity=0.9,
                ),
                hovertemplate=(
                    "a/(H/2): %{x:.4f}<br>"
                    "r/(H/2): %{y:.4f}<br>"
                    f"type: {eq_type}<extra></extra>"
                ),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=z_vals,
                mode="markers",
                name=style["label"],
                marker=dict(
                    size=9,
                    color=style["color"],
                    symbol=style["symbol"],
                    line=dict(color="black", width=1),
                    opacity=0.9,
                ),
                hovertemplate=(
                    "a/(H/2): %{x:.4f}<br>"
                    "z/(H/2): %{y:.4f}<br>"
                    f"type: {eq_type}<extra></extra>"
                ),
            ))

    W_hat = W / (H / 2)
    H_hat = H / (H / 2)
    x_min = min(a_hat_values) - 0.005
    x_max = max(a_hat_values) + 0.005

    if plot_mode == "3d":
        fig.update_layout(
            title="Bifurcation Diagram (3D): particle size vs equilibrium positions",
            scene=dict(
                xaxis_title="Relative particle size a/(H/2)",
                yaxis_title="Equilibrium r/(H/2)",
                zaxis_title="Equilibrium z/(H/2)",
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[-W_hat / 2, W_hat / 2]),
                zaxis=dict(range=[-H_hat / 2, H_hat / 2]),
            ),
            legend=dict(x=0.01, y=0.99),
            template="plotly_white",
        )
    elif plot_mode == "2d_r":
        fig.add_hline(y=W_hat / 2, line_dash="dash", line_color="black")
        fig.add_hline(y=-W_hat / 2, line_dash="dash", line_color="black")
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Bifurcation Diagram (2D projection on r-axis)",
            xaxis_title="Relative particle size a/(H/2)",
            yaxis_title="Equilibrium r/(H/2)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[-W_hat / 2, W_hat / 2]),
            legend=dict(x=0.01, y=0.99),
            template="plotly_white",
        )
    else:
        fig.add_hline(y=H_hat / 2, line_dash="dash", line_color="black")
        fig.add_hline(y=-H_hat / 2, line_dash="dash", line_color="black")
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
        fig.update_layout(
            title="Bifurcation Diagram (2D projection on z-axis)",
            xaxis_title="Relative particle size a/(H/2)",
            yaxis_title="Equilibrium z/(H/2)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[-H_hat / 2, H_hat / 2]),
            legend=dict(x=0.01, y=0.99),
            template="plotly_white",
        )

    if save:
        a_min = min(a_hat_values)
        a_max = max(a_hat_values)
        out_dir = "../images/Sweep_a=0.01_to_0.20_R=500_H=W=2_ss=0.0025/images"
        os.makedirs(out_dir, exist_ok=True)
        html_path = (
            f"{out_dir}/Bifurcation_diagram_{plot_mode}_a_min={a_min:.3f}_a_max={a_max:.3f}"
            f"_R={R:.0f}_W={W:.0f}_H={H:.0f}_N_grid={N_grid}.html"
        )
        fig.write_html(html_path)
        print(f"Bifurcation diagram saved to {html_path}")

    if show:
        fig.show()


if __name__ == "__main__":

    from config_paper_parameters import *

    auto_start_mpi()

    os.makedirs("../images/Sweep_a=0.01_to_0.20_R=500_H=W=2_ss=0.0025/images", exist_ok=True)

    use_cache = False
    if rank == 0:
        use_cache = os.path.exists(RESULTS_FILE)
    use_cache = comm.bcast(use_cache, root=0)

    if use_cache:
        if rank == 0:
            print(f"Loading cached results from {RESULTS_FILE} ...")
            with open(RESULTS_FILE, "r") as f:
                bifurcation_data = json.load(f)
            plot_bifurcation_diagram(bifurcation_data, plot_mode=PLOT_MODE, save=True, show=True)
        comm.Barrier()
        sys.exit(0)

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

    bifurcation_data = []

    for a_hat in a_hat_values:
        a_phys = float(a_hat) * (H / 2)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  a_hat = {a_hat:.5f}  |  a = {a_phys * 1e6:.2f} µm")
            print(f"{'='*60}")

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

        if rank == 0:
            print("  Computing force grid ...")

        force_grid = F_p_grid(
            R_hh, H_hh, W_hh, a_hh, G_hh, Re_p,
            L=4 * max(H_hh, W_hh),
            particle_maxh=particle_maxh_rel * a_hh,
            global_maxh=global_maxh_rel * min(H_hh, W_hh),
            eps=0.2 * a_hh,
        )

        grid_values = force_grid.compute_F_p_grid_ensemble(
            N_grid=N_grid,
            u_bg_data_np=u_hh_np,
            p_bg_data_np=p_hh_np,
        )

        if rank == 0:
            r_vals, z_vals, phi, Fr_grid, Fz_grid = grid_values
            initial_guesses      = force_grid.generate_initial_guesses()
            classified_equilibria = force_grid.classify_equilibria_on_grid(initial_guesses)

            force_grid.plot(L_c_p, L_c, classified_equilibria=classified_equilibria)

            print(f"  Found {len(classified_equilibria)} equilibrium/a.")

            for eq in classified_equilibria:
                r_eq, z_eq = eq["x_eq"]
                r_norm = float(r_eq) * (L_c_p / L_c)
                z_norm = float(z_eq) * (L_c_p / L_c)

                bifurcation_data.append({
                    "a_hat":  float(a_hat),
                    "r_raw":  float(r_eq),
                    "z_raw":  float(z_eq),
                    "r_norm": r_norm,
                    "z_norm": z_norm,
                    "type":   eq["type"],
                    "color":  eq["color"],
                })

            with open(RESULTS_FILE, "w") as f:
                json.dump(bifurcation_data, f, indent=2)
            print(f"  Results written to {RESULTS_FILE} ({len(bifurcation_data)} entries total)")

    if rank == 0:
        plot_bifurcation_diagram(bifurcation_data, plot_mode=PLOT_MODE, save=True, show=True)
