import os
os.environ["OMP_NUM_THREADS"] = "1"

import json
import sys
import warnings
import plotly.graph_objects as go

from problem_setup import *
from find_equilbrium_points import *

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*import SLEPc.*", category=UserWarning)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


RESULTS_FILE = f"../images/bifurcation_results.json"


def plot_bifurcation_diagram(data, save=True, show=True):

    if go is None:
        raise ImportError("plotly is required for this plot. Install with: pip install plotly")

    if len(data) == 0:
        print("No bifurcation data available to plot.")
        return

    type_styles = {
        "attracting":   {"symbol": "circle", "color": "green",  "label": "Attracting (stable)"},
        "repelling":    {"symbol": "circle", "color": "red",    "label": "Repelling (unstable)"},
        "saddle":       {"symbol": "circle", "color": "orange", "label": "Saddle point"},
        "unclassified": {"symbol": "x",      "color": "gray",   "label": "Unclassified"},
    }

    fig = go.Figure()

    for eq_type in ["attracting", "repelling", "saddle", "unclassified"]:
        points = [entry for entry in data if entry.get("type", "unclassified") == eq_type]
        if len(points) == 0:
            continue

        style = type_styles.get(eq_type, type_styles["unclassified"])
        x_vals = [entry["a"] for entry in points]
        r_vals = [entry["r_norm"] for entry in points]
        z_vals = [entry["z_norm"] for entry in points]

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

    x_min = min(a_values) - 0.005
    x_max = max(a_values) + 0.005

    fig.update_layout(
        title="Bifurcation Diagram (3D): particle size vs equilibrium positions",
        scene=dict(
            xaxis_title="Relative particle size a/(H/2)",
            yaxis_title="Equilibrium r/(H/2)",
            zaxis_title="Equilibrium z/(H/2)",
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[-W / 2, W / 2]),
            zaxis=dict(range=[-H / 2, H / 2]),
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
    )

    if save:
        a_min = min(a_values)
        a_max = max(a_values)
        out_dir = "../images/Sweep_a=0.134_to_0.137_R=500_H=W=2_ss=0.0025"
        os.makedirs(out_dir, exist_ok=True)
        html_path = (
            f"{out_dir}/Bifurcation_diagram_3d_a_min={a_min:.3f}_a_max={a_max:.3f}"
            f"_R={R:.0f}_W={W:.0f}_H={H:.0f}_N_grid={N_grid}.html"
        )
        fig.write_html(html_path)
        print(f"Bifurcation diagram saved to {html_path}")

    if show:
        fig.show()


if __name__ == "__main__":

    os.makedirs("../images/Sweep_a=0.134_to_0.137_R=500_H=W=2_ss=0.0025", exist_ok=True)

    use_cache = False
    if rank == 0:
        use_cache = os.path.exists(RESULTS_FILE)
    use_cache = comm.bcast(use_cache, root=0)

    if use_cache:
        if rank == 0:
            print(f"Loading cached results from {RESULTS_FILE} ...")
            with open(RESULTS_FILE, "r") as f:
                bifurcation_data = json.load(f)
            plot_bifurcation_diagram(bifurcation_data, save=True, show=True)
        comm.Barrier()
        sys.exit(0)

    if rank == 0:
        print("Computing background flow (shared across all particle sizes)...")
        bg = background_flow(R, H, W, Re, comm=MPI.COMM_SELF)
        G, U_m, u_bar_2d, p_bar_2d = bg.solve_2D_background_flow()

        u_data_np = u_bar_2d.dat.data_ro.copy()
        p_data_np = p_bar_2d.dat.data_ro.copy()

        scalar_bg = (R, H, W, L_c, U_c, Re, G, U_m)
        print("Background flow done.")
    else:
        scalar_bg = None
        u_data_np = None
        p_data_np = None

    scalar_bg = comm.bcast(scalar_bg, root=0)
    u_data_np = comm.bcast(u_data_np, root=0)
    p_data_np = comm.bcast(p_data_np, root=0)
    (R, H, W, L_c, U_c, Re, G, U_m) = scalar_bg

    bifurcation_data = []

    for a in a_values:
        a_si = float(a) * L_c   # dimensional radius in meters, for display only

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"  a = {a:.5f}  |  a = {a_si * 1e6:.2f} µm")
            print(f"{'='*60}")
            print("  Computing force grid ...")

        force_grid = F_p_grid(
            R, H, W, a, G, Re,
            L=4 * max(H, W),
            particle_maxh=particle_maxh_rel * a,
            global_maxh=global_maxh_rel * min(H, W),
            eps=0.2 * a,
            U_m=U_m,
        )

        grid_values = force_grid.compute_F_p_grid_ensemble(
            N_grid=N_grid,
            u_bg_data_np=u_data_np,
            p_bg_data_np=p_data_np,
        )

        if rank == 0:
            r_vals, z_vals, phi, Fr_grid, Fz_grid = grid_values
            initial_guesses      = force_grid.generate_initial_guesses()
            classified_equilibria = force_grid.classify_equilibria_on_grid(initial_guesses)

            force_grid.plot(classified_equilibria=classified_equilibria)

            print(f"  Found {len(classified_equilibria)} equilibrium/a.")

            for eq in classified_equilibria:
                r_eq, z_eq = eq["x_eq"]

                bifurcation_data.append({
                    "a":  float(a),
                    "r_raw":  float(r_eq),
                    "z_raw":  float(z_eq),
                    "r_norm": float(r_eq),
                    "z_norm": float(z_eq),
                    "type":   eq["type"],
                    "color":  eq["color"],
                })

            with open(RESULTS_FILE, "w") as f:
                json.dump(bifurcation_data, f, indent=2)
            print(f"  Results written to {RESULTS_FILE} ({len(bifurcation_data)} entries total)")

    if rank == 0:
        plot_bifurcation_diagram(bifurcation_data, save=True, show=True)
