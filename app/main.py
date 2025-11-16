import os
os.environ["OMP_NUM_THREADS"] = "1"

from config_paper_parameters import *

from background_flow import background_flow
from Backup.plot_everything import (
    plot_2d_background_flow,
    plot_curved_channel_section_with_spherical_hole,
    plot_3d_background_flow,
    plot_force_grid,
)
from find_equilbrium_points import FpEvaluator, DeflatedRootFinder, coarse_candidates, plot_equilibria


if __name__ == "__main__":

    # 1. Hintergrundströmung einmal lösen
    bg_flow = background_flow(R, H, W, Q, Re)
    bg_flow.solve_2D_background_flow()

    # 2. Evaluator für F_p erstellen
    fp_eval = FpEvaluator(R, W, H, L, a,
                          particle_maxh, global_maxh,
                          Re, Re_p,
                          bg_flow=bg_flow)

    # 3. Grobes Gitter zur Kandidatensuche (z.B. 5x5)
    #    Du kannst n_r / n_z hier natürlich ändern (z.B. 7x7 etc.)
    candidate_x0 = coarse_candidates(fp_eval, n_r=5, n_z=5, verbose=True)

    # 4. Deflations-Solver aufsetzen
    #    tol_F hier absichtlich etwas entspannter (10^-6)
    root_finder = DeflatedRootFinder(fp_eval,
                                     alpha=1.0,
                                     power=2.0,
                                     tol_F=1e-6,
                                     tol_x=1e-3,
                                     max_newton_it=12,
                                     fd_step=1e-4,
                                     min_root_dist=5e-3)

    # 5. Alle Nullstellen suchen, beginnend von den Kandidaten
    roots = root_finder.find_all_roots(candidate_x0, verbose=True)

    print("\nGefundene Gleichgewichtslagen (r_off, z_off):")
    for r in roots:
        print(r)

    # 6. Visualisieren und Typ bestimmen
    fig, ax, eq_info = plot_equilibria(fp_eval, roots, fd_step=1e-3, title="Equilibria & Stabilität")
    fig.tight_layout()
    fig.savefig("equilibria_stability.png", dpi=200)
    # oder plt.show()