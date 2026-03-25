import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from build_3d_geometry_gmsh import *

if __name__ == '__main__':

    # =========================================================================
    # --- INPUT
    # =========================================================================

    R_hat = 500.0
    H_hat = 2.0
    W_hat = 2.0
    L_hat = 4 * max(H_hat, W_hat)
    Re = 1.0

    # -------------------------------------------------------------------------

    # Initial guess (from the bifurcation diagram)
    r_tilde = 0.61
    z_tilde = 0.0
    a_tilde = 0.135

    # Initial domain
    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R_hat, H_hat, W_hat, L=L_hat, a=a_tilde,
        particle_maxh=0.2 * a_tilde,
        global_maxh=0.2 * min(H_hat, W_hat),
        r_off=r_tilde, z_off=z_tilde
    )

    # target bifurcation parameter
    a_star = 0.08

    # tolerance
    epsilon = 1e-6

    # =========================================================================
    # --- 1.  Solve the eigenvalue problem to generate initial guesses
    # =========================================================================



    # =========================================================================
    # --- 2 - 4.  Solve the Moore–Spence system to obtain a solution ((r_i_bar, z_i_bar), a_i_bar, phi_i_bar)
    # =========================================================================



    # =========================================================================
    # --- 5. Select the initial solution (u^0, a^0, phi^0)
    # =========================================================================







