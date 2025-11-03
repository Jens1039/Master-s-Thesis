import os

os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import pickle

from preprocess_and_nondimensionalize import Nondimensionalizer
from background_flow import background_flow, make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow, F_p_r_z
from find_equilibrium_points import sample_grid
from plot_everything import plot_2D_background_flow, plot_curved_channel_section_with_spherical_hole, plot_background_flow, plot_lift_force_field_with_streamlines

# ---INPUT---------------------------------------------
d_1 = 8e-6          # diameter of the smaller cell [m]
d_2 = 12e-6         # diameter of the bigger cell [m]
# tau_max = 20.0      # maximal shear stress on cells [Pa]
# ------------------------------------------------------

# ---VARIABLE ASSUMPTION--------------------------------
R = 1000e-5          # bend radius [m]
H = 120e-6          # duct height [m]
W = H               # duct width  [m]
Q = 1e-9            # volumetric flow rate [m^3/s]
# ------------------------------------------------------

# ---FIXED ASSUMPTIONS----------------------------------
rho = 1000.0        # density [kg/m^3]
mu  = 1.2e-3        # dyn. viscosity of water [Pa·s]
# ------------------------------------------------------



if __name__ == "__main__":

    # find_optimal_H_W_and_Q() idea: optimise the input variable according to biological constraints.
    # right now, we're choosing standard values used in most microfluidic chips

    # d1_nd, d2_nd, R_nd, H_nd, W_nd, Q_nd, rho_nd, mu_nd, Re, De = Nondimensionalizer(...)

    # For reproducing figure 2:
    W = 2.0
    H = 2.0
    R = 160.0
    a = 0.05

    L = 3
    h = 0.02
    mu = 1
    Re = 6
    G = 1


    mesh2d = RectangleMesh(120, 120, W, H, quadrilateral=False)
    print("background_flow_solved")

    bg = background_flow(R, H, W, Q, rho=1.0, mu=1.0, a=a)
    bg.get_G(mesh2d, calculate_G=False, G=G)
    bg.solve_2D_background_flow(mesh2d)

    def sample_grid_cached(background_flow, R, H, W, L, a, h, Re, N_r, N_z, eps=1e-10, cache_file="cache_force_grid.pkl"):
        if os.path.exists(cache_file):
            print(f"Cache found: loading {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print("Computing new grid (this may take a while)...")
        grid = sample_grid(background_flow, R, H, W, L, a, h, Re, N_r, N_z, eps)

        with open(cache_file, "wb") as f:
            pickle.dump(grid, f)
            print(f"Saved cache to {cache_file}")

        return grid

    gridinfos = sample_grid_cached(bg, R, H, W, L, a, h, Re, N_r=10, N_z=10)

    plot_lift_force_field_with_streamlines(gridinfos, W, H, a)