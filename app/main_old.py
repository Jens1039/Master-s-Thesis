import os
os.environ["OMP_NUM_THREADS"]="1"
from firedrake import *
import pickle

from config_paper_parameters import *

from background_flow import background_flow_G_approximated
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow
from compute_force_grid import sample_grid
from plot_everything import plot_2d_background_flow, plot_curved_channel_section_with_spherical_hole, plot_3d_background_flow, plot_force_grid



if __name__ == "__main__":

    filename = f"lift_force_grid_{N_r}_times_{N_z}.pkl"
    filepath = os.path.join(os.path.join(os.path.dirname(__file__), "..", "cache"), filename)


    if os.path.exists(filepath):
        print(f"Load data from cache: {filepath}")
        with open(filepath, "rb") as f:
                data = pickle.load(f)
    else:
        print("Calculate data new...")

        bg = background_flow_G_approximated(R, H, W, Q, rho, mu, a)
        bg.solve_2D_background_flow()

        data = sample_grid(bg, R, H, W, L, a, particle_maxh, global_maxh, Re, N_r=N_r, N_z=N_z, nproc=12)
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Data stored in the cache under: {filepath}")


    plot_force_grid(data, a)