import os
os.environ["OMP_NUM_THREADS"]="1"
import pickle

from config_lab_parameters import *

from background_flow import background_flow
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow
from compute_force_grid import sample_grid
from plot_everything import plot_2d_background_flow, plot_curved_channel_section_with_spherical_hole, plot_3d_background_flow, plot_force_grid



if __name__ == "__main__":

    bg = background_flow(R, H, W, Q, Re)
    bg.solve_2D_background_flow()
    plot_2d_background_flow(bg.mesh2d, bg.u_bar)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, W, H, L, a, particle_maxh, global_maxh, r_off=0.0, z_off=0.0, order=3)
    plot_curved_channel_section_with_spherical_hole(mesh3d)

    filename = f"lift_force_grid_{N_r}_times_{N_z}.pkl"
    filepath = os.path.join(os.path.join(os.path.dirname(__file__), "..", "cache"), filename)

    if os.path.exists(filepath):
        print(f"Load data from cache: {filepath}")
        with open(filepath, "rb") as f:
                data = pickle.load(f)
    else:
        print("Calculate data new...")

        data = sample_grid(R, H, W, Q, L, a, particle_maxh, global_maxh, Re, N_r=N_r, N_z=N_z)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Data stored in the cache under: {filepath}")

    plot_force_grid(data)