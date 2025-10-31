import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *

from preprocess_and_nondimensionalize import Nondimensionalizer
from background_flow import get_G, solve_2D_background_flow, make_curved_channel_section_with_spherical_hole, build_background_flow
from perturbed_flow import Stokes_solver_3d, F_minus_1_a, T_minus_1_a, numerically_stable_solver_2x2
from plot_everything import plot_2D_background_flow, plot_curved_channel_section_with_spherical_hole, plot_background_flow


# ---INPUT---------------------------------------------
d_1 = 8e-6          # diameter of the smaller cell [m]
d_2 = 12e-6         # diameter of the bigger cell [m]
# tau_max = 20.0      # maximal shear stress on cells [Pa]
# ------------------------------------------------------

# ---VARIABLE ASSUMPTION--------------------------------
R = 500e-6          # bend radius [m]
H = 120e-6          # duct height [m]
W = H               # duct width  [m]
Q = 3e-9            # volumetric flow rate [m^3/s]
# ------------------------------------------------------

# ---FIXED ASSUMPTIONS----------------------------------
rho = 1000.0        # density [kg/m^3]
mu  = 1.2e-3        # dyn. viscosity of water [Pa·s]
# ------------------------------------------------------



if __name__ == "__main__":

    # find_optimal_H_W_and_Q() idea: optimise the input variable according to biological constraints.
    # right now, we're choosing standard values used in most microfluidic chips

    result_nd = Nondimensionalizer(d_1, d_2, R, rho, mu, H, W, Q)

    dimensionless = result_nd["dimensionless"]

    d_1 = dimensionless["d1"]
    d_2 = dimensionless["d2"]
    R = dimensionless["Rc"]
    rho = dimensionless["rho"]
    mu = dimensionless["mu"]
    H = dimensionless["H"]
    W = dimensionless["W"]
    Q = dimensionless["Q"]
    Re = dimensionless["Re"]
    De = dimensionless["De"]
    print("Re = ", Re)
    print("De = ", De)

    nx, nz = 60, 60
    mesh2d = RectangleMesh(nx, nz, W, H, quadrilateral=False)

    G = get_G(R, W, Q, mesh2d)
    print("G = ", G)

    u_bar_2d, p_bar_2d = solve_2D_background_flow(mesh2d, rho, mu, R, G, H, W)
    # plot_2D_background_flow(mesh2d, u_bar_2d)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, W, H, L=2.5, a=d_1/2, h=0.3)
    # plot_curved_channel_section_with_spherical_hole(mesh3d)

    u_bar_3d, p_bar_3d = build_background_flow(u_bar_2d, p_bar_2d, mesh2d, mesh3d, R, W, H)
    # plot_background_flow(mesh3d, u_bar_3d, p_bar_3d)

    x = SpatialCoordinate(mesh3d)
    x_p = tags["center"]
    r = sqrt(x[0] ** 2 + x[1] ** 2)
    e_theta = as_vector((-x[1] / r, x[0] / r, 0.0))

    bcs_Theta = cross(as_vector((0.0, 0.0, 1.0)), as_vector(x))
    bcs_Omega = cross(as_vector((0.0, 0.0, 1.0)), as_vector(x) - as_vector(x_p))
    bcs_bg = - -dot(u_bar_3d, e_theta) * e_theta

    v_0_a_Theta, q_0_a_Theta = Stokes_solver_3d(mesh3d, bcs_Theta, tags["walls"], tags["particle"], mu)
    v_0_a_Omega, q_0_a_Omega = Stokes_solver_3d(mesh3d, bcs_Omega, tags["walls"], tags["particle"], mu)
    v_0_a_bg, q_0_a_bg = Stokes_solver_3d(mesh3d, bcs_bg, tags["walls"], tags["particle"], mu)

    F_minus_1_a_Omega = F_minus_1_a(v_0_a_Theta, q_0_a_Theta, mesh3d, tags["particle"])
    F_minus_1_a_Theta = F_minus_1_a(v_0_a_Omega, q_0_a_Omega, mesh3d, tags["particle"])
    F_minus_1_a_bg = F_minus_1_a(v_0_a_bg, q_0_a_bg,mesh3d, tags["particle"])

    T_Theta = T_minus_1_a(v_0_a_Theta, q_0_a_Theta, mesh3d, tags["particle"], x_p)
    T_Omega = T_minus_1_a(v_0_a_Omega, q_0_a_Omega, mesh3d, tags["particle"], x_p)
    T_minus_1_a_bg = T_minus_1_a(v_0_a_bg, q_0_a_bg, mesh3d, tags["particle"], x_p)

    x0, y0, z0 = tags["center"]
    r0 = np.hypot(x0, y0)

    A = np.array([
        [np.dot(np.array([-y0 / r0, x0 / r0, 0.0]), F_minus_1_a_Theta), np.dot(np.array([-y0 / r0, x0 / r0, 0.0]), F_minus_1_a_Omega)],
        [np.dot(np.array([0.0, 0.0, 1.0]), T_Theta), np.dot(np.array([0.0, 0.0, 1.0]), T_Omega)]
    ], dtype=float)

    b = -np.array([
        np.dot(np.array([-y0 / r0, x0 / r0, 0.0]), F_minus_1_a_bg),
        np.dot(np.array([0.0, 0.0, 1.0]), T_minus_1_a_bg)
    ], dtype=float)

    Theta, Omega = numerically_stable_solver_2x2(A, b, tol = 1e-10)