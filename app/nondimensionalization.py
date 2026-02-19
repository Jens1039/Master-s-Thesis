from firedrake import *
from app.background_flow import background_flow


def first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=False):

    # characteristic length is the hydraulic diameter D_h = (2*H*W)/(W + H)
    L_c = H/2

    # characteristic velocity is the velocity of a fluid particle without perturbations (based on the input flow rate)
    U_c = Q/(W*H)

    # nondimensionalize every input variable
    R_hat = R/L_c
    H_hat = H/L_c
    W_hat = W/L_c

    # compute the flow Reynolds number
    Re = (rho*U_c*L_c)/mu

    if print_values:
        print("R_hat = ", R_hat)
        print("H_hat = ", H_hat)
        print("W_hat = ", W_hat)
        print("L_c = ", L_c)
        print("U_c = ", U_c)
        print("Re = ", Re)

    return R_hat, H_hat, W_hat, L_c, U_c, Re


def second_nondimensionalisation(R_hat, H_hat, W_hat, a, L_c, U_c, G_hat, Re, u_bar_2d_hat, p_bar_2d_hat, U_m_hat, print_values=False):

    # characteristic length is now the particle diameter a
    L_c_p = a

    # characteristic velocity is the maximal axial velocity with a length correction (analogous to the paper)
    U_c_p = (U_m_hat * U_c) * (L_c_p / L_c)

    # Rescale physical parameters
    R_hat_hat = R_hat * (L_c/L_c_p)
    H_hat_hat = H_hat * (L_c/L_c_p)
    W_hat_hat = W_hat * (L_c/L_c_p)
    a_hat_hat = a * (1.0 / L_c_p)
    G_hat_hat = G_hat * ((U_c / U_c_p)**2) * (L_c_p / L_c)

    # For code hygiene: Create deepcopies to keep the original mesh and functions intact
    mesh_hat = u_bar_2d_hat.function_space().mesh()
    coords_hat_hat = mesh_hat.coordinates.copy(deepcopy=True)

    # Rescale the coordinates of the copied mesh
    coords_hat_hat.dat.data[:] *= (L_c / L_c_p)
    mesh_hat_hat = Mesh(coords_hat_hat)

    # Create function spaces on the rescaled coordinates
    V_hat_hat = FunctionSpace(mesh_hat_hat, u_bar_2d_hat.function_space().ufl_element())
    Q_hat_hat = FunctionSpace(mesh_hat_hat, p_bar_2d_hat.function_space().ufl_element())
    
    # Create functions on the rescaled function spaces
    u_bar_2d_hat_hat = Function(V_hat_hat)
    p_bar_2d_hat_hat = Function(Q_hat_hat)
    
    # Rescale function values (Since the topology and node ordering are identical, we can copy data directly)
    u_bar_2d_hat_hat.dat.data[:] = u_bar_2d_hat.dat.data_ro[:] * (U_c / U_c_p)
    p_bar_2d_hat_hat.dat.data[:] = p_bar_2d_hat.dat.data_ro[:] * ((U_c / U_c_p) ** 2)

    # Compute the particle Reynolds number
    Re_p = Re * U_m_hat * ((L_c_p/L_c)**2)

    if print_values:
        print("R_hat_hat = ", R_hat_hat)
        print("H_hat_hat = ", H_hat_hat)
        print("W_hat_hat = ", W_hat_hat)
        print("a_hat_hat = ", a_hat_hat)
        print("Re_p = ", Re_p)

    return R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, L_c_p, U_c_p, u_bar_2d_hat_hat, p_bar_2d_hat_hat, Re_p


if __name__ == "__main__":

    H = 240e-6
    rho = 998
    mu = 10.02e-4
    a = 0.05 * (H / 2)
    W = 1 * H
    R = 160 * (H / 2)
    Q = 2 * 2.40961923848e-10


    # ---TEST-first_nondimensionalisation---

    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

    assert abs(Re - 1) < 0.01, "Re off"
    a_hat = a / L_c
    kappa = (H_hat ** 4) / (4 * (a_hat ** 3) * R_hat)
    assert abs(kappa - 200) < 0.01, "kappa off"


    # ---TEST-second_nondimensionalisation---

    from background_flow import background_flow

    bg = background_flow(R_hat, H_hat, W_hat, Re)
    G_hat, U_m_hat, u_bar_2d_hat, p_bar_2d_hat = bg.solve_2D_background_flow()
    R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, L_c_p, U_c_p, u_bar_2d_hat_hat, p_bar_2d_hat_hat, Re_p = second_nondimensionalisation(R_hat, H_hat, W_hat, a, L_c, U_c, G_hat, Re, u_bar_2d_hat, p_bar_2d_hat, U_m_hat, print_values=True)

    mesh_hat = u_bar_2d_hat.function_space().mesh()
    mesh_hat_hat = u_bar_2d_hat_hat.function_space().mesh()

    import matplotlib.pyplot as plt
    import firedrake.pyplot as fplt

    plt.figure()
    ax = plt.gca()
    fplt.triplot(mesh_hat, axes=ax)
    fplt.triplot(mesh_hat_hat, axes=ax)
    ax.set_aspect("equal")
    ax.set_title("Overlay: mesh_hat + mesh_hat_hat")
    # plt.show()

    mesh_hat_coordinates_np = mesh_hat.coordinates.dat.data_ro
    mesh_hat_hat_coordinates_np = mesh_hat_hat.coordinates.dat.data_ro
    assert np.allclose(mesh_hat_hat_coordinates_np, (L_c/L_c_p) * mesh_hat_coordinates_np, rtol=1e-12, atol=1e-12), "Mesh coordinates not scaled correctly"

    import numpy as np

    pe_hat = PointEvaluator(mesh_hat, mesh_hat_coordinates_np, redundant=True, missing_points_behaviour="error")
    pe_hat_hat = PointEvaluator(mesh_hat_hat, mesh_hat_hat_coordinates_np, redundant=True, missing_points_behaviour="error")

    vals_hat = pe_hat.evaluate(u_bar_2d_hat)
    vals_hat_hat = pe_hat_hat.evaluate(u_bar_2d_hat_hat)

    diff = np.abs(vals_hat_hat - vals_hat * (U_c/U_c_p))

    assert np.abs(np.max(diff)) < 1e-12, "function value scaling error"

    print("All checks passed.")