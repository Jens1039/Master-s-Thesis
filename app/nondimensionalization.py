from firedrake import *

def first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True):

    # characteristic length is the hydraulic diameter D_h
    L_c = (2*H*W)/(W + H)

    # characteristic velocity is the velocity of a fluid particle without perturbations (based on the input flow rate)
    U_c = Q/(W*H)

    # nondimensionalize every input variable
    R_hat = R/L_c
    H_hat = H/L_c
    W_hat = W/L_c

    # we compute the flow Reynolds number
    Re = (rho*U_c*L_c)/mu

    if print_values:
        print("R_hat = ", R_hat)
        print("H_hat = ", H_hat)
        print("W_hat= ", W_hat)
        print("L_c = ", L_c)
        print("U_c = ", U_c)
        print("Re = ", Re)

    return R_hat, H_hat, W_hat, L_c, U_c, Re


def second_nondimensionalisation(R_hat, H_hat, W_hat, a, L_c, U_c, G_hat, Re, u_bar_2d_hat, p_bar_2d_hat, U_m_hat, print_values=True):

    # characteristic length is now the particle diameter a
    L_c_p = a

    # characteristic velocity is the maximal axial velocity with a length correction (analogous to the paper)
    U_c_p = (U_m_hat * U_c) * (L_c_p / L_c)

    # Rescale Scalar Geometry Parameters
    R_hat_hat = R_hat * (L_c/L_c_p)
    H_hat_hat = H_hat * (L_c/L_c_p)
    W_hat_hat = W_hat * (L_c/L_c_p)
    a_hat_hat = a * (1.0 / L_c_p)
    G_hat_hat = (L_c/L_c_p) * (U_c_p/U_c) * G_hat

    mesh2d = u_bar_2d_hat.function_space().mesh()
    mesh2d.coordinates.dat.data[:] *= (L_c/L_c_p)
    u_bar_2d_hat.dat.data[:] *= (U_c/U_c_p)
    p_bar_2d_hat.dat.data[:] *= ((U_c/U_c_p)**2)
    u_bar_2d_hat_hat = u_bar_2d_hat
    p_bar_2d_hat_hat = p_bar_2d_hat

    Re_p = Re * U_m_hat * ((L_c_p/L_c)**2)

    if print_values:
        print("R_hat_hat = ", R_hat_hat)
        print("H_hat_hat = ", H_hat_hat)
        print("W_hat_hat = ", W_hat_hat)
        print("a_hat_hat = ", a_hat_hat)
        print("Re_p = ", Re_p)

    return R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, L_c_p, U_c_p, u_bar_2d_hat_hat, p_bar_2d_hat_hat, G_hat_hat, Re_p