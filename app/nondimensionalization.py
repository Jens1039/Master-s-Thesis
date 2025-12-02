

def first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=False):

    # characteristic length is the hydraulic diameter D_h
    L_c = (2*H*W)/(W + H)

    # characteristic velocity is the velocity of a fluid particle without perturbations (based on the input flow rate)
    U_c = Q/(W*H)

    # nondimensionalize every input variable
    R_nd = R/L_c
    H_nd = H/L_c
    W_nd = W/L_c

    # we compute the flow Reynolds number
    Re = (rho*U_c*L_c)/mu

    if print_values:
        print("R = ", R_nd)
        print("H = ", H_nd)
        print("W = ", W_nd)
        print("L_c = ", L_c)
        print("U_c = ", U_c)
        print("Re = ", Re)

    return R_nd, H_nd, W_nd, L_c, U_c, Re


def second_nondimensionalisation(R, H, W, a, L_c, U_c, Re, u_bar_2d, p_bar_2d, U_m, print_values=False):

    # characteristic length is now the particle diameter a
    L_c_p = a

    # characteristic velocity is the maximal axial velocity
    U_c_p = (U_m*U_c) * (L_c_p/L_c)

    # We translate the variables from the first nondimensionalisation system to the second one
    R_nd = (L_c/L_c_p) * R
    H_nd = (L_c/L_c_p) * H
    W_nd = (L_c/L_c_p) * W
    a_nd = (1/L_c_p) * a    # Note, that a does not need to be unscaled from the first nondimensionalisation, since it is not used for the background flow
    u_bar_nd = (U_c/U_c_p) * u_bar_2d.dat.data_ro
    p_bar_nd = (Re / U_m) * p_bar_2d.dat.data_ro

    # we compute the flow Reynolds number, which we later use for a perturbation expansion
    Re_p = Re * (U_c_p/U_c) * (L_c_p/L_c)

    if print_values:
        print("R = ", R_nd)
        print("H = ", H_nd)
        print("W = ", W_nd)
        print("a = ", a_nd)
        print("u_bar = ", u_bar_nd)
        print("p_bar = ", p_bar_nd)
        print("Re_p = ", Re_p)

    return R_nd, H_nd, W_nd, a_nd, U_c_p, u_bar_nd, p_bar_nd, Re_p