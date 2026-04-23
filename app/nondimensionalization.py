import os
os.environ["OMP_NUM_THREADS"] = "1"


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
        print("Re = ", Re)

    return R_hat, H_hat, W_hat, L_c, U_c, Re