from firedrake import *
from copy import deepcopy

def first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=False):

    # characteristic length is the hydraulic diameter D_h
    L_c = (2*H*W)/(W + H)

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
        print("W_hat= ", W_hat)
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
    coords_hat_hat.dat.data[:] *= (L_c / L_c_p)
    mesh_hat_hat = Mesh(coords_hat_hat)
    # Rescale the mesh for the velocity (and automatically also the pressure since they live on the same mesh)
    mesh_hat_hat.coordinates.dat.data[:] *= (L_c / L_c_p)

    V_hat_hat = FunctionSpace(mesh_hat_hat, u_bar_2d_hat.function_space().ufl_element())
    u_bar_2d_hat_hat = Function(V_hat_hat)
    # Since the topology and node ordering are identical, we can copy data directly
    u_bar_2d_hat_hat.dat.data[:] = u_bar_2d_hat.dat.data_ro[:] * (U_c / U_c_p)

    Q_hat_hat = FunctionSpace(mesh_hat_hat, p_bar_2d_hat.function_space().ufl_element())
    p_bar_2d_hat_hat = Function(Q_hat_hat)
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