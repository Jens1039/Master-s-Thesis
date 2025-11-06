import numpy as np

# ---INPUT---------------------------------------------
d_1 = 8e-6          # diameter of the smaller cell [m]
d_2 = 12e-6         # diameter of the bigger cell [m]
# ------------------------------------------------------

# Notice: right now, we're choosing standard values used in most microfluidic chips
# ---VARIABLE ASSUMPTION--------------------------------
R = 2800e-6         # bend radius [m]
H = 120e-6          # duct height [m]
W = 500e-6          # duct width  [m]
Q = 3e-8            # volumetric flow rate [m^3/s]
# ------------------------------------------------------

# ---FIXED ASSUMPTIONS----------------------------------
rho = 998           # density [kg/m^3]
mu  = 10.02e-4      # dyn. viscosity [Pa·s]
# ------------------------------------------------------

# ---RESOLUTION OF THE GRID OF PARTICLE FORCES----------
N_r = 20
N_z = 20
# ------------------------------------------------------

def nondimensionalize(d_1, d_2, R, H, W, Q, rho, mu, print_values=False):

    # Characteristic length is the hydraulic diameter D_h
    L_ref = (2*H*W)/(W + H)

    # Characteristic velocity is the velocity of a fluid particle without perturbations (based on the input flow rate)
    U_ref = Q/(W*H)

    # Characteristic time is the time, that it takes for a fluid particle to cross the hydraulic diameter
    T_ref = L_ref/U_ref

    # typical reference pressure in fluid dynamics
    P_ref = rho*(U_ref**2)

    # nondimensionalize every input variable
    a_1_nd = d_1/(2*L_ref)
    a_2_nd = d_2/(2*L_ref)
    R_nd = R/L_ref
    H_nd = H/L_ref
    W_nd = W/L_ref

    # we nondimensionalize the Navier-Stokes equation (and its reduced forms) in the classical way, therefore mu and rho are contained in Re
    Re = (rho*U_ref*L_ref)/mu

    # in our case Re_p to is a second important quantity to determine whether we can use a perturbation expansion to calculate the perturbed flow
    Re_p = Re*(a_2_nd)**2

    # the dean number is important to determine the secondary fluid regime
    De = Re * np.sqrt(L_ref / (2 * R))

    if print_values:
        print("H = ", H_nd)
        print("W = ", W_nd)
        print("R = ", R_nd)
        print("a_1 = ", a_1_nd)
        print("a_2 = ", a_2_nd)
        print("Re = ", Re)
        print("Re_p = ", Re_p)
        print("De = ", De)

    return a_1_nd, a_2_nd, H_nd, W_nd, R_nd, Re, Re_p

a_1, a_2, H, W, R, Re, Re_p = nondimensionalize(d_1, d_2, R, H, W, Q, rho, mu, print_values=False)

Q = Re * mu * H / rho # in an updatet Version the computation of G will take place here and Q will be eleviated from the main



















