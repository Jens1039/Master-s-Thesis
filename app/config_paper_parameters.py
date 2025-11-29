import numpy as np

# ---REALISTIC PHYSICAL ASSUMPTIONS---------------------
H = 240e-6                  # duct height [m]
rho = 998                   # density [kg/m^3]
mu  = 10.02e-4              # dyn. viscosity [Pa·s]
# ------------------------------------------------------

# ---LENGTH RATIOS FROM THE PAPER-----------------------
a = 0.05*H / 2               # diameter of the smaller cell [m]
W = H                        # duct width  [m]
R = 160*H / 2                # bend radius [m]
# ------------------------------------------------------

# ---VOLUMETRIC FLOW RATE (CHOSEN TO MATCH ASSUMPTIONS FROM THE PAPER)
Q = 1e-9   # 3e-8            # volumetric flow rate [m^3/s]
# ------------------------------------------------------


# For numerical conditioning and comparability reasons, all our computations should be done in the nondimensional settings
def nondimensionalize(a, R, H, W, Q, rho, mu, print_values=False):

    # Characteristic length is the hydraulic diameter D_h
    L_c = (2*H*W)/(W + H)

    # Characteristic velocity is the velocity of a fluid particle without perturbations (based on the input flow rate)
    U_c = Q/(W*H)
    print("U_c = ", U_c)

    # Characteristic time is the time, that it takes for a fluid particle to cross the hydraulic diameter
    T_ref = L_c/U_c

    # typical reference pressure in fluid dynamics
    P_ref = rho*(U_c**2)

    # nondimensionalize every input variable
    a_nd = a/L_c
    R_nd = R/L_c
    H_nd = H/L_c
    W_nd = W/L_c

    # we nondimensionalize the Navier-Stokes equation (and its reduced forms) in the classical way, therefore mu and rho are contained in Re
    Re = (rho*U_c*L_c)/mu

    # in our case Re_p to is a second important quantity to determine whether we can use a perturbation expansion to calculate the perturbed flow
    Re_p = Re*((a_nd)**2)

    if print_values:
        print("H = ", H_nd)
        print("W = ", W_nd)
        print("R = ", R_nd)
        print("a = ", a_nd)
        print("Re = ", Re)
        print("Re_p = ", Re_p)

    return a_nd, H_nd, W_nd, R_nd, Re, Re_p

a, H, W, R, Re, Re_p = nondimensionalize(a, R, H, W, Q, rho, mu, print_values=True)

'''
H =  1.0000000000000002
W =  1.0000000000000002
R =  80.00000000000001
a =  0.025
Re =  4.150033266799734
Re_p =  0.002593770791749834
'''

# ---LENGTH OF THE DUCT SECTION WE'RE LOOKING AT--------
L = 4*max(W, H)
# ------------------------------------------------------

# ---RESOLUTION OF THE GRID OF PARTICLE FORCES----------
N_r = 10
N_z = 10
# ------------------------------------------------------

# ---RESOLUTION OF THE MESH AROUND THE PARTICLE---------
particle_maxh = 0.01
global_maxh = 0.3*max(W,H)
# ------------------------------------------------------