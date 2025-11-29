import numpy as np

# ---INPUT---------------------------------------------
d_1 = 8e-6          # diameter of the smaller cell [m]
d_2 = 12e-6         # diameter of the bigger cell [m]
# ------------------------------------------------------

# Notice: right now, we're choosing standard values used in most microfluidic chips
# ---VARIABLE ASSUMPTION--------------------------------
R = 2800e-6         # bend radius [m]
H = 240e-6 # 120e-6          # duct height [m]
W = 500e-6                   # duct width  [m]
Q = 2e-8   # 3e-8            # volumetric flow rate [m^3/s]
# ------------------------------------------------------

# ---FIXED ASSUMPTIONS----------------------------------
rho = 998           # density [kg/m^3]
mu  = 10.02e-4      # dyn. viscosity [Pa·s]
# ------------------------------------------------------

# For numerical conditioning and comparability reasons, all our computations should be done in the nondimensional settings

def nondimensionalize(d_1, d_2, R, H, W, Q, rho, mu, print_values=True):

    # Characteristic length is the hydraulic diameter D_h
    L_c = (2*H*W)/(W + H)

    # Characteristic velocity is the velocity of a fluid particle without perturbations (based on the input flow rate)
    U_c = Q/(W*H)

    # Characteristic time is the time, that it takes for a fluid particle to cross the hydraulic diameter
    T_ref = L_c/U_c

    # typical reference pressure in fluid dynamics
    P_ref = rho*(U_c**2)

    # nondimensionalize every input variable
    a_1_nd = d_1/(2*L_c)
    a_2_nd = d_2/(2*L_c)
    R_nd = R/L_c
    H_nd = H/L_c
    W_nd = W/L_c

    # we nondimensionalize the Navier-Stokes equation (and its reduced forms) in the classical way, therefore mu and rho are contained in Re
    Re = (rho*U_c*L_c)/mu

    # in our case Re_p to is a second important quantity to determine whether we can use a perturbation expansion to calculate the perturbed flow
    Re_p = Re*(a_2_nd)**2

    # the dean number is important to determine the secondary fluid regime
    De = Re * np.sqrt(L_c / (2 * R))

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

a_1, a, H, W, R, Re, Re_p = nondimensionalize(d_1, d_2, R, H, W, Q, rho, mu)

'''
H =  1.0000000000000002
W =  1.0000000000000002
R =  11.666666666666668
a_1 =  0.016666666666666666
a_2 =  0.025
Re =  83.00066533599465
Re_p =  0.05187541583499667
De =  17.18277016526121
'''

# ---LENGTH OF THE DUCT SECTION WE'RE LOOKING AT--------
L = 4*max(W, H)
# ------------------------------------------------------

# ---RESOLUTION OF THE GRID OF PARTICLE FORCES----------
N_r = 10
N_z = 10
# ------------------------------------------------------

# ---RESOLUTION OF THE MESH AROUND THE PARTICLE---------
particle_maxh = 0.02
global_maxh = 0.3*max(W,H)
# ------------------------------------------------------



















