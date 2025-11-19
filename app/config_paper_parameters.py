import numpy as np

# ---DIMENSIONLESS CELL SIZE---------------------------
a = 0.05
# -----------------------------------------------------

# ---DIMENSIONLESS ENGINEERING SETUP-------------------
R = 160
W = 2.0
H = 2.0
Q = 20
# -----------------------------------------------------

# ---DERIVED QUANTITIES--------------------------------
D_h = (2*H*W)/(H + W)
U_c = Q/(H*W)

Re = (D_h*U_c)
Re_p = Re*(a/D_h)**2
De = Re*np.sqrt(D_h/(2*R))
kappa = (D_h**4)/(4*(a**3)*R)
# ------------------------------------------------------

# ---LENGTH OF THE DUCT SECTION WE'RE LOOKING AT--------
L = 3
# ------------------------------------------------------

# ---RESOLUTION OF THE GRID OF PARTICLE FORCES----------
N_r = 14
N_z = 14
# ------------------------------------------------------

# ---RESOLUTION OF THE MESH AROUND THE PARTICLE---------
particle_maxh = 0.01
global_maxh = 0.05*max(W,H)
# ------------------------------------------------------