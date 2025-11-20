import numpy as np

# ---DIMENSIONLESS CELL SIZE---------------------------
a = 0.15
# -----------------------------------------------------

# ---DIMENSIONLESS ENGINEERING SETUP-------------------
R = 160
W = 2.0
H = 2.0
Q = 10 # We are setting this, so that our flow Reynolds number = 5 (consistent with the statement in the paper "results hold up to O(10)"
# -----------------------------------------------------

# ---DERIVED QUANTITIES--------------------------------
D_h = (2*H*W)/(H + W)
U_c = Q/(H*W)
Re = (D_h*U_c)
De = Re*np.sqrt(D_h/(2*R))
kappa = (D_h**4)/(4*(a**3)*R)
# ------------------------------------------------------

# ---LENGTH OF THE DUCT SECTION WE'RE LOOKING AT--------
L = 4 * max(H, W)
# ------------------------------------------------------

# ---RESOLUTION OF THE GRID OF PARTICLE FORCES----------
N_r = 10
N_z = 10
# ------------------------------------------------------

# ---RESOLUTION OF THE MESH AROUND THE PARTICLE---------
particle_maxh = 0.2 * a
global_maxh = 0.2*max(W,H)
# ------------------------------------------------------

# ------------------------------------------------------
nproc = 10