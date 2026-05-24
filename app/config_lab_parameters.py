'''
The following parameters are derived from standard experimental setups at the
Microfluidics Laboratory in Heidelberg. They are categorized into two groups:

1. FIXED CONSTRAINTS: Parameters specific to the biological system (e.g., fluid
   properties and cell dimensions). These should remain constant to ensure
   biological accuracy for the specific cell types modeled.

2. ADAPTABLE ENGINEERING PARAMETERS: Geometric and operational values serving as
   a realistic baseline. These are intended to be flexible, provided that the
   resulting flow conditions remain within the experimentally viable regime
   (specifically regarding the Reynolds number).
'''

# ---FIXED BIOLOGICAL PARAMETERS (ACTIVE VS INACTIVE T-CELLS)-----------
a = 5e-6                    # radius of the cell [m]
rho = 998                   # density [kg/m^3]
mu  = 1.002e-3              # dyn. viscosity [Pa·s]
# ----------------------------------------------------------------------

# ---GEOMETRIC PARAMETERS (ADAPTABLE BASELINE)--------------------------
H = 1.20e-4                 # duct height [m]  # H = 240e-6
W = H                       # duct width  [m]
R = 2e-2                    # bend radius [m] # 23*(H/2)
# ----------------------------------------------------------------------

# ---OPERATIONAL PARAMETER (CONSTRAINED BY EXPERIMENTAL FEASIBILITY)----
# The flow rate is chosen to target the regime 90 <= Re <= 100.
# This range is recommended by experimental partners for optimal cell handling.
Q = 9.54e-8                 # volumetric flow rate [m^3/s] (Re approx. 83)
# ----------------------------------------------------------------------

# ---GRID RESOLUTION----------------------------------------------------
N_grid = 20
particle_maxh_rel = 0.2
global_maxh_rel = 0.2
eps_rel = 0.2
# ----------------------------------------------------------------------
