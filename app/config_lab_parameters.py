'''
The following parameters are derived from standard experimental setups at the
Microfluidics Laboratory in Heidelberg. They are categorized into two groups:

1. FIXED CONSTRAINTS: Parameters specific to the biological system (e.g., fluid
   properties and T-cell dimensions). These should remain constant to ensure
   biological accuracy for the specific cell types modeled.

2. ADAPTABLE ENGINEERING PARAMETERS: Geometric and operational values serving as
   a realistic baseline. These are intended to be flexible, provided that the
   resulting flow conditions remain within the experimentally viable regime
   (specifically regarding the Reynolds number).
'''

# ---FIXED BIOLOGICAL PARAMETERS (ACTIVE VS INACTIVE T-CELLS)-----------
# a   = 8e-6                  # radius of the smaller cell [m]
a = 12e-6                 # radius of the larger cell [m]
rho = 998                   # density [kg/m^3]
mu  = 10.02e-4              # dyn. viscosity [Pa·s]
# ----------------------------------------------------------------------

# ---GEOMETRIC PARAMETERS (ADAPTABLE BASELINE)--------------------------
H = 240e-6                  # duct height [m]
W = H                       # duct width  [m]
R = 2800e-6                 # bend radius [m]
# ----------------------------------------------------------------------

# ---OPERATIONAL PARAMETER (CONSTRAINED BY EXPERIMENTAL FEASIBILITY)----
# The flow rate is chosen to target the regime 50 <= Re <= 100.
# This range is recommended by experimental partners for optimal cell handling.
Q = 2e-8                    # volumetric flow rate [m^3/s] (Re approx. 83)
# ----------------------------------------------------------------------