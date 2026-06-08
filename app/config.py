# --- FIXED BIOLOGICAL PARAMETERS -----------
a = 6e-6                    # radius of the cell [m]
rho = 998                   # density of the fluid (= density of the particle in our model) [kg/m^3]
mu  = 1.002e-3              # dyn. viscosity [Pa·s]
# ----------------------------------------------------------------------

# --- GEOMETRIC PARAMETERS (ADAPTABLE BASELINE) --------------------------
H = 1.20e-4                 # duct height [m]
W = H                       # duct width  [m]
R = 500*(H/2)               # bend radius [m]
# ----------------------------------------------------------------------

# ---OPERATIONAL PARAMETER----
Q = 7.23e-9                 # volumetric flow rate [m^3/s]
# ----------------------------------------------------------------------

# ---NUMERICAL RESOLUTION----------------------------------------------------
N_grid = 20                 # only relevant for "find_equilbrium_points.py" - Hier fehlt noch eine knappe beschreibung was das macht
particle_maxh_rel = 0.1     # Hier fehlt noch eine knappe beschreibung was das macht
global_maxh_rel = 0.2       # Hier fehlt noch eine knappe beschreibung was das macht
eps_rel = 0.2               # Hier fehlt noch eine knappe beschreibung was das macht - sollte ich vlt rausnehmen aus den configs?
L_rel = 4                   # Hier fehlt noch eine knappe beschreibung was das macht - sollte ich vlt rausnehmen aus den configs?
# ----------------------------------------------------------------------
