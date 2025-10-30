from preprocess_and_nondimensionalize import find_optimal_H_and_Q, Nondimensionalizer


# ---INPUT---------------------------------------------
d_1 = 8     # diameter of the smaller cell [µm]
d_2 = 12    # diameter of the bigger cell [µm]
tau_max = 20.0     # Pa
# ------------------------------------------------------

# ---VARIABLE ASSUMPTION--------------------------------
R = 500e-6 # bend radius [m]
# ------------------------------------------------------

# ---FIXED ASSUMPTIONS----------------------------------
rho = 1000.0           # density [kg/m^3]
mu  = 1.0e-3           # dyn. viscosity of water [Pa·s]
# ------------------------------------------------------





if __name__ == "__main__":



