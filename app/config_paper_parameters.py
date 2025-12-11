

# ---REALISTIC PHYSICAL ASSUMPTIONS---------------------
H = 240e-6                  # duct height [m]
rho = 998                   # density [kg/m^3]
mu  = 10.02e-4              # dyn. viscosity [Pa·s]
# ------------------------------------------------------

# ---LENGTH RATIOS FROM THE PAPER-----------------------
a = 0.05*H / 2               # radius of the smaller cell [m]
W = H                        # duct width  [m]
R = 160*H / 2                # bend radius [m]
# ------------------------------------------------------

# ---VOLUMETRIC FLOW RATE (CHOSEN TO MATCH ASSUMPTIONS FROM THE PAPER)
Q = 2e-9                    # volumetric flow rate [m^3/s]
# ------------------------------------------------------

kappa = (H**4)/(4*(a**3)*R)
# print(kappa)  # = 200