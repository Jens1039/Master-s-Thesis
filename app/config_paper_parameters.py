'''
The following parameters are defined using realistic physical and engineering units
to ensure compatibility with experimental setups utilized at the Microfluidics
Laboratory in Heidelberg.
The values have been specifically selected so that
their derived non-dimensional quantities align with the parameters specified in the reference
paper. This approach ensures the physical realizability of the simulation setup
while maintaining direct numerical comparability with the published results.
'''

# ---REALISTIC PHYSICAL ASSUMPTIONS---------------------
H = 240e-6                  # duct height [m]
rho = 998                   # density [kg/m^3]
mu  = 10.02e-4              # dyn. viscosity [Pa·s]
# ------------------------------------------------------

# ---LENGTH RATIOS FROM THE PAPER-----------------------
a = 0.05 * (H/2)            # radius of the cell/particle [m]
W = 1*H                       # duct width  [m]
R = 160 * (H/2)             # bend radius [m]
# ------------------------------------------------------

# ---VOLUMETRIC FLOW RATE (Chosen to match a low Reynolds number (According to the paper results hold up to Re = O(10)))
Q = 2*2.40961923848e-10                   # volumetric flow rate [m^3/s]
# ------------------------------------------------------