from mpi4py import MPI
import numpy as np


# --- FLUID PARAMETERS ------------------------------------------------
rho_phys = 998              # density of the fluid (= density of the particle in our model) [kg/m^3]
mu_phys  = 1.002e-3         # dyn. viscosity [Pa·s]
# ---------------------------------------------------------------------


# --- GEOMETRIC PARAMETERS --------------------------------------------
H_phys = 1.20e-4            # duct height [m]
W_phys = H_phys             # duct width  [m]
R_phys = 500*(H_phys/2)     # bend radius [m]
# ---------------------------------------------------------------------


# --- OPERATIONAL PARAMETER -------------------------------------------
Q_phys = 7.23e-9            # volumetric flow rate [m^3/s]
# ---------------------------------------------------------------------


# --- PARTICLE SIZE FOR THE CROSS-SECTIONAL FORCE MAP EVALUATION ------
a_phys = 6e-6               # particle-radius [m]
# ---------------------------------------------------------------------


# --- PARTICLE SIZES FOR THE BIFURCATION DIAGRAM ----------------------
a_start_phys = 0.01*(H_phys/2)         # smallest particle-radius to be evaluated [m]
a_end_phys = 0.2*(H_phys/2)          # largest particle-radius to be evaluated [m]
a_stepsize_phys = 1/4*10e-6 # stepsize of the sweep over the particle-radius [m]
# ---------------------------------------------------------------------


# --- PARTICLE-SIZE OF THE POPULATION TO SEPARATE VIA SHAPE OPTIMIZATION ----------------
a_minus_phys = 4e-6         # particle-radius of the smaller population [m]
a_plus_phys = 6e-6          # particle-radius of the larger population [m]
a_target_phys = 1/2 * (a_minus_phys + a_plus_phys) # particle-radius at which the bifurcation should occur after the deformation [m]
# ---------------------------------------------------------------------


# ---NUMERICAL RESOLUTION----------------------------------------------
particle_maxh_rel = 0.1    # target edge length near the particle surface, relative to a (absolute: particle_maxh_rel * a)
global_maxh_rel   = 0.2     # target edge length in the bulk fluid, relative to min(H, W) (absolute: global_maxh_rel * min(H, W))
L_rel             = 4       # length of the 3D channel section, relative to max(H, W) (absolute: L_rel * max(H, W))
# ----------------------------------------------------------------------


# ---EQUILIBRIUM SEARCH ------------------------------------------------
N_grid  = 20                # number of sample points per axis on the (r, z) force grid (total: N_grid x N_grid evaluations)
eps_rel = 0.2               # wall-safety margin for the grid bounds, relative to a; shrinks the sampled (r, z) box by (a + eps_rel*a) on each side
# ----------------------------------------------------------------------


def nondimensionalisation(R_phys, H_phys, W_phys, Q_phys, rho_phys, mu_phys,
                          a_phys, a_start_phys, a_end_phys, a_stepsize_phys,
                          a_minus_phys, a_plus_phys, a_target_phys):

    L_c = H_phys / 2

    U_c = Q_phys / (W_phys * H_phys)

    R = R_phys / L_c
    H = H_phys / L_c
    W = W_phys / L_c
    Re = (rho_phys * U_c * L_c) / mu_phys

    a = a_phys / L_c
    a_start    = a_start_phys    / L_c
    a_end      = a_end_phys      / L_c
    a_stepsize = a_stepsize_phys / L_c
    a_values   = np.round(np.arange(a_start, a_end, a_stepsize), 7)

    a_minus  = a_minus_phys  / L_c
    a_plus   = a_plus_phys   / L_c
    a_target = a_target_phys / L_c

    return R, H, W, L_c, U_c, Re, a, a_start, a_end, a_stepsize, a_values, a_minus, a_plus, a_target


R, H, W, L_c, U_c, Re, a, a_start, a_end, a_stepsize, a_values, a_minus, a_plus, a_target = nondimensionalisation(
    R_phys, H_phys, W_phys, Q_phys, rho_phys, mu_phys, a_phys, a_start_phys, a_end_phys, a_stepsize_phys, a_minus_phys,
    a_plus_phys, a_target_phys)


def print_problem_setup():

    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    print("=" * 60)
    print("  PROBLEM SETUP")
    print("=" * 60)

    print("\n  Fluid + geometry (SI):")
    print(f"    rho_phys = {rho_phys:.3e} kg/m^3  (fluid density)")
    print(f"    mu_phys  = {mu_phys:.3e} Pa*s     (dynamic viscosity)")
    print(f"    H_phys   = {H_phys:.3e} m         (duct height)")
    print(f"    W_phys   = {W_phys:.3e} m         (duct width)")
    print(f"    R_phys   = {R_phys:.3e} m         (bend radius)")
    print(f"    Q_phys   = {Q_phys:.3e} m^3/s     (volumetric flow rate)")

    print("\n  Particle-size inputs (SI → nondim):")
    print(f"    [force map]      a_phys          = {a_phys:.3e} m"
          f"   →  a        = {a:.4f}")
    print(f"    [bif diagram]    a_start_phys    = {a_start_phys:.3e} m"
          f"   →  a_start   = {a_start:.4f}")
    print(f"                     a_end_phys      = {a_end_phys:.3e} m"
          f"   →  a_end     = {a_end:.4f}")
    print(f"                     a_stepsize_phys = {a_stepsize_phys:.3e} m"
          f"   →  a_stepsize= {a_stepsize:.4f}")
    print(f"                     → a_values ({len(a_values)} pts) "
          f"= {a_values.tolist()}")
    print(f"    [shape opt]      a_minus_phys    = {a_minus_phys:.3e} m"
          f"   →  a_minus   = {a_minus:.4f}")
    print(f"                     a_plus_phys     = {a_plus_phys:.3e} m"
          f"   →  a_plus    = {a_plus:.4f}")
    print(f"                     a_target_phys   = {a_target_phys:.3e} m"
          f"   →  a_target  = {a_target:.4f}")

    print("\n  Other dimensionless quantities:")
    print(f"    R   = {R:.4f}")
    print(f"    H   = {H:.4f}")
    print(f"    W   = {W:.4f}")
    print(f"    Re  = {Re:.2f}")
    print(f"    L_c = {L_c:.3e} m     (characteristic length, H_phys/2)")
    print(f"    U_c = {U_c:.3e} m/s   (characteristic velocity, Q_phys/(W_phys*H_phys))")

    print("\n  Numerical resolution:")
    print(f"    particle_maxh_rel = {particle_maxh_rel}")
    print(f"    global_maxh_rel   = {global_maxh_rel}")
    print(f"    L_rel             = {L_rel}")

    print("\n  Equilibrium search grid:")
    print(f"    N_grid  = {N_grid}")
    print(f"    eps_rel = {eps_rel}")

    print("=" * 60, flush=True)


print_problem_setup()