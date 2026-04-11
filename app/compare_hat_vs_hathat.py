"""
Compare forces at the SAME physical point (r=0.61, z=0, a=0.135) using:
  1. Reference:  perturbed_flow.py in hat_hat coords (known correct)
  2. Hat_hat:    perturbed_flow_return_UFL.py in hat_hat coords (already verified)
  3. Hat:        perturbed_flow_return_UFL.py in hat coords (the suspect)

If the hat path gives different forces (after scaling), it explains why
the Newton solver finds r≈0.96 instead of r≈0.6.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
from firedrake.adjoint import stop_annotating
from pyadjoint import set_working_tape, Tape

from nondimensionalization import first_nondimensionalisation, second_nondimensionalisation
from background_flow import background_flow, build_3d_background_flow
from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole
from config_paper_parameters import R, H, W, Q, rho, mu, particle_maxh_rel, global_maxh_rel

from perturbed_flow import perturbed_flow as pf_reference_class
from perturbed_flow_return_UFL import perturbed_flow_differentiable as pf_ufl_class


if __name__ == "__main__":

    r_off_hat = 0.61
    z_off_hat = 0.0
    a_hat = 0.135

    # --- First nondimensionalization ---
    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(
        R, H, W, Q, rho, mu, print_values=True)

    # --- 2D background flow ---
    bg_orig = background_flow(R_hat, H_hat, W_hat, Re)
    G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat = bg_orig.solve_2D_background_flow()

    bg_ufl = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_hat_ufl, U_m_hat_ufl, u_bar_2d_hat_ufl, p_bar_tilde_2d_hat_ufl = bg_ufl.solve_2D_background_flow()

    print(f"\nU_m_hat = {U_m_hat:.10f}")
    print(f"G_hat   = {G_hat:.10f}")

    # --- Second nondimensionalization (for hat_hat paths) ---
    a_physical = a_hat * L_c

    (R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, U_c_p,
     u_2d_hh, p_2d_hh, Re_p) = second_nondimensionalisation(
        R_hat, H_hat, W_hat, a_physical, L_c, U_c, G_hat, Re,
        u_bar_2d_hat, p_bar_tilde_2d_hat, U_m_hat, print_values=True)

    (_, _, _, _, G_hh_ufl, _, _,
     u_2d_hh_ufl, p_2d_hh_ufl, Re_p_ufl) = second_nondimensionalisation(
        R_hat, H_hat, W_hat, a_physical, L_c, U_c, G_hat_ufl, Re,
        u_bar_2d_hat_ufl, p_bar_tilde_2d_hat_ufl, U_m_hat_ufl, print_values=False)

    scale = L_c / L_c_p
    r_off_hh = r_off_hat * scale
    z_off_hh = z_off_hat * scale
    L_hh = 4 * max(H_hh, W_hh)

    # Expected scaling: F_hat = (U_c_p * L_c_p) / (U_c * L_c) * F_hh
    force_scaling = (U_c_p * L_c_p) / (U_c * L_c)
    print(f"\nForce scaling factor = {force_scaling:.10e}")

    # ================================================================
    #  1. REFERENCE: perturbed_flow.py in hat_hat coords
    # ================================================================
    print("\n" + "=" * 70)
    print("  1. REFERENCE: perturbed_flow.py (hat_hat coords)")
    print("=" * 70)

    mesh3d_hh, tags_hh = make_curved_channel_section_with_spherical_hole(
        R_hh, H_hh, W_hh, L_hh, a_hh,
        particle_maxh_rel * a_hh,
        global_maxh_rel * min(H_hh, W_hh),
        r_off_hh, z_off_hh)

    u_bar_3d_orig, p_bar_3d_orig = build_3d_background_flow(
        R_hh, H_hh, W_hh, G_hh,
        mesh3d_hh, tags_hh, u_2d_hh, p_2d_hh)

    pf_ref = pf_reference_class(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
        mesh3d_hh, tags_hh, u_bar_3d_orig, p_bar_3d_orig)

    F_ref_x, F_ref_z = pf_ref.F_p()
    F_ref = np.array([float(F_ref_x), float(F_ref_z)])
    print(f"  F_ref = [{F_ref[0]:.10e}, {F_ref[1]:.10e}]")

    # ================================================================
    #  2. UFL in hat_hat coords (same mesh as reference)
    # ================================================================
    print("\n" + "=" * 70)
    print("  2. UFL REWRITE: perturbed_flow_return_UFL.py (hat_hat coords)")
    print("=" * 70)

    V_def_hh = VectorFunctionSpace(mesh3d_hh, "CG", 1)
    with stop_annotating():
        X_ref_hh = Function(V_def_hh, name="X_ref")
        X_ref_hh.interpolate(SpatialCoordinate(mesh3d_hh))

    xi_hh = Function(V_def_hh, name="xi")
    xi_hh.assign(0)

    u_bar_3d_ufl_hh, p_bar_3d_ufl_hh, u_cyl_3d_hh = build_3d_background_flow_differentiable(
        R_hh, H_hh, W_hh, G_hh_ufl,
        mesh3d_hh, tags_hh, u_2d_hh_ufl, p_2d_hh_ufl,
        X_ref=X_ref_hh, xi=xi_hh)

    set_working_tape(Tape())
    pf_ufl_hh = pf_ufl_class(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p_ufl,
        mesh3d_hh, tags_hh, u_bar_3d_ufl_hh, p_bar_3d_ufl_hh,
        X_ref_hh, xi_hh, u_cyl_3d_hh)

    F_ufl_hh_x, F_ufl_hh_z = pf_ufl_hh.F_p()
    F_ufl_hh = np.array([float(F_ufl_hh_x), float(F_ufl_hh_z)])
    print(f"  F_ufl_hh = [{F_ufl_hh[0]:.10e}, {F_ufl_hh[1]:.10e}]")

    # ================================================================
    #  3. UFL in hat coords (THE SUSPECT)
    # ================================================================
    print("\n" + "=" * 70)
    print("  3. UFL REWRITE: perturbed_flow_return_UFL.py (hat coords)")
    print("=" * 70)

    L_hat = 4 * max(H_hat, W_hat)
    Re_p_hat = Re  # In hat coords, use channel Re (not particle Re_p)

    mesh3d_hat, tags_hat = make_curved_channel_section_with_spherical_hole(
        R_hat, H_hat, W_hat, L_hat, a_hat,
        particle_maxh_rel * a_hat,
        global_maxh_rel * min(H_hat, W_hat),
        r_off_hat, z_off_hat)

    V_def_hat = VectorFunctionSpace(mesh3d_hat, "CG", 1)
    with stop_annotating():
        X_ref_hat = Function(V_def_hat, name="X_ref")
        X_ref_hat.interpolate(SpatialCoordinate(mesh3d_hat))

    xi_hat = Function(V_def_hat, name="xi")
    xi_hat.assign(0)

    u_bar_3d_ufl_hat, p_bar_3d_ufl_hat, u_cyl_3d_hat = build_3d_background_flow_differentiable(
        R_hat, H_hat, W_hat, G_hat_ufl,
        mesh3d_hat, tags_hat, u_bar_2d_hat_ufl, p_bar_tilde_2d_hat_ufl,
        X_ref=X_ref_hat, xi=xi_hat)

    set_working_tape(Tape())
    pf_ufl_hat = pf_ufl_class(
        R_hat, H_hat, W_hat, L_hat, a_hat, Re_p_hat,
        mesh3d_hat, tags_hat, u_bar_3d_ufl_hat, p_bar_3d_ufl_hat,
        X_ref_hat, xi_hat, u_cyl_3d_hat)

    F_ufl_hat_x, F_ufl_hat_z = pf_ufl_hat.F_p()
    F_ufl_hat = np.array([float(F_ufl_hat_x), float(F_ufl_hat_z)])
    print(f"  F_hat    = [{F_ufl_hat[0]:.10e}, {F_ufl_hat[1]:.10e}]")

    # ================================================================
    #  COMPARISON
    # ================================================================
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    # 1 vs 2: should match (already verified)
    rel_12 = np.abs(F_ufl_hh - F_ref) / (np.abs(F_ref) + 1e-30)
    print(f"\n  1 vs 2 (ref vs UFL, hat_hat): rel_diff = [{rel_12[0]:.4e}, {rel_12[1]:.4e}]")

    # 1 vs 3: hat path scaled to hat_hat units
    F_hat_rescaled = F_ufl_hat / force_scaling
    rel_13 = np.abs(F_hat_rescaled - F_ref) / (np.abs(F_ref) + 1e-30)
    print(f"\n  1 vs 3 (ref vs hat, after scaling):")
    print(f"    F_ref         = [{F_ref[0]:.10e}, {F_ref[1]:.10e}]")
    print(f"    F_hat/scaling = [{F_hat_rescaled[0]:.10e}, {F_hat_rescaled[1]:.10e}]")
    print(f"    rel_diff      = [{rel_13[0]:.4e}, {rel_13[1]:.4e}]")

    if abs(F_ref[0]) > 1e-20:
        actual_ratio = F_ufl_hat[0] / F_ref[0]
        print(f"\n  Actual ratio F_hat_x / F_ref_x = {actual_ratio:.10e}")
        print(f"  Expected ratio (scaling)       = {force_scaling:.10e}")
        print(f"  Ratio / expected               = {actual_ratio / force_scaling:.10e}")

    tol = 0.05
    if np.all(rel_13 < tol):
        print(f"\n  [PASS] Hat-coord path agrees within {tol*100:.0f}%")
        print(f"         -> Problem is NOT in force computation, look at Newton/moving-mesh.")
    else:
        print(f"\n  [FAIL] Hat-coord path disagrees by more than {tol*100:.0f}%!")
        print(f"         -> Force computation in hat coordinates is WRONG.")
        print(f"         -> This explains why Newton finds r≈0.96 instead of r≈0.6.")
