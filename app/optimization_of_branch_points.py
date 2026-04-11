import os
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import numpy as np
import math
from copy import deepcopy

from firedrake import *
from firedrake.adjoint import (
    stop_annotating, continue_annotation,
    ReducedFunctional, Control,
)
from pyadjoint import set_working_tape, get_working_tape, Tape

from firedrake import JacobianDeterminant

from nondimensionalization import first_nondimensionalisation, second_nondimensionalisation
from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable
from perturbed_flow_return_UFL import perturbed_flow_differentiable
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole
from config_paper_parameters import R, H, W, Q, rho, mu, particle_maxh_rel, global_maxh_rel


class MeshFlippedError(Exception):
    """Raised when mesh deformation produces inverted elements."""
    pass


def check_mesh_quality(mesh3d, ref_signs=None):
    """Check for flipped (inverted) elements after mesh deformation.

    Compares the sign of each element's Jacobian determinant against
    *ref_signs* (the signs from the undeformed mesh).  An element is
    flipped when its sign changes, not simply when det(J) < 0 (gmsh
    vertex ordering can make the reference determinant negative).

    Returns (min_abs_det, signs) where *signs* is a numpy array of
    per-element signs (+1/-1) that can be passed as *ref_signs* on
    subsequent calls.

    Raises MeshFlippedError if any element changed sign.
    """
    DG0 = FunctionSpace(mesh3d, "DG", 0)
    jac_det = Function(DG0)
    jac_det.interpolate(JacobianDeterminant(mesh3d))
    det_vals = jac_det.dat.data_ro
    signs = np.sign(det_vals)
    min_abs = float(np.abs(det_vals).min())

    if ref_signs is not None and not np.array_equal(signs, ref_signs):
        n_flipped = int(np.sum(signs != ref_signs))
        raise MeshFlippedError(
            f"Mesh has {n_flipped} flipped element(s): min(|det(J)|) = {min_abs:.4e}")

    if min_abs == 0.0:
        raise MeshFlippedError("Mesh has degenerate element(s): det(J) = 0")

    return min_abs, signs


def setup_moving_mesh(r_off_hat_init, z_off_hat_init, a_hat, R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat):

    a_physical = a_hat * L_c

    (R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, U_c_p, u_2d_hh, p_2d_hh, Re_p) = second_nondimensionalisation(R_hat, H_hat, W_hat, a_physical, L_c, U_c, G_hat, Re,
                                                                            u_bar_2d_hat, p_bar_tilde_2d_hat, U_m_hat, print_values=False)

    scale = L_c / L_c_p
    r_off_hh_init = r_off_hat_init * scale
    z_off_hh_init = z_off_hat_init * scale
    L_hh = 4 * max(H_hh, W_hh)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R_hh, H_hh, W_hh, L_hh, a_hh, particle_maxh_rel * a_hh,
                                                                   global_maxh_rel * min(H_hh, W_hh), r_off_hh_init, z_off_hh_init)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():

        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    dist = sqrt((X_ref[0] - cx)**2 + (X_ref[1] - cy)**2 + (X_ref[2] - cz)**2)
    r_cut = Constant(0.5 * min(H_hh, W_hh))
    bump = max_value(Constant(0.0), 1.0 - dist / r_cut)

    theta_half = tags["theta"] / 2.0

    _, ref_signs = check_mesh_quality(mesh3d)

    print(f"  [setup] Mesh generated: R_hh={R_hh:.2f}, H_hh={H_hh:.2f}, "
          f"W_hh={W_hh:.2f}, a_hh={a_hh:.2f}")
    print(f"  [setup] scale={scale:.4f}, r_off_hh_init={r_off_hh_init:.4f}, "
          f"z_off_hh_init={z_off_hh_init:.4f}")

    return {
        'mesh3d': mesh3d, 'tags': tags,
        'X_ref': X_ref, 'V_def': V_def,
        'bump': bump,
        'cos_th': math.cos(theta_half),
        'sin_th': math.sin(theta_half),
        'R_hh': R_hh, 'H_hh': H_hh, 'W_hh': W_hh,
        'L_hh': L_hh, 'a_hh': a_hh,
        'G_hh': G_hh, 'Re_p': Re_p,
        'u_2d_hh': u_2d_hh, 'p_2d_hh': p_2d_hh,
        'scale': scale,
        'ref_signs': ref_signs,
    }


def evaluate_forces(delta_r_hh, delta_z_hh, mesh_data, compute_jacobian=True):

    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']

    R_space = FunctionSpace(mesh3d, "R", 0)
    delta_r = Function(R_space, name="delta_r").assign(delta_r_hh)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hh)

    xi = Function(md['V_def'], name="xi")
    xi.interpolate(as_vector([
        delta_r * md['cos_th'] * md['bump'],
        delta_r * md['sin_th'] * md['bump'],
        delta_z * md['bump'],
    ]))
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hh'], md['H_hh'], md['W_hh'], md['G_hh'],
        mesh3d, md['tags'], md['u_2d_hh'], md['p_2d_hh'],
        X_ref=md['X_ref'], xi=xi)

    pf = perturbed_flow_differentiable(
        md['R_hh'], md['H_hh'], md['W_hh'], md['L_hh'],
        md['a_hh'], md['Re_p'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    if not compute_jacobian:
        stop_annotating()
        get_working_tape().clear_tape()
        return F

    '''
    with stop_annotating():
        eps = 1e-5
        F_rp = evaluate_forces(delta_r_hh + eps, delta_z_hh, mesh_data, compute_jacobian=False)
        F_rm = evaluate_forces(delta_r_hh - eps, delta_z_hh, mesh_data, compute_jacobian=False)
        F_zp = evaluate_forces(delta_r_hh, delta_z_hh + eps, mesh_data, compute_jacobian=False)
        F_zm = evaluate_forces(delta_r_hh, delta_z_hh - eps, mesh_data, compute_jacobian=False)
    
        J = np.zeros((2, 2))
        J[:, 0] = (F_rp - F_rm) / (2 * eps)
        J[:, 1] = (F_zp - F_zm) / (2 * eps)
    
        return F, J
    '''

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    Jhat_x = ReducedFunctional(F_p_x, [c_r, c_z])
    Jhat_z = ReducedFunctional(F_p_z, [c_r, c_z])

    dFx = Jhat_x.derivative()
    dFz = Jhat_z.derivative()

    J_00 = float(dFx[0].dat.data_ro[0])
    J_01 = float(dFx[1].dat.data_ro[0])

    J_10 = float(dFz[0].dat.data_ro[0])
    J_11 = float(dFz[1].dat.data_ro[0])

    J = np.array([[J_00, J_01],
                  [J_10, J_11]])

    stop_annotating()
    get_working_tape().clear_tape()
    return F, J


def evaluate_forces_tlm(delta_r_hh, delta_z_hh, mesh_data, phi):
    """Compute F and J@phi via forward-mode AD (TLM).

    One PDE solve + one forward AD pass, instead of one PDE solve + two
    reverse AD passes that would be needed to build the full Jacobian.
    """
    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']

    R_space = FunctionSpace(mesh3d, "R", 0)
    delta_r = Function(R_space, name="delta_r").assign(delta_r_hh)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hh)

    xi = Function(md['V_def'], name="xi")
    xi.interpolate(as_vector([
        delta_r * md['cos_th'] * md['bump'],
        delta_r * md['sin_th'] * md['bump'],
        delta_z * md['bump'],
    ]))
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hh'], md['H_hh'], md['W_hh'], md['G_hh'],
        mesh3d, md['tags'], md['u_2d_hh'], md['p_2d_hh'],
        X_ref=md['X_ref'], xi=xi)

    pf = perturbed_flow_differentiable(
        md['R_hh'], md['H_hh'], md['W_hh'], md['L_hh'],
        md['a_hh'], md['Re_p'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    # Set TLM seeds on the controls
    delta_r.block_variable.tlm_value = Function(R_space).assign(phi[0])
    delta_z.block_variable.tlm_value = Function(R_space).assign(phi[1])

    # Forward propagation through the tape
    tape = get_working_tape()
    tape.evaluate_tlm()

    # Read TLM output: J @ phi
    Jphi = np.array([
        float(F_p_x.block_variable.tlm_value),
        float(F_p_z.block_variable.tlm_value),
    ])

    stop_annotating()
    get_working_tape().clear_tape()
    return F, Jphi


def evaluate_forces_and_fd_jacobian(delta_r_hh, delta_z_hh, mesh_data, eps=1e-5):

    F0 = evaluate_forces(delta_r_hh, delta_z_hh, mesh_data, compute_jacobian=False)

    # dF/d(delta_r)
    F_rp = evaluate_forces(delta_r_hh + eps, delta_z_hh, mesh_data, compute_jacobian=False)
    F_rm = evaluate_forces(delta_r_hh - eps, delta_z_hh, mesh_data, compute_jacobian=False)

    # dF/d(delta_z)
    F_zp = evaluate_forces(delta_r_hh, delta_z_hh + eps, mesh_data, compute_jacobian=False)
    F_zm = evaluate_forces(delta_r_hh, delta_z_hh - eps, mesh_data, compute_jacobian=False)

    J_fd = np.zeros((2, 2))
    J_fd[:, 0] = (F_rp - F_rm) / (2 * eps)
    J_fd[:, 1] = (F_zp - F_zm) / (2 * eps)

    return F0, J_fd


def newton_root_refine(r_off_hat_init, z_off_hat_init, a_hat, shared_data, *, tol=1e-10, max_iter=15):

    print("\n" + "=" * 65)
    print("  PHASE 1: Newton Root Refinement (fixed a_hat = {:.6f})".format(a_hat))
    print("=" * 65)

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat = shared_data

    mesh_data = setup_moving_mesh(r_off_hat_init, z_off_hat_init, a_hat, R_hat, H_hat, W_hat, L_c, U_c, Re,
                                  G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat)

    scale = mesh_data['scale']

    # Newton unknowns: displacement in hat_hat coordinates
    delta_r_hh = 0.0
    delta_z_hh = 0.0

    for k in range(max_iter):
        F, J = evaluate_forces(delta_r_hh, delta_z_hh, mesh_data)

        # --- FD-Jacobian-Check ---
        # F_fd, J_fd = evaluate_forces_and_fd_jacobian(delta_r_hh, delta_z_hh, mesh_data, eps=1e-5)

        print(f"  AD  Jacobian:\n    {J}")
        # print(f"  FD  Jacobian:\n    {J_fd}")
        # print(f"  Difference (AD - FD):\n    {J - J_fd}")
        # print(f"  Relative diff: {np.linalg.norm(J - J_fd) / (np.linalg.norm(J_fd) + 1e-30):.4e}")

        dx_ad = np.linalg.solve(J, -F)
        # dx_fd = np.linalg.solve(J_fd, -F)
        print(f"  Newton step (AD):  {dx_ad}")
        # print(f"  Newton step (FD):  {dx_fd}")
        print(f"  F·dx_ad = {np.dot(F, dx_ad):.6e}  (sollte < 0 für Abstieg)")
        # print(f"  F·dx_fd = {np.dot(F, dx_fd):.6e}")

        r_off_hat = r_off_hat_init + delta_r_hh / scale
        z_off_hat = z_off_hat_init + delta_z_hh / scale

        res = np.linalg.norm(F)
        cond = np.linalg.cond(J)
        print(f"  Iter {k:2d} | r_off_hat = {r_off_hat:+.8f}  "
              f"z_off_hat = {z_off_hat:+.8f}"
              f" | |F| = {res:.4e}  cond(J) = {cond:.2e}")

        if res < tol:
            print(f"  -> Converged after {k} iterations.\n")
            return r_off_hat, z_off_hat, True, mesh_data

        # Armijo step size
        alpha = 1
        for ls_step in range(10):
            dr_trial = delta_r_hh + alpha * dx_ad[0]
            dz_trial = delta_z_hh + alpha * dx_ad[1]
            try:
                F_trial = evaluate_forces(dr_trial, dz_trial, mesh_data, compute_jacobian=False)
            except MeshFlippedError:
                print(f"         | line search: flipped elements at alpha={alpha:.4f}, halving")
                alpha *= 0.5
                continue
            if np.linalg.norm(F_trial) < (1.0 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5
        else:
            print(f"Line-Search: alpha = {alpha:.4f} (minimum reached)")

        delta_r_hh += alpha * dx_ad[0]
        delta_z_hh += alpha * dx_ad[1]

    r_off_hat = r_off_hat_init + delta_r_hh / scale
    z_off_hat = z_off_hat_init + delta_z_hh / scale
    print(f"Did not converge after {max_iter} iterations.\n")
    return r_off_hat, z_off_hat, False, mesh_data


def moore_spence(r_off_hat, z_off_hat, a_hat, phi, l_vec, shared_data, *, tol=1e-8, max_iter=20, eps_fd=1e-4):

    print("\n" + "=" * 65)
    print("  PHASE 2: Moore-Spence Bifurcation Detection")
    print("=" * 65)

    phi = phi / np.dot(l_vec, phi)

    for k in range(max_iter):

        F_base, J_base = evaluate_forces(r_off_hat, z_off_hat, a_hat, *shared_data)

        G1 = F_base
        G2 = J_base @ phi
        G3 = np.dot(l_vec, phi) - 1.0

        G = np.concatenate([G1, G2, [G3]])
        res = np.linalg.norm(G)

        sv = np.linalg.svd(J_base, compute_uv=False)

        print(f"  Iter {k:2d} | r_hat = {r_off_hat:+.8f}  "
              f"z_hat = {z_off_hat:+.8f}  a_hat = {a_hat:.8f}")
        print(f"         | |G| = {res:.4e}  |F| = {np.linalg.norm(G1):.4e}"
              f"  |J*phi| = {np.linalg.norm(G2):.4e}"
              f"  sigma_min(J) = {sv.min():.4e}")

        if res < tol:
            print(f"\n  -> Bifurcation point found after {k} iterations!")
            _print_result(r_off_hat, z_off_hat, a_hat, phi, sv,
                          shared_data)
            return r_off_hat, z_off_hat, a_hat, phi, True

        eps_a = eps_fd * max(abs(a_hat), 1e-4)
        F_a_plus, J_a_plus = evaluate_forces(
            r_off_hat, z_off_hat, a_hat + eps_a, *shared_data)

        dF_da = (F_a_plus - F_base) / eps_a
        dJphi_da = (J_a_plus @ phi - J_base @ phi) / eps_a

        eps_r = eps_fd * max(abs(r_off_hat), 1e-4)
        _, J_r_plus = evaluate_forces(
            r_off_hat + eps_r, z_off_hat, a_hat, *shared_data)
        dJphi_dr = (J_r_plus @ phi - J_base @ phi) / eps_r

        eps_z = eps_fd * max(abs(z_off_hat), 1e-4)
        _, J_z_plus = evaluate_forces(
            r_off_hat, z_off_hat + eps_z, a_hat, *shared_data)
        dJphi_dz = (J_z_plus @ phi - J_base @ phi) / eps_z

        # ──────────────────────────────────────────────────
        # (e) Construct 5x5 Jacobian matrix DG
        # ──────────────────────────────────────────────────
        #
        # Column order: [dr, dz, da, dphi_r, dphi_z]
        #
        #        dr          dz          da         dphi_r     dphi_z
        # G1: [ J[0,0]      J[0,1]      dF_da[0]    0          0      ]
        #     [ J[1,0]      J[1,1]      dF_da[1]    0          0      ]
        # G2: [ dJphi_dr[0] dJphi_dz[0] dJphi_da[0] J[0,0]     J[0,1] ]
        #     [ dJphi_dr[1] dJphi_dz[1] dJphi_da[1] J[1,0]     J[1,1] ]
        # G3: [ 0           0           0           l[0]       l[1]   ]

        DG = np.zeros((5, 5))

        # Rows 0-1: dG1/dy
        DG[0:2, 0:2] = J_base
        DG[0:2, 2] = dF_da

        # Rows 2-3: dG2/dy
        DG[2, 0] = dJphi_dr[0]
        DG[3, 0] = dJphi_dr[1]
        DG[2, 1] = dJphi_dz[0]
        DG[3, 1] = dJphi_dz[1]
        DG[2:4, 2] = dJphi_da
        DG[2:4, 3:5] = J_base

        # Row 4: dG3/dy
        DG[4, 3:5] = l_vec

        # ──────────────────────────────────────────────────
        # (f) Solve Newton step: DG * dy = -G
        # ──────────────────────────────────────────────────
        cond_DG = np.linalg.cond(DG)
        print(f"         | cond(DG) = {cond_DG:.2e}")

        if cond_DG > 1e14:
            print("  !! DG extremely poorly conditioned – "
                  "iteration aborted.")
            return r_off_hat, z_off_hat, a_hat, phi, False

        try:
            delta_y = np.linalg.solve(DG, -G)
        except np.linalg.LinAlgError:
            print("  !! DG is singular – iteration aborted.")
            return r_off_hat, z_off_hat, a_hat, phi, False

        # ──────────────────────────────────────────────────
        # (g) Damped Update
        # ──────────────────────────────────────────────────
        # Limit the step in a_hat to max 20% relative change
        max_a_step = 0.2 * max(abs(a_hat), 0.01)
        if abs(delta_y[2]) > max_a_step:
            sf = max_a_step / abs(delta_y[2])
            delta_y *= sf
            print(f"         | Step damped (factor {sf:.3f})")

        r_off_hat += delta_y[0]
        z_off_hat += delta_y[1]
        a_hat += delta_y[2]
        phi += delta_y[3:5]

        if a_hat <= 0:
            a_hat = 1e-4
            print("  !! a_hat set to minimum value")

    print(f"  Did not converge after {max_iter} iterations.\n")
    return r_off_hat, z_off_hat, a_hat, phi, False


def estimate_null_vector(J):

    U, S, Vt = np.linalg.svd(J)
    print(f"  Singular values of J: sigma_1 = {S[0]:.6e}, "
          f"sigma_2 = {S[1]:.6e}")
    print(f"  sigma_min/sigma_max = {S[1]/S[0]:.6e}")
    return Vt[-1, :]


def _print_result(r_off_hat, z_off_hat, a_hat, phi, sv, shared_data):

    R_hat, H_hat, W_hat, L_c, U_c, Re = shared_data[:6]
    a_physical = a_hat * L_c
    L_c_p = a_physical
    s = L_c / L_c_p

    print(f"\n  Result in hat coordinates:")
    print(f"    r_off_hat  = {r_off_hat:.10f}")
    print(f"    z_off_hat  = {z_off_hat:.10f}")
    print(f"    a_hat      = {a_hat:.10f}")
    print(f"    phi        = ({phi[0]:.8f}, {phi[1]:.8f})")
    print(f"    sigma_min  = {sv.min():.6e}")

    print(f"\n  In hat_hat coordinates:")
    print(f"    r_off_hh   = {r_off_hat * s:.10f}")
    print(f"    z_off_hh   = {z_off_hat * s:.10f}")
    print(f"    a_hat_hat  = 1.0  (by definition)")

    print(f"\n  Physical:")
    print(f"    a          = {a_physical * 1e6:.4f} um")
    print(f"    r_off      = {r_off_hat * L_c * 1e6:.4f} um")
    print(f"    z_off      = {z_off_hat * L_c * 1e6:.4f} um")


# ============================================================
# BEGIN: Moore-Spence bifurcation solver (added section)
# ============================================================


def _moore_spence_eval(r_off_hat, z_off_hat, a_hat, shared_data):
    """Build mesh at (r, z, a) and evaluate forces + AD Jacobian.
    Returns (F, J_hh, mesh_data)."""
    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data
    md = setup_moving_mesh(r_off_hat, z_off_hat, a_hat,
                           R_hat, H_hat, W_hat, L_c, U_c, Re,
                           G_hat, U_m_hat, u_2d, p_2d)
    F, J = evaluate_forces(0.0, 0.0, md, compute_jacobian=True)
    return F, J, md


def _moore_spence_eval_tlm(r_off_hat, z_off_hat, a_hat, shared_data, phi):
    """Build mesh at (r, z, a) and evaluate forces + J@phi via TLM.
    Returns (F, Jphi)."""
    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data
    md = setup_moving_mesh(r_off_hat, z_off_hat, a_hat,
                           R_hat, H_hat, W_hat, L_c, U_c, Re,
                           G_hat, U_m_hat, u_2d, p_2d)
    F, Jphi = evaluate_forces_tlm(0.0, 0.0, md, phi)
    return F, Jphi


def moore_spence_solve(r_off_hat_eq, z_off_hat_eq, a_hat_init, shared_data, *, tol=1e-8, max_iter=20, eps_fd=1e-5):
    """
    Find a bifurcation point (r*, z*, a*) of the particle force map
    F(r, z; a) by solving the Moore-Spence extended system:

        G1:  F(r, z, a)         = 0    (equilibrium condition)
        G2:  J_hh(r,z,a) * phi  = 0    (singularity of Jacobian)
        G3:  l^T phi - 1        = 0    (null-vector normalisation)

    5 unknowns: y = (delta_r_hh, delta_z_hh, a_hat, phi[0], phi[1])
    solved iteratively with Newton's method.

    Spatial derivatives use the mesh displacement mechanism (no re-meshing).
    The parameter derivative d/da requires re-meshing.

    Parameters
    ----------
    r_off_hat_eq, z_off_hat_eq : float
        Equilibrium position (hat coords) at starting particle size.
    a_hat_init : float
        Initial particle-size parameter (hat coords).
    shared_data : tuple
        (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d).
    tol : float
        Convergence tolerance on |G|.
    max_iter : int
        Maximum Newton iterations.
    eps_fd : float
        FD step size in hat-hat coordinates for spatial perturbations.

    Returns
    -------
    r_off_hat, z_off_hat, a_hat, phi, converged
    """
    r = float(r_off_hat_eq)
    z = float(z_off_hat_eq)
    a = float(a_hat_init)

    F0, J0, _ = _moore_spence_eval(r, z, a, shared_data)
    phi = estimate_null_vector(J0)
    l_vec = phi.copy()
    phi = phi / np.dot(l_vec, phi)

    print("\n" + "=" * 65)
    print("  MOORE-SPENCE Bifurcation Solver")
    print(f"  Start: r = {r:.6f}, z = {z:.6f}, a = {a:.6f}")
    sv0 = np.linalg.svd(J0, compute_uv=False)
    print(f"  |F_0| = {np.linalg.norm(F0):.4e}, "
          f"sigma_min(J_0) = {sv0.min():.4e}")
    print("=" * 65)

    for k in range(max_iter):

        # (a) Base evaluation at current (r, z, a)
        F_base, J_base, md_base = _moore_spence_eval(r, z, a, shared_data)
        scale = md_base['scale']

        # (b) Residual of the extended system
        G1 = F_base
        G2 = J_base @ phi
        G3_val = np.dot(l_vec, phi) - 1.0
        G = np.concatenate([G1, G2, [G3_val]])
        res = np.linalg.norm(G)

        sv = np.linalg.svd(J_base, compute_uv=False)
        print(f"\n  Iter {k:2d} | r = {r:+.8f}  z = {z:+.8f}  a = {a:.8f}")
        print(f"         | |G| = {res:.4e}  |F| = {np.linalg.norm(G1):.4e}"
              f"  |J*phi| = {np.linalg.norm(G2):.4e}"
              f"  sigma_min = {sv.min():.4e}")

        if res < tol:
            L_c = shared_data[3]
            a_phys = a * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations!")
            print(f"     r_off_hat  = {r:.10f}")
            print(f"     z_off_hat  = {z:.10f}")
            print(f"     a_hat      = {a:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi        = ({phi[0]:.8f}, {phi[1]:.8f})")
            print(f"     sigma_min  = {sv.min():.6e}")
            return r, z, a, phi, True

        # (c) Spatial dJ·phi via central FD + TLM on mesh displacement
        eps_hh = eps_fd

        _, Jphi_rp = evaluate_forces_tlm(+eps_hh, 0.0, md_base, phi)
        _, Jphi_rm = evaluate_forces_tlm(-eps_hh, 0.0, md_base, phi)
        dJphi_dr = (Jphi_rp - Jphi_rm) / (2 * eps_hh)

        _, Jphi_zp = evaluate_forces_tlm(0.0, +eps_hh, md_base, phi)
        _, Jphi_zm = evaluate_forces_tlm(0.0, -eps_hh, md_base, phi)
        dJphi_dz = (Jphi_zp - Jphi_zm) / (2 * eps_hh)

        # (d) Parameter central FD w.r.t. a_hat (requires re-meshing)
        eps_a = eps_fd * max(abs(a), 1e-4)
        F_ap, Jphi_ap = _moore_spence_eval_tlm(r, z, a + eps_a, shared_data, phi)
        F_am, Jphi_am = _moore_spence_eval_tlm(r, z, a - eps_a, shared_data, phi)

        dF_da = (F_ap - F_am) / (2 * eps_a)
        dJphi_da = (Jphi_ap - Jphi_am) / (2 * eps_a)

        # (e) Assemble 5x5 Jacobian DG
        #     columns: [d(delta_r_hh), d(delta_z_hh), da, dphi0, dphi1]
        DG = np.zeros((5, 5))

        DG[0:2, 0:2] = J_base              # dG1/d(r_hh, z_hh) = J_hh
        DG[0:2, 2]   = dF_da               # dG1/da

        DG[2:4, 0]   = dJphi_dr            # dG2/d(r_hh)
        DG[2:4, 1]   = dJphi_dz            # dG2/d(z_hh)
        DG[2:4, 2]   = dJphi_da            # dG2/da
        DG[2:4, 3:5] = J_base              # dG2/dphi = J_hh

        DG[4, 3:5]   = l_vec               # dG3/dphi = l

        # (f) Solve Newton step  DG * dy = -G
        cond_DG = np.linalg.cond(DG)
        print(f"         | cond(DG) = {cond_DG:.2e}")

        if cond_DG > 1e14:
            print("  !! DG extremely ill-conditioned – aborting.")
            return r, z, a, phi, False

        try:
            dy = np.linalg.solve(DG, -G)
        except np.linalg.LinAlgError:
            print("  !! DG singular – aborting.")
            return r, z, a, phi, False

        # (g) Clamp a_hat step to 20 % relative change (only a, not spatial)
        max_a_step = 0.2 * max(abs(a), 0.01)
        if abs(dy[2]) > max_a_step:
            dy[2] = np.sign(dy[2]) * max_a_step
            print(f"         | a_hat step clamped to {dy[2]:+.4e}")

        # (h) Clamp spatial steps to avoid unreasonable jumps
        max_spatial_step = 0.5 * scale
        for idx in [0, 1]:
            if abs(dy[idx]) > max_spatial_step:
                dy[idx] = np.sign(dy[idx]) * max_spatial_step
                print(f"         | spatial step dy[{idx}] clamped to {dy[idx]:+.4e}")

        print(f"         | step: dr_hh={dy[0]:+.4e}  dz_hh={dy[1]:+.4e}"
              f"  da={dy[2]:+.4e}")

        # (i) Armijo backtracking line search
        alpha = 1.0
        for ls in range(12):
            r_try = r + alpha * dy[0] / scale
            z_try = z + alpha * dy[1] / scale
            a_try = max(a + alpha * dy[2], 1e-6)
            phi_try = phi + alpha * dy[3:5]

            try:
                F_try, J_try, _ = _moore_spence_eval(r_try, z_try, a_try, shared_data)
            except MeshFlippedError:
                print(f"         | line search: flipped elements at alpha={alpha:.4e}, halving")
                alpha *= 0.5
                continue

            G_try = np.concatenate([F_try, J_try @ phi_try,
                                    [np.dot(l_vec, phi_try) - 1.0]])
            res_try = np.linalg.norm(G_try)

            if res_try < (1.0 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5
        else:
            print(f"         | line search: no descent found, alpha={alpha:.4e}")

        if alpha < 1.0:
            print(f"         | line search: alpha={alpha:.4f}")

        r += alpha * dy[0] / scale
        z += alpha * dy[1] / scale
        a += alpha * dy[2]
        phi += alpha * dy[3:5]
        phi = phi / np.dot(l_vec, phi)

        if a <= 0:
            a = 1e-4
            print("  !! a_hat clamped to minimum")

    print(f"\n  Moore-Spence did not converge after {max_iter} iterations.")
    return r, z, a, phi, False


# ============================================================
# END: Moore-Spence bifurcation solver (added section)
# ============================================================


# ============================================================
# HAT-SYSTEM IMPLEMENTATION (no second nondimensionalization)
# Moving mesh for all parameters: r, z, and a
# ============================================================


def setup_moving_mesh_hat(r_off_hat_init, z_off_hat_init, a_hat_init,
                          R_hat, H_hat, W_hat, Re, G_hat, U_m_hat,
                          u_bar_2d_hat, p_bar_tilde_2d_hat,
                          a_mesh_size_ref=None):
    """Generate mesh ONCE in hat coordinates. Sphere size changes via moving mesh.

    Parameters
    ----------
    a_mesh_size_ref : float, optional
        If given, the local element size near the particle is computed as
        ``particle_maxh_rel * a_mesh_size_ref`` *instead of* the actual
        a_hat_init.  This lets us decouple the gmsh refinement scale from
        the geometric particle radius, which is essential when comparing
        moving-mesh evaluations against fresh-mesh evaluations: only with
        a fixed reference mesh size are the two discretizations directly
        comparable.  Defaults to a_hat_init (legacy behaviour).
    """

    L_hat = 4 * max(H_hat, W_hat)
    if a_mesh_size_ref is None:
        a_mesh_size_ref = a_hat_init

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R_hat, H_hat, W_hat, L_hat, a_hat_init,
        particle_maxh_rel * a_mesh_size_ref,
        global_maxh_rel * min(H_hat, W_hat),
        r_off_hat_init, z_off_hat_init)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    dist = sqrt((X_ref[0] - cx)**2 + (X_ref[1] - cy)**2 + (X_ref[2] - cz)**2)

    # Bump: 1 on sphere surface (dist = a_hat_init), linear decay to 0 at r_cut.
    # All mesh nodes satisfy dist >= a_hat_init (outside the sphere).
    r_cut = Constant(0.5 * min(H_hat, W_hat))
    a_c = Constant(a_hat_init)
    bump = max_value(Constant(0.0),
                     1.0 - max_value(Constant(0.0), dist - a_c) / (r_cut - a_c))

    # Unit direction from sphere center (for radial sphere scaling).
    # Safe because dist >= a_hat_init > 0 on all mesh nodes.
    d_hat_x = (X_ref[0] - cx) / dist
    d_hat_y = (X_ref[1] - cy) / dist
    d_hat_z = (X_ref[2] - cz) / dist

    theta_half = tags["theta"] / 2.0

    _, ref_signs = check_mesh_quality(mesh3d)

    print(f"  [setup_hat] Mesh in hat coords: R={R_hat:.2f}, H={H_hat:.2f}, "
          f"W={W_hat:.2f}, a={a_hat_init:.4f}, L={L_hat:.2f}")

    return {
        'mesh3d': mesh3d, 'tags': tags,
        'X_ref': X_ref, 'V_def': V_def,
        'bump': bump,
        'd_hat_x': d_hat_x, 'd_hat_y': d_hat_y, 'd_hat_z': d_hat_z,
        'cos_th': math.cos(theta_half),
        'sin_th': math.sin(theta_half),
        'R_hat': R_hat, 'H_hat': H_hat, 'W_hat': W_hat,
        'L_hat': L_hat, 'a_hat_init': a_hat_init,
        'G_hat': G_hat, 'Re': Re, 'U_m_hat': U_m_hat,
        'u_bar_2d_hat': u_bar_2d_hat,
        'p_bar_tilde_2d_hat': p_bar_tilde_2d_hat,
        'ref_signs': ref_signs,
    }


def _build_xi_hat(delta_r, delta_z, delta_a, md):
    """Build mesh deformation: translation (delta_r, delta_z) + sphere scaling (delta_a)."""

    xi = Function(md['V_def'], name="xi")
    xi.interpolate(as_vector([
        (delta_r * md['cos_th'] + delta_a * md['d_hat_x']) * md['bump'],
        (delta_r * md['sin_th'] + delta_a * md['d_hat_y']) * md['bump'],
        (delta_z              + delta_a * md['d_hat_z']) * md['bump'],
    ]))
    return xi


def _build_a_total_fn(delta_a, md, mesh3d):
    """Build a_total = a_hat_init + delta_a as Function(R_space) on the tape."""
    R_space = FunctionSpace(mesh3d, "R", 0)
    a_fn = Function(R_space, name="a_total")
    a_fn.interpolate(Constant(md['a_hat_init']) + delta_a)
    return a_fn


def _build_Re_p_fn(delta_a, md, mesh3d):
    """Compute Re_p(a_hat) = Re * U_m_hat * (a_hat_init + delta_a)^2 as Function(R_space) on the tape."""

    R_space = FunctionSpace(mesh3d, "R", 0)
    Re_p_fn = Function(R_space, name="Re_p")
    Re_p_fn.interpolate(
        Constant(md['Re'] * md['U_m_hat'])
        * (Constant(md['a_hat_init']) + delta_a)
        * (Constant(md['a_hat_init']) + delta_a))
    return Re_p_fn


def evaluate_forces_hat(delta_r_hat, delta_z_hat, delta_a_hat, mesh_data,
                        compute_jacobian=True):
    """Evaluate particle forces in hat coordinates with AD Jacobian.

    Returns F (2-vector) if compute_jacobian=False,
    or (F, J) where J is (2x3) if compute_jacobian=True.
    J columns: [dF/d(delta_r), dF/d(delta_z), dF/d(delta_a)].
    """

    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']
    R_space = FunctionSpace(mesh3d, "R", 0)

    delta_r = Function(R_space, name="delta_r").assign(delta_r_hat)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hat)
    delta_a = Function(R_space, name="delta_a").assign(delta_a_hat)

    xi = _build_xi_hat(delta_r, delta_z, delta_a, md)
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['G_hat'],
        mesh3d, md['tags'], md['u_bar_2d_hat'], md['p_bar_tilde_2d_hat'],
        X_ref=md['X_ref'], xi=xi)

    pf = perturbed_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['L_hat'],
        md['a_hat_init'], md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    if not compute_jacobian:
        stop_annotating()
        get_working_tape().clear_tape()
        return F

    c_r = Control(delta_r)
    c_z = Control(delta_z)
    c_a = Control(delta_a)

    Jhat_x = ReducedFunctional(F_p_x, [c_r, c_z, c_a])
    Jhat_z = ReducedFunctional(F_p_z, [c_r, c_z, c_a])

    dFx = Jhat_x.derivative()
    dFz = Jhat_z.derivative()

    J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = float(dFx[j].dat.data_ro[0])
        J[1, j] = float(dFz[j].dat.data_ro[0])

    stop_annotating()
    get_working_tape().clear_tape()
    return F, J


def evaluate_forces_tlm_hat(delta_r_hat, delta_z_hat, delta_a_hat,
                             mesh_data, phi):
    """Compute F and J_spatial @ phi via TLM (forward-mode AD).

    phi is a 2-vector [phi_r, phi_z] (spatial null vector).
    Returns (F, Jphi) where Jphi = J_spatial @ phi (2-vector).
    """

    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']
    R_space = FunctionSpace(mesh3d, "R", 0)

    delta_r = Function(R_space, name="delta_r").assign(delta_r_hat)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hat)
    delta_a = Function(R_space, name="delta_a").assign(delta_a_hat)

    xi = _build_xi_hat(delta_r, delta_z, delta_a, md)
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['G_hat'],
        mesh3d, md['tags'], md['u_bar_2d_hat'], md['p_bar_tilde_2d_hat'],
        X_ref=md['X_ref'], xi=xi)

    pf = perturbed_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['L_hat'],
        md['a_hat_init'], md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    # Seed TLM on spatial controls only (phi is the spatial null vector)
    delta_r.block_variable.tlm_value = Function(R_space).assign(phi[0])
    delta_z.block_variable.tlm_value = Function(R_space).assign(phi[1])

    tape = get_working_tape()
    tape.evaluate_tlm()

    Jphi = np.array([
        float(F_p_x.block_variable.tlm_value),
        float(F_p_z.block_variable.tlm_value),
    ])

    stop_annotating()
    get_working_tape().clear_tape()
    return F, Jphi


def evaluate_forces_jac_hessian_hat(delta_r_hat, delta_z_hat, delta_a_hat,
                                     mesh_data, phi):
    """Evaluate F, J, and d(J·phi)/dx via reverse + Hessian-vector AD.

    For each force component F_i (i = x, z), pyadjoint's Hessian-vector
    product H_i · m_dot with m_dot = (phi[0], phi[1], 0) evaluates to
        d/dx [ sum_j dF_i/dx_j · phi_j ] = d/dx [ (J·phi)_i ]
    which is exactly the column-block dG2/dx that Moore-Spence needs,
    without any finite differences.

    Returns
    -------
    F        : (2,)  forces (F_x, F_z)
    J        : (2,3) Jacobian [dF/d(delta_r), dF/d(delta_z), dF/d(delta_a)]
    dJphi_dx : (2,3) entries d/dx_k [(J·phi)_i] from H_i · phi_extended.
               Rows = force components, cols = [d/dr, d/dz, d/da].
    """

    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']
    R_space = FunctionSpace(mesh3d, "R", 0)

    delta_r = Function(R_space, name="delta_r").assign(delta_r_hat)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hat)
    delta_a = Function(R_space, name="delta_a").assign(delta_a_hat)

    xi = _build_xi_hat(delta_r, delta_z, delta_a, md)
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['G_hat'],
        mesh3d, md['tags'], md['u_bar_2d_hat'], md['p_bar_tilde_2d_hat'],
        X_ref=md['X_ref'], xi=xi)

    pf = perturbed_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['L_hat'],
        md['a_hat_init'], md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    c_r = Control(delta_r)
    c_z = Control(delta_z)
    c_a = Control(delta_a)
    controls = [c_r, c_z, c_a]

    rf_x = ReducedFunctional(F_p_x, controls)
    rf_z = ReducedFunctional(F_p_z, controls)

    # First-order: full Jacobian via reverse AD
    dFx = rf_x.derivative()
    dFz = rf_z.derivative()
    J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = float(dFx[j].dat.data_ro[0])
        J[1, j] = float(dFz[j].dat.data_ro[0])

    # Hessian-vector seed: phi extended with 0 for the a-direction
    phi_r_fn = Function(R_space).assign(float(phi[0]))
    phi_z_fn = Function(R_space).assign(float(phi[1]))
    phi_a_fn = Function(R_space).assign(0.0)
    m_dot = [phi_r_fn, phi_z_fn, phi_a_fn]

    Hphi_x = rf_x.hessian(m_dot)
    Hphi_z = rf_z.hessian(m_dot)

    dJphi_dx = np.zeros((2, 3))
    for j in range(3):
        dJphi_dx[0, j] = float(Hphi_x[j].dat.data_ro[0])
        dJphi_dx[1, j] = float(Hphi_z[j].dat.data_ro[0])

    stop_annotating()
    get_working_tape().clear_tape()
    return F, J, dJphi_dx


def diag_hessian_vs_fd(dr, dz, da, md, phi, eps=1e-4):
    """Sanity check: AD Hessian-vector product vs FD of reverse-AD Jacobian.

    Both should give d(J·phi)/dx for x = (delta_r, delta_z, delta_a).
    Useful to confirm pyadjoint's Hessian works through the
    mesh-deformation pathway before relying on it in Moore-Spence.
    """
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 7: AD Hessian-vector product vs FD-of-reverse-AD")
    print("=" * 70)

    _, _, dJphi_dx_AD = evaluate_forces_jac_hessian_hat(dr, dz, da, md, phi)

    dJphi_dx_FD = np.zeros((2, 3))
    for k, (dr_off, dz_off, da_off) in enumerate(
            [(eps, 0, 0), (0, eps, 0), (0, 0, eps)]):
        _, J_p = evaluate_forces_hat(dr + dr_off, dz + dz_off, da + da_off, md)
        _, J_m = evaluate_forces_hat(dr - dr_off, dz - dz_off, da - da_off, md)
        dJphi_dx_FD[:, k] = (J_p[:, :2] @ phi - J_m[:, :2] @ phi) / (2 * eps)

    print(f"\n  eps (FD) = {eps:.1e}")
    print(f"\n  AD H·phi (rows = F components, cols = d/dr, d/dz, d/da):")
    for i in range(2):
        print(f"    [{' '.join(f'{dJphi_dx_AD[i,j]:+12.6e}' for j in range(3))}]")
    print(f"\n  FD-of-AD:")
    for i in range(2):
        print(f"    [{' '.join(f'{dJphi_dx_FD[i,j]:+12.6e}' for j in range(3))}]")

    diff = dJphi_dx_AD - dJphi_dx_FD
    print(f"\n  Difference (AD - FD):")
    for i in range(2):
        print(f"    [{' '.join(f'{diff[i,j]:+12.6e}' for j in range(3))}]")

    rel = np.linalg.norm(diff) / (np.linalg.norm(dJphi_dx_FD) + 1e-30)
    print(f"\n  Relative diff: {rel:.4e}")
    return dJphi_dx_AD, dJphi_dx_FD


def moore_spence_solve_hat_ad(r_off_hat_eq, z_off_hat_eq, a_hat_init, shared_data,
                               *, tol=1e-8, max_iter=20):
    """Moore-Spence with AD-based Hessian (pure reverse + Hessian-vector AD).

    Builds the full 5x5 DG every iteration without any finite differences.
    Cost per iteration: 1 forward + 2 reverse + 2 Hessian-vector products
    (plus line-search forward evaluations).

    Should give Newton-like quadratic convergence if the mesh-deformation
    pathway in pyadjoint is twice differentiable.
    """

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    r = float(r_off_hat_eq)
    z = float(z_off_hat_eq)
    a = float(a_hat_init)

    md = setup_moving_mesh_hat(r, z, a, R_hat, H_hat, W_hat,
                                Re, G_hat, U_m_hat, u_2d, p_2d)

    F0, J0_full = evaluate_forces_hat(0.0, 0.0, 0.0, md)
    J0 = J0_full[:, :2]
    phi = estimate_null_vector(J0)
    l_vec = phi.copy()
    phi = phi / np.dot(l_vec, phi)

    dr, dz, da = 0.0, 0.0, 0.0

    sv0 = np.linalg.svd(J0, compute_uv=False)
    print("\n" + "=" * 65)
    print(f"  MOORE-SPENCE (hat system, AD Hessian, single mesh)")
    print(f"  Start: r = {r:.6f}, z = {z:.6f}, a = {a:.6f}")
    print(f"  |F_0| = {np.linalg.norm(F0):.4e}, sigma_min = {sv0.min():.4e}")
    print("=" * 65)

    for k in range(max_iter):

        # All entries of DG via AD: 1 forward + 2 reverse + 2 H·v
        F_base, J_full, dJphi_dx = evaluate_forces_jac_hessian_hat(
            dr, dz, da, md, phi)
        J_sp = J_full[:, :2]
        dF_da = J_full[:, 2]

        G1 = F_base
        G2 = J_sp @ phi
        G3_val = np.dot(l_vec, phi) - 1.0
        G = np.concatenate([G1, G2, [G3_val]])
        res = np.linalg.norm(G)

        sv = np.linalg.svd(J_sp, compute_uv=False)
        r_cur, z_cur, a_cur = r + dr, z + dz, a + da
        print(f"\n  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f}  a = {a_cur:.8f}")
        print(f"         | |G| = {res:.4e}  |F| = {np.linalg.norm(G1):.4e}"
              f"  |J*phi| = {np.linalg.norm(G2):.4e}"
              f"  sigma_min = {sv.min():.4e}")

        if res < tol:
            a_phys = a_cur * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations!")
            print(f"     r_off_hat = {r_cur:.10f}")
            print(f"     z_off_hat = {z_cur:.10f}")
            print(f"     a_hat     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = ({phi[0]:.8f}, {phi[1]:.8f})")
            print(f"     sigma_min = {sv.min():.6e}")
            return r_cur, z_cur, a_cur, phi, True

        # Assemble DG (5x5) — every block from AD
        DG = np.zeros((5, 5))
        DG[0:2, 0:2] = J_sp                # dG1/d(r,z)
        DG[0:2, 2]   = dF_da               # dG1/da
        DG[2:4, 0]   = dJphi_dx[:, 0]      # d(J·phi)/dr  via H·phi
        DG[2:4, 1]   = dJphi_dx[:, 1]      # d(J·phi)/dz  via H·phi
        DG[2:4, 2]   = dJphi_dx[:, 2]      # d(J·phi)/da  via H·phi
        DG[2:4, 3:5] = J_sp                # dG2/dphi
        DG[4, 3:5]   = l_vec               # dG3/dphi

        cond_DG = np.linalg.cond(DG)
        print(f"         | cond(DG) = {cond_DG:.2e}")

        if cond_DG > 1e14:
            print("  !! DG ill-conditioned — aborting.")
            return r_cur, z_cur, a_cur, phi, False

        try:
            dy = np.linalg.solve(DG, -G)
        except np.linalg.LinAlgError:
            print("  !! DG singular — aborting.")
            return r_cur, z_cur, a_cur, phi, False

        print(f"         | step: dr={dy[0]:+.4e}  dz={dy[1]:+.4e}  da={dy[2]:+.4e}")

        # Line search (uses cheaper Jacobian-only evaluator)
        alpha = 1.0
        for _ in range(12):
            dr_try = dr + alpha * dy[0]
            dz_try = dz + alpha * dy[1]
            da_try = da + alpha * dy[2]
            phi_try = phi + alpha * dy[3:5]

            if a + da_try <= 0:
                alpha *= 0.5
                continue

            try:
                F_try, J_try = evaluate_forces_hat(dr_try, dz_try, da_try, md)
            except MeshFlippedError:
                print(f"         | ls: flipped at alpha={alpha:.4e}, halving")
                alpha *= 0.5
                continue

            G_try = np.concatenate([F_try, J_try[:, :2] @ phi_try,
                                    [np.dot(l_vec, phi_try) - 1.0]])
            res_try = np.linalg.norm(G_try)
            print(f"         | ls alpha={alpha:.4f}: |G|={res_try:.4e}"
                  f"  {'OK' if res_try < (1 - 1e-4 * alpha) * res else 'reject'}")

            if res_try < (1 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5

        dr += alpha * dy[0]
        dz += alpha * dy[1]
        da += alpha * dy[2]
        phi += alpha * dy[3:5]
        phi = phi / np.dot(l_vec, phi)

        gc.collect()

    r_cur, z_cur, a_cur = r + dr, z + dz, a + da
    print(f"\n  Moore-Spence (AD) did not converge after {max_iter} iterations.")
    return r_cur, z_cur, a_cur, phi, False


def newton_root_refine_hat(r_off_hat_init, z_off_hat_init, a_hat, shared_data,
                           *, tol=1e-10, max_iter=15):
    """Phase 1: Find equilibrium at fixed a_hat in hat coordinates (single mesh)."""

    print("\n" + "=" * 65)
    print(f"  Newton Root Refinement (hat system, a_hat = {a_hat:.6f})")
    print("=" * 65)

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    md = setup_moving_mesh_hat(r_off_hat_init, z_off_hat_init, a_hat,
                               R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d)

    dr, dz = 0.0, 0.0

    for k in range(max_iter):
        F, J_full = evaluate_forces_hat(dr, dz, 0.0, md)
        J = J_full[:, :2]

        res = np.linalg.norm(F)
        r_cur = r_off_hat_init + dr
        z_cur = z_off_hat_init + dz

        print(f"  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f}"
              f" | |F| = {res:.4e}  cond(J) = {np.linalg.cond(J):.2e}")

        if res < tol:
            print(f"  -> Converged after {k} iterations.\n")
            return r_cur, z_cur, True, md

        dx_n = np.linalg.solve(J, -F)

        alpha = 1.0
        for ls in range(10):
            try:
                F_try = evaluate_forces_hat(
                    dr + alpha * dx_n[0], dz + alpha * dx_n[1], 0.0,
                    md, compute_jacobian=False)
            except MeshFlippedError:
                print(f"         | line search: flipped elements at alpha={alpha:.4e}, halving")
                alpha *= 0.5
                continue
            if np.linalg.norm(F_try) < (1 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5

        dr += alpha * dx_n[0]
        dz += alpha * dx_n[1]

        gc.collect()

    r_cur = r_off_hat_init + dr
    z_cur = z_off_hat_init + dz
    print(f"  Did not converge after {max_iter} iterations.\n")
    return r_cur, z_cur, False, md


def moore_spence_solve_hat(r_off_hat_eq, z_off_hat_eq, a_hat_init, shared_data,
                           *, tol=1e-8, max_iter=20, eps_fd=1e-5,
                           jacobian_update='broyden'):
    """Moore-Spence bifurcation solver in hat coordinates.

    All three parameters (r, z, a) use moving mesh on a SINGLE mesh —
    no remeshing when a changes.

    Parameters
    ----------
    jacobian_update : str
        'exact'   — rebuild DG every iteration via FD of reverse-AD Jacobian.
                     Accurate but expensive (~7 AD evaluations per iteration).
        'broyden' — build DG exactly once (iter 0), then apply Broyden rank-1
                     updates. Only 1 AD evaluation per iteration, superlinear
                     convergence. Exact AD values are injected into DG rows
                     0-1 and the J_sp block in rows 2-3 after each update.
    """

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    r = float(r_off_hat_eq)
    z = float(z_off_hat_eq)
    a = float(a_hat_init)

    # Generate mesh once
    md = setup_moving_mesh_hat(r, z, a, R_hat, H_hat, W_hat,
                               Re, G_hat, U_m_hat, u_2d, p_2d)

    # Initial evaluation at delta = 0
    F0, J0_full = evaluate_forces_hat(0.0, 0.0, 0.0, md)
    J0 = J0_full[:, :2]
    phi = estimate_null_vector(J0)
    l_vec = phi.copy()
    phi = phi / np.dot(l_vec, phi)

    # Running displacement from initial mesh position
    dr, dz, da = 0.0, 0.0, 0.0

    sv0 = np.linalg.svd(J0, compute_uv=False)
    print("\n" + "=" * 65)
    print(f"  MOORE-SPENCE (hat system, single mesh, update={jacobian_update})")
    print(f"  Start: r = {r:.6f}, z = {z:.6f}, a = {a:.6f}")
    print(f"  |F_0| = {np.linalg.norm(F0):.4e}, sigma_min = {sv0.min():.4e}")
    print("=" * 65)

    DG = None       # built on first iteration (always exact)
    G_prev = None   # for Broyden update
    dy_accepted = None  # for Broyden update

    for k in range(max_iter):

        # (a) Base evaluation — full 2x3 Jacobian via reverse AD
        F_base, J_full = evaluate_forces_hat(dr, dz, da, md)
        J_sp = J_full[:, :2]    # 2x2 spatial Jacobian
        dF_da = J_full[:, 2]    # 2x1 parameter derivative

        # (b) Residual of the extended system
        G1 = F_base
        G2 = J_sp @ phi
        G3_val = np.dot(l_vec, phi) - 1.0
        G = np.concatenate([G1, G2, [G3_val]])
        res = np.linalg.norm(G)

        sv = np.linalg.svd(J_sp, compute_uv=False)
        r_cur, z_cur, a_cur = r + dr, z + dz, a + da
        print(f"\n  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f}"
              f"  a = {a_cur:.8f}")
        print(f"         | |G| = {res:.4e}  |F| = {np.linalg.norm(G1):.4e}"
              f"  |J*phi| = {np.linalg.norm(G2):.4e}"
              f"  sigma_min = {sv.min():.4e}")

        if res < tol:
            a_phys = a_cur * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations!")
            print(f"     r_off_hat = {r_cur:.10f}")
            print(f"     z_off_hat = {z_cur:.10f}")
            print(f"     a_hat     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = ({phi[0]:.8f}, {phi[1]:.8f})")
            print(f"     sigma_min = {sv.min():.6e}")
            return r_cur, z_cur, a_cur, phi, True

        # (c) Build or update DG
        build_exact = (jacobian_update == 'exact') or (k == 0)

        if build_exact:
            # Exact DG via central FD of reverse-AD Jacobian
            eps = eps_fd

            _, J_rp = evaluate_forces_hat(dr + eps, dz, da, md)
            _, J_rm = evaluate_forces_hat(dr - eps, dz, da, md)
            dJphi_dr = (J_rp[:, :2] @ phi - J_rm[:, :2] @ phi) / (2 * eps)

            _, J_zp = evaluate_forces_hat(dr, dz + eps, da, md)
            _, J_zm = evaluate_forces_hat(dr, dz - eps, da, md)
            dJphi_dz = (J_zp[:, :2] @ phi - J_zm[:, :2] @ phi) / (2 * eps)

            eps_a = eps * max(abs(a_cur), 1e-4)
            _, J_ap = evaluate_forces_hat(dr, dz, da + eps_a, md)
            _, J_am = evaluate_forces_hat(dr, dz, da - eps_a, md)
            dJphi_da = (J_ap[:, :2] @ phi - J_am[:, :2] @ phi) / (2 * eps_a)

            # ---- DIAGNOSTIC: FD convergence check (first iteration only) ----
            if k == 0:
                eps_large = 1e-3
                _, J_rp_lg = evaluate_forces_hat(dr + eps_large, dz, da, md)
                _, J_rm_lg = evaluate_forces_hat(dr - eps_large, dz, da, md)
                dJphi_dr_lg = (J_rp_lg[:, :2] @ phi - J_rm_lg[:, :2] @ phi) / (2 * eps_large)

                _, J_zp_lg = evaluate_forces_hat(dr, dz + eps_large, da, md)
                _, J_zm_lg = evaluate_forces_hat(dr, dz - eps_large, da, md)
                dJphi_dz_lg = (J_zp_lg[:, :2] @ phi - J_zm_lg[:, :2] @ phi) / (2 * eps_large)

                print(f"         | --- FD convergence of dJphi (eps={eps:.1e} vs eps={eps_large:.1e}) ---")
                print(f"         |   dJphi/dr  eps={eps:.0e}: {dJphi_dr}  eps={eps_large:.0e}: {dJphi_dr_lg}  diff: {np.linalg.norm(dJphi_dr - dJphi_dr_lg):.2e}")
                print(f"         |   dJphi/dz  eps={eps:.0e}: {dJphi_dz}  eps={eps_large:.0e}: {dJphi_dz_lg}  diff: {np.linalg.norm(dJphi_dz - dJphi_dz_lg):.2e}")

            # (d) Assemble 5x5 Jacobian DG
            DG = np.zeros((5, 5))
            DG[0:2, 0:2] = J_sp
            DG[0:2, 2]   = dF_da
            DG[2:4, 0]   = dJphi_dr
            DG[2:4, 1]   = dJphi_dz
            DG[2:4, 2]   = dJphi_da
            DG[2:4, 3:5] = J_sp
            DG[4, 3:5]   = l_vec
            print(f"         | DG built exactly (FD of reverse-AD)")

        else:
            # Broyden rank-1 update: DG += (delta_G - DG @ delta_y) @ delta_y^T / |delta_y|^2
            delta_G = G - G_prev
            Br = delta_G - DG @ dy_accepted
            DG += np.outer(Br, dy_accepted) / np.dot(dy_accepted, dy_accepted)

            # Inject exact AD values into the blocks we know exactly
            DG[0:2, 0:2] = J_sp       # dG1/d(r,z) = J_sp (exact via AD)
            DG[0:2, 2]   = dF_da      # dG1/da (exact via AD)
            DG[2:4, 3:5] = J_sp       # dG2/dphi = J_sp (exact via AD)
            DG[4, 3:5]   = l_vec      # dG3/dphi = l (exact, constant)
            print(f"         | DG updated (Broyden rank-1 + exact AD blocks)")

        cond_DG = np.linalg.cond(DG)
        print(f"         | cond(DG) = {cond_DG:.2e}")

        if cond_DG > 1e14:
            print("  !! DG ill-conditioned — aborting.")
            return r_cur, z_cur, a_cur, phi, False

        try:
            dy = np.linalg.solve(DG, -G)
        except np.linalg.LinAlgError:
            print("  !! DG singular — aborting.")
            return r_cur, z_cur, a_cur, phi, False

        # ---- DIAGNOSTIC: DG solve quality ----
        residual_check = DG @ dy + G
        print(f"         | |DG*dy + G| = {np.linalg.norm(residual_check):.2e}")
        print(f"         | DG matrix:")
        for row_idx in range(5):
            print(f"         |   [{' '.join(f'{DG[row_idx,j]:+.4e}' for j in range(5))}]")
        print(f"         | G  = [{' '.join(f'{g:+.4e}' for g in G)}]")
        print(f"         | dy = [{' '.join(f'{d:+.4e}' for d in dy)}]")

        # (e) Clamp a step to 20% relative change
        max_a_step = 0.2 * max(abs(a_cur), 0.01)
        if abs(dy[2]) > max_a_step:
            dy[2] = np.sign(dy[2]) * max_a_step
            print(f"         | a step clamped to {dy[2]:+.4e}")

        print(f"         | step: dr={dy[0]:+.4e}  dz={dy[1]:+.4e}"
              f"  da={dy[2]:+.4e}")

        # (f) Line search using reverse AD
        alpha = 1.0
        for _ in range(12):
            dr_try = dr + alpha * dy[0]
            dz_try = dz + alpha * dy[1]
            da_try = da + alpha * dy[2]
            phi_try = phi + alpha * dy[3:5]

            if a + da_try <= 0:
                alpha *= 0.5
                continue

            try:
                F_try, J_try = evaluate_forces_hat(
                    dr_try, dz_try, da_try, md)
            except MeshFlippedError:
                print(f"         | line search: flipped elements at alpha={alpha:.4e}, halving")
                alpha *= 0.5
                continue

            G_try = np.concatenate([
                F_try, J_try[:, :2] @ phi_try,
                [np.dot(l_vec, phi_try) - 1.0]])
            res_try = np.linalg.norm(G_try)

            print(f"         | ls alpha={alpha:.4f}: |G|={res_try:.4e}"
                  f"  (|F|={np.linalg.norm(F_try):.4e}"
                  f"  |Jphi|={np.linalg.norm(J_try[:, :2] @ phi_try):.4e})"
                  f"  {'OK' if res_try < (1 - 1e-4 * alpha) * res else 'reject'}")

            if res_try < (1 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5
        else:
            print(f"         | line search: no descent, alpha={alpha:.4e}")

        if alpha < 1:
            print(f"         | line search: alpha={alpha:.4f}")

        # Save for Broyden update (before modifying state)
        G_prev = G.copy()
        dy_accepted = alpha * dy.copy()

        dr += alpha * dy[0]
        dz += alpha * dy[1]
        da += alpha * dy[2]
        phi += alpha * dy[3:5]
        phi = phi / np.dot(l_vec, phi)

        if a + da <= 0:
            da = -a + 1e-4
            print("  !! a_hat clamped to minimum")

        gc.collect()

    r_cur, z_cur, a_cur = r + dr, z + dz, a + da
    print(f"\n  Moore-Spence did not converge after {max_iter} iterations.")
    return r_cur, z_cur, a_cur, phi, False


def bisection_bifurcation_hat(r_off_hat_eq, z_off_hat_eq, a_hat_init,
                               a_hat_lo, a_hat_hi, shared_data,
                               *, tol_a=1e-6, max_bisect=30,
                               newton_tol=1e-10, newton_max_iter=15):
    """Find bifurcation point via bisection on a_hat (no second derivatives).

    Creates a single mesh at (r_off_hat_eq, z_off_hat_eq, a_hat_init) and
    varies a_hat via moving mesh.  For each bisection candidate, runs a
    Newton loop on (r, z) at fixed a_hat to find the equilibrium, then
    checks sigma_min(J_sp).

    Bisection criterion: sign(det(J_sp)) flips at the bifurcation point.
    For a pitchfork on a symmetric branch, det(J) crosses zero (one
    eigenvalue — the symmetry-breaking direction — changes sign) while
    the equilibrium itself persists on both sides.  Newton therefore
    converges on both sides, so a Newton-success bracket would be wrong;
    a sign-of-det bracket is the right invariant.

    Fallback: if only one endpoint converges (e.g. at a fold), the code
    falls back to the classical "Newton-success bracket" logic.

    Parameters
    ----------
    r_off_hat_eq, z_off_hat_eq : float
        Equilibrium position found by Phase 1 at a_hat_init.
    a_hat_init : float
        Reference a_hat (mesh is built here).
    a_hat_lo, a_hat_hi : float
        Bracket for bisection.  One should be on the "equilibrium exists"
        side, the other on the "no equilibrium" side (or both converge but
        sigma_min differs).
    """

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    md = setup_moving_mesh_hat(r_off_hat_eq, z_off_hat_eq, a_hat_init,
                                R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d)

    print("\n" + "=" * 65)
    print(f"  BISECTION on sigma_min")
    print(f"  a_hat in [{a_hat_lo:.6f}, {a_hat_hi:.6f}]")
    print(f"  Mesh ref: r = {r_off_hat_eq:.6f}, z = {z_off_hat_eq:.6f},"
          f" a = {a_hat_init:.6f}")
    print("=" * 65)

    # --- inner Newton at fixed a_hat on the existing mesh -----------------
    def newton_at_a(a_hat_target, r_guess, z_guess):
        """Find equilibrium (r, z) at given a_hat.

        Returns (r, z, sigma_min, det_J, converged).
        """
        da = a_hat_target - a_hat_init
        dr = r_guess - r_off_hat_eq
        dz = z_guess - z_off_hat_eq

        for it in range(newton_max_iter):
            try:
                F, J_full = evaluate_forces_hat(dr, dz, da, md)
            except MeshFlippedError:
                return (r_off_hat_eq + dr, z_off_hat_eq + dz,
                        np.nan, np.nan, False)
            J = J_full[:, :2]
            res = np.linalg.norm(F)

            if res < newton_tol:
                sv = np.linalg.svd(J, compute_uv=False)
                det = np.linalg.det(J)
                return (r_off_hat_eq + dr, z_off_hat_eq + dz,
                        float(sv.min()), float(det), True)

            try:
                dx = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                return (r_off_hat_eq + dr, z_off_hat_eq + dz,
                        0.0, 0.0, False)

            alpha = 1.0
            for _ in range(10):
                try:
                    F_try = evaluate_forces_hat(
                        dr + alpha * dx[0], dz + alpha * dx[1], da,
                        md, compute_jacobian=False)
                except MeshFlippedError:
                    alpha *= 0.5
                    continue
                if np.linalg.norm(F_try) < (1 - 1e-4 * alpha) * res:
                    break
                alpha *= 0.5

            dr += alpha * dx[0]
            dz += alpha * dx[1]

        # did not converge — return last state
        try:
            F, J_full = evaluate_forces_hat(dr, dz, da, md)
            J = J_full[:, :2]
            sv = np.linalg.svd(J, compute_uv=False)
            det = np.linalg.det(J)
            return (r_off_hat_eq + dr, z_off_hat_eq + dz,
                    float(sv.min()), float(det), False)
        except MeshFlippedError:
            return (r_off_hat_eq + dr, z_off_hat_eq + dz,
                    np.nan, np.nan, False)

    # --- evaluate endpoints -----------------------------------------------
    r_guess, z_guess = r_off_hat_eq, z_off_hat_eq

    r_lo, z_lo, sig_lo, det_lo, conv_lo = newton_at_a(
        a_hat_lo, r_guess, z_guess)
    print(f"  a_hat_lo = {a_hat_lo:.6f} | conv={conv_lo}"
          f"  sigma_min={sig_lo:.4e}  det(J)={det_lo:+.4e}"
          f"  r={r_lo:+.8f}  z={z_lo:+.8f}")

    r_hi, z_hi, sig_hi, det_hi, conv_hi = newton_at_a(
        a_hat_hi, r_guess, z_guess)
    print(f"  a_hat_hi = {a_hat_hi:.6f} | conv={conv_hi}"
          f"  sigma_min={sig_hi:.4e}  det(J)={det_hi:+.4e}"
          f"  r={r_hi:+.8f}  z={z_hi:+.8f}")

    if not conv_lo and not conv_hi:
        print("  !! Newton fails at both endpoints — cannot bisect.")
        return np.nan, np.nan, np.nan, False

    # Ensure that a_hat_lo is the "converging" side
    if not conv_lo and conv_hi:
        a_hat_lo, a_hat_hi = a_hat_hi, a_hat_lo
        r_lo, z_lo, sig_lo, det_lo = r_hi, z_hi, sig_hi, det_hi
        conv_lo = True
        conv_hi = False
        print("  (swapped lo/hi so that lo is the converging side)")

    # --- bisection loop ----------------------------------------------------
    r_best, z_best, a_best, sig_best = r_lo, z_lo, a_hat_lo, sig_lo

    for step in range(max_bisect):
        a_mid = 0.5 * (a_hat_lo + a_hat_hi)

        # Use closest converged point as initial guess (continuation)
        r_mid, z_mid, sig_mid, det_mid, conv_mid = newton_at_a(
            a_mid, r_best, z_best)

        tag = "conv" if conv_mid else "FAIL"
        print(f"  Bisect {step:2d} | a_mid = {a_mid:.8f}  [{tag}]"
              f"  sigma_min={sig_mid:.4e}  det(J)={det_mid:+.4e}"
              f"  |a_hi-a_lo|={abs(a_hat_hi - a_hat_lo):.2e}")

        if conv_mid:
            r_best, z_best, a_best, sig_best = r_mid, z_mid, a_mid, sig_mid

        if conv_lo and conv_hi:
            # Both sides converge — bisect toward smaller sigma_min
            if sig_mid <= sig_lo and sig_mid <= sig_hi:
                if sig_lo < sig_hi:
                    a_hat_hi = a_mid
                    sig_hi = sig_mid
                else:
                    a_hat_lo = a_mid
                    sig_lo = sig_mid
            elif sig_mid < sig_lo:
                a_hat_lo = a_mid
                sig_lo = sig_mid
                r_lo, z_lo = r_mid, z_mid
            else:
                a_hat_hi = a_mid
                sig_hi = sig_mid
                r_hi, z_hi = r_mid, z_mid
        else:
            # One side converges, one doesn't — standard bisection
            if conv_mid:
                a_hat_lo = a_mid
                r_lo, z_lo, sig_lo, det_lo = r_mid, z_mid, sig_mid, det_mid
            else:
                a_hat_hi = a_mid

        if abs(a_hat_hi - a_hat_lo) < tol_a:
            a_phys = a_best * L_c
            print(f"\n  -> Bisection converged: a_hat = {a_best:.10f}"
                  f"  (|interval| = {abs(a_hat_hi - a_hat_lo):.2e})")
            print(f"     r_off_hat = {r_best:.10f}")
            print(f"     z_off_hat = {z_best:.10f}")
            print(f"     a_hat     = {a_best:.10f}"
                  f"  (a = {a_phys * 1e6:.4f} um)")
            print(f"     sigma_min = {sig_best:.6e}")
            return r_best, z_best, a_best, True

    print(f"\n  Bisection did not converge after {max_bisect} steps."
          f"  Best: a_hat = {a_best:.10f}, sigma_min = {sig_best:.6e}")
    return r_best, z_best, a_best, False


# ============================================================
# END: Hat-system implementation
# ============================================================


# ============================================================
# DIAGNOSTIC TESTS for Moore-Spence convergence issues
# ============================================================


def diag_fd_convergence_study(dr, dz, da, md, phi, eps_values=None):
    """Test 1: FD convergence study for d(J·phi)/d(r,z,a).

    Computes dJphi via central FD at multiple eps values.
    If the FD derivative is well-resolved, consecutive eps values
    should show O(eps^2) convergence.  Large or erratic differences
    indicate noise in the AD Jacobian.
    """
    if eps_values is None:
        eps_values = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6]

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 1: FD convergence study for dJphi")
    print("=" * 70)

    for label, idx_r, idx_z, idx_a in [("dr", 1, 0, 0), ("dz", 0, 1, 0), ("da", 0, 0, 1)]:
        print(f"\n  --- d(J·phi)/d({label}) ---")
        print(f"  {'eps':>10s}  {'dJphi[0]':>14s}  {'dJphi[1]':>14s}  {'|diff to prev|':>14s}")

        prev = None
        for eps in eps_values:
            dr_p = dr + eps * idx_r
            dz_p = dz + eps * idx_z
            da_p = da + eps * idx_a
            dr_m = dr - eps * idx_r
            dz_m = dz - eps * idx_z
            da_m = da - eps * idx_a

            try:
                _, J_p = evaluate_forces_hat(dr_p, dz_p, da_p, md)
                _, J_m = evaluate_forces_hat(dr_m, dz_m, da_m, md)
            except MeshFlippedError:
                print(f"  {eps:10.1e}  MESH FLIPPED")
                continue

            dJphi = (J_p[:, :2] @ phi - J_m[:, :2] @ phi) / (2 * eps)

            diff_str = ""
            if prev is not None:
                diff_str = f"{np.linalg.norm(dJphi - prev):14.4e}"
            prev = dJphi

            print(f"  {eps:10.1e}  {dJphi[0]:+14.6e}  {dJphi[1]:+14.6e}  {diff_str}")


def diag_ad_vs_fd_jacobian(dr, dz, da, md, eps=1e-5):
    """Test 2: Compare AD Jacobian with FD Jacobian at the same point.

    Quantifies the noise level in the AD Jacobian.
    """
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 2: AD vs FD Jacobian comparison")
    print("=" * 70)

    _, J_ad = evaluate_forces_hat(dr, dz, da, md)
    J_sp_ad = J_ad[:, :2]

    F0 = evaluate_forces_hat(dr, dz, da, md, compute_jacobian=False)
    F_rp = evaluate_forces_hat(dr + eps, dz, da, md, compute_jacobian=False)
    F_rm = evaluate_forces_hat(dr - eps, dz, da, md, compute_jacobian=False)
    F_zp = evaluate_forces_hat(dr, dz + eps, da, md, compute_jacobian=False)
    F_zm = evaluate_forces_hat(dr, dz - eps, da, md, compute_jacobian=False)
    F_ap = evaluate_forces_hat(dr, dz, da + eps, md, compute_jacobian=False)
    F_am = evaluate_forces_hat(dr, dz, da - eps, md, compute_jacobian=False)

    J_fd = np.zeros((2, 3))
    J_fd[:, 0] = (F_rp - F_rm) / (2 * eps)
    J_fd[:, 1] = (F_zp - F_zm) / (2 * eps)
    J_fd[:, 2] = (F_ap - F_am) / (2 * eps)

    print(f"\n  AD Jacobian (2x3):")
    for i in range(2):
        print(f"    [{' '.join(f'{J_ad[i,j]:+12.6e}' for j in range(3))}]")

    print(f"\n  FD Jacobian (eps={eps:.1e}):")
    for i in range(2):
        print(f"    [{' '.join(f'{J_fd[i,j]:+12.6e}' for j in range(3))}]")

    diff = J_ad - J_fd
    print(f"\n  Difference (AD - FD):")
    for i in range(2):
        print(f"    [{' '.join(f'{diff[i,j]:+12.6e}' for j in range(3))}]")

    rel = np.linalg.norm(diff) / (np.linalg.norm(J_fd) + 1e-30)
    print(f"\n  Relative diff: {rel:.4e}")
    print(f"  |F at base|:   {np.linalg.norm(F0):.4e}")

    return J_ad, J_fd


def diag_a_dependence(dr, dz, md, da_values=None):
    """Test 3: Check how F depends on delta_a.

    If Re_p and the a^3 volume term are NOT updated with delta_a,
    the force will show the wrong a-dependence.
    Compare F(da) for several da values and check whether dF/da
    is consistent with the AD derivative at da=0.
    """
    if da_values is None:
        da_values = [-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02]

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 3: Force dependence on delta_a")
    print("=" * 70)

    # AD derivative at da=0
    _, J_full = evaluate_forces_hat(dr, dz, 0.0, md)
    dF_da_ad = J_full[:, 2]
    F0 = evaluate_forces_hat(dr, dz, 0.0, md, compute_jacobian=False)

    print(f"\n  dF/da (AD at da=0): [{dF_da_ad[0]:+.6e}, {dF_da_ad[1]:+.6e}]")
    print(f"\n  {'da':>10s}  {'F_r':>14s}  {'F_z':>14s}  {'|F|':>12s}  {'F_r predicted':>14s}  {'F_z predicted':>14s}")

    for da in da_values:
        try:
            F = evaluate_forces_hat(dr, dz, da, md, compute_jacobian=False)
        except MeshFlippedError:
            print(f"  {da:10.4f}  MESH FLIPPED")
            continue

        F_pred = F0 + dF_da_ad * da
        print(f"  {da:10.4f}  {F[0]:+14.6e}  {F[1]:+14.6e}  {np.linalg.norm(F):12.4e}"
              f"  {F_pred[0]:+14.6e}  {F_pred[1]:+14.6e}")

    # Check: does Re_p change with a?
    a_init = md['a_hat_init']
    Re_chan = md['Re']
    U_m = md['U_m_hat']
    print(f"\n  Physics check:")
    print(f"    a_hat_init = {a_init:.6f}")
    print(f"    Re (channel) = {Re_chan:.6e}")
    print(f"    U_m_hat = {U_m:.6e}")
    print(f"    Re_p = Re * U_m * a^2 = {Re_chan * U_m * a_init**2:.6e}")
    print(f"    BUT evaluate_forces_hat passes Re={Re_chan} as Re_p to perturbed_flow!")
    print(f"    And passes a={a_init} (fixed) instead of a_init + delta_a for volume term.")
    print(f"    -> dF/da from AD does NOT capture Re_p(a) or a^3 volume dependencies!")


def diag_moving_mesh_error(r_eq, z_eq, a_hat, shared_data, md_orig, da_test_values=None):
    """Test 4: Moving mesh error vs remeshing.

    For each delta_a, compare F computed via moving mesh (single mesh)
    with F computed on a fresh mesh generated at the actual (r, z, a+da).

    Two fresh-mesh variants are compared:
      A) "scaled" — local element size scales with the actual a (legacy
         behaviour: each fresh mesh has a different local resolution).
      B) "fixed"  — local element size held at the *original* reference
         (particle_maxh_rel * a_hat), so the discretization matches the
         moving mesh exactly.  Variant B isolates pure deformation error.
    """
    if da_test_values is None:
        da_test_values = [0.0, 0.005, 0.01, 0.02, -0.005, -0.01, -0.02]

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 4: Moving mesh error vs remeshing")
    print("  (A = fresh mesh with a-scaled resolution,")
    print("   B = fresh mesh with FIXED reference resolution)")
    print("=" * 70)
    header = (f"\n  {'da':>10s}  {'|F_move|':>12s}  "
              f"{'|F_freshA|':>12s}  {'rel_errA':>10s}  "
              f"{'|F_freshB|':>12s}  {'rel_errB':>10s}")
    print(header)

    for da in da_test_values:
        a_cur = a_hat + da

        # Moving mesh evaluation
        try:
            F_move = evaluate_forces_hat(0.0, 0.0, da, md_orig, compute_jacobian=False)
        except MeshFlippedError:
            print(f"  {da:10.4f}  MESH FLIPPED (moving)")
            continue

        # Fresh mesh A: legacy — local mesh size scales with a_cur
        try:
            md_freshA = setup_moving_mesh_hat(
                r_eq, z_eq, a_cur,
                R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d)
            F_freshA = evaluate_forces_hat(0.0, 0.0, 0.0, md_freshA, compute_jacobian=False)
            diffA = np.linalg.norm(F_move - F_freshA)
            relA = diffA / (np.linalg.norm(F_freshA) + 1e-30)
        except Exception as e:
            print(f"  {da:10.4f}  {np.linalg.norm(F_move):12.4e}  FRESH-A ERROR: {e}")
            continue

        # Fresh mesh B: same particle resolution as the original moving mesh
        try:
            md_freshB = setup_moving_mesh_hat(
                r_eq, z_eq, a_cur,
                R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d,
                a_mesh_size_ref=a_hat)
            F_freshB = evaluate_forces_hat(0.0, 0.0, 0.0, md_freshB, compute_jacobian=False)
            diffB = np.linalg.norm(F_move - F_freshB)
            relB = diffB / (np.linalg.norm(F_freshB) + 1e-30)
        except Exception as e:
            print(f"  {da:10.4f}  {np.linalg.norm(F_move):12.4e}  "
                  f"{np.linalg.norm(F_freshA):12.4e}  {relA:10.4e}  "
                  f"FRESH-B ERROR: {e}")
            continue

        print(f"  {da:10.4f}  {np.linalg.norm(F_move):12.4e}  "
              f"{np.linalg.norm(F_freshA):12.4e}  {relA:10.4e}  "
              f"{np.linalg.norm(F_freshB):12.4e}  {relB:10.4e}")


def diag_tlm_vs_fd_dJphi(dr, dz, da, md, phi, eps=1e-4):
    """Test 5: Compare FD-of-AD-Jacobian vs FD-of-TLM for dJphi.

    Method A: dJphi/dr = (J(r+eps)·phi - J(r-eps)·phi) / (2·eps)  [FD of reverse-AD]
    Method B: dJphi/dr = (Jphi(r+eps) - Jphi(r-eps)) / (2·eps)    [FD of TLM]

    Both should give the same result, but may differ due to noise.
    """
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 5: FD-of-AD vs FD-of-TLM for dJphi")
    print("=" * 70)

    # Method A: FD of reverse-AD Jacobian
    _, J_rp = evaluate_forces_hat(dr + eps, dz, da, md)
    _, J_rm = evaluate_forces_hat(dr - eps, dz, da, md)
    dJphi_dr_A = (J_rp[:, :2] @ phi - J_rm[:, :2] @ phi) / (2 * eps)

    _, J_zp = evaluate_forces_hat(dr, dz + eps, da, md)
    _, J_zm = evaluate_forces_hat(dr, dz - eps, da, md)
    dJphi_dz_A = (J_zp[:, :2] @ phi - J_zm[:, :2] @ phi) / (2 * eps)

    # Method B: FD of TLM
    _, Jphi_rp = evaluate_forces_tlm_hat(dr + eps, dz, da, md, phi)
    _, Jphi_rm = evaluate_forces_tlm_hat(dr - eps, dz, da, md, phi)
    dJphi_dr_B = (Jphi_rp - Jphi_rm) / (2 * eps)

    _, Jphi_zp = evaluate_forces_tlm_hat(dr, dz + eps, da, md, phi)
    _, Jphi_zm = evaluate_forces_tlm_hat(dr, dz - eps, da, md, phi)
    dJphi_dz_B = (Jphi_zp - Jphi_zm) / (2 * eps)

    print(f"\n  eps = {eps:.1e}")
    print(f"\n  dJphi/dr:")
    print(f"    Method A (FD of reverse-AD): [{dJphi_dr_A[0]:+.6e}, {dJphi_dr_A[1]:+.6e}]")
    print(f"    Method B (FD of TLM):        [{dJphi_dr_B[0]:+.6e}, {dJphi_dr_B[1]:+.6e}]")
    print(f"    |A - B| = {np.linalg.norm(dJphi_dr_A - dJphi_dr_B):.4e}")

    print(f"\n  dJphi/dz:")
    print(f"    Method A (FD of reverse-AD): [{dJphi_dz_A[0]:+.6e}, {dJphi_dz_A[1]:+.6e}]")
    print(f"    Method B (FD of TLM):        [{dJphi_dz_B[0]:+.6e}, {dJphi_dz_B[1]:+.6e}]")
    print(f"    |A - B| = {np.linalg.norm(dJphi_dz_A - dJphi_dz_B):.4e}")


def diag_jacobian_noise(dr, dz, da, md, n_samples=5):
    """Test 6: Measure AD Jacobian noise by repeated evaluation.

    Calls evaluate_forces_hat multiple times at the exact same point
    and measures the variation.  Any variation is due to floating-point
    non-determinism in the PDE solve / assembly.
    """
    print("\n" + "=" * 70)
    print("  DIAGNOSTIC 6: AD Jacobian repeatability (noise floor)")
    print("=" * 70)

    Js = []
    Fs = []
    for i in range(n_samples):
        F, J = evaluate_forces_hat(dr, dz, da, md)
        Fs.append(F)
        Js.append(J)

    Fs = np.array(Fs)
    Js = np.array(Js)

    F_std = np.std(Fs, axis=0)
    J_std = np.std(Js, axis=0)

    print(f"\n  {n_samples} evaluations at (dr={dr}, dz={dz}, da={da}):")
    print(f"  F mean: [{Fs.mean(0)[0]:+.8e}, {Fs.mean(0)[1]:+.8e}]")
    print(f"  F std:  [{F_std[0]:.4e}, {F_std[1]:.4e}]")
    print(f"\n  J mean:")
    for i in range(2):
        print(f"    [{' '.join(f'{Js.mean(0)[i,j]:+.8e}' for j in range(3))}]")
    print(f"  J std:")
    for i in range(2):
        print(f"    [{' '.join(f'{J_std[i,j]:.4e}' for j in range(3))}]")
    print(f"\n  -> FD with eps=h gives noise/h in dJphi.")
    print(f"     At eps=1e-4, J noise {J_std.max():.2e} → dJphi noise ~ {J_std.max()/1e-4:.2e}")
    print(f"     At eps=1e-5, J noise {J_std.max():.2e} → dJphi noise ~ {J_std.max()/1e-5:.2e}")


def run_all_diagnostics(r_eq, z_eq, a_hat, shared_data):

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    print("\n" + "#" * 70)
    print("#  MOORE-SPENCE CONVERGENCE DIAGNOSTICS")
    print("#" * 70)

    # Setup mesh at equilibrium
    md = setup_moving_mesh_hat(r_eq, z_eq, a_hat,
                                R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d)

    # Get initial Jacobian and null vector
    F0, J0_full = evaluate_forces_hat(0.0, 0.0, 0.0, md)
    J0 = J0_full[:, :2]
    phi = estimate_null_vector(J0)
    l_vec = phi.copy()
    phi = phi / np.dot(l_vec, phi)

    print(f"\n  Base point: r={r_eq:.8f}, z={z_eq:.8f}, a={a_hat:.6f}")
    print(f"  |F| = {np.linalg.norm(F0):.4e}")
    print(f"  phi = ({phi[0]:.8f}, {phi[1]:.8f})")
    print(f"  |J·phi| = {np.linalg.norm(J0 @ phi):.4e}")

    # Test 6 first (cheapest, most informative)
    diag_jacobian_noise(0.0, 0.0, 0.0, md, n_samples=5)

    # Test 2: AD vs FD Jacobian
    diag_ad_vs_fd_jacobian(0.0, 0.0, 0.0, md, eps=1e-5)

    # Test 1: FD convergence study
    diag_fd_convergence_study(0.0, 0.0, 0.0, md, phi,
                              eps_values=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])

    # Test 3: a-dependence
    diag_a_dependence(0.0, 0.0, md)

    # Test 4: Moving mesh error
    diag_moving_mesh_error(r_eq, z_eq, a_hat, shared_data, md,
                           da_test_values=[0.0, 0.005, 0.01, 0.02, -0.005, -0.01, -0.02])

    # Test 5: FD-of-AD vs FD-of-TLM
    diag_tlm_vs_fd_dJphi(0.0, 0.0, 0.0, md, phi, eps=1e-4)

    # Test 7: AD Hessian-vector product vs FD-of-AD
    diag_hessian_vs_fd(0.0, 0.0, 0.0, md, phi, eps=1e-4)

    print("\n" + "#" * 70)
    print("#  END DIAGNOSTICS")
    print("#" * 70)


if __name__ == "__main__":

    r_off_hat_init = 0.61558964
    z_off_hat_init = 0.00176900
    a_hat_init = 0.135

    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)

    G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat)

    # ============================================================
    # Phase 1: Find equilibrium on initial mesh.
    # ============================================================
    r_hat, z_hat, converged, md = newton_root_refine_hat(
        r_off_hat_init, z_off_hat_init, a_hat_init, shared_data, tol=1e-10, max_iter=15)
    if not converged:
        raise ValueError("Newton refinement did not converge.")

    # ============================================================
    # Sanity check: remesh at found equilibrium and evaluate |F|.
    # ============================================================
    print("=" * 65)
    print("  Remesh sanity check at converged equilibrium")
    print("=" * 65)
    R_hat, H_hat, W_hat = shared_data[:3]
    md_check = setup_moving_mesh_hat(r_hat, z_hat, a_hat_init,
                                     R_hat, H_hat, W_hat, Re, G_hat, U_m_hat,
                                     u_bar_2d_hat, p_bar_tilde_2d_hat)
    F_check = evaluate_forces_hat(0.0, 0.0, 0.0, md_check, compute_jacobian=False)
    print(f"  |F| on fresh mesh at (r={r_hat:.8f}, z={z_hat:.8f}): {np.linalg.norm(F_check):.4e}")
    print(f"  F = ({F_check[0]:.6e}, {F_check[1]:.6e})\n")

    # ============================================================
    # DIAGNOSTICS: Run before Moore-Spence to identify issues.
    # Comment out once the problem is understood.
    # ============================================================
    run_all_diagnostics(r_hat, z_hat, a_hat_init, shared_data)

    # ============================================================
    # Phase 2: Bifurcation detection (hat system).
    # Uses the fresh mesh at the equilibrium; a changes via moving
    # mesh from here — no further remeshing.
    #
    # Four switches are available:
    #   'moore_spence_ad'      — Moore-Spence with AD-based Hessian
    #                            (1 forward + 2 reverse + 2 H·v per iter).
    #                            No FD anywhere — Newton-like convergence
    #                            if the tape is twice differentiable.
    #   'moore_spence_broyden' — Moore-Spence with Broyden rank-1 update
    #                            of the 5x5 DG (cheap, superlinear).
    #   'moore_spence_exact'   — Moore-Spence with exact FD rebuild of DG
    #                            in every iteration (~7 AD evals/iter).
    #   'bisection'            — pure bisection on a_hat with an inner
    #                            Newton on (r, z); no second derivatives.
    # ============================================================
    BIFURCATION_METHOD = 'moore_spence_ad'

    if BIFURCATION_METHOD == 'moore_spence_ad':
        r_bif, z_bif, a_bif, phi_bif, ms_converged = moore_spence_solve_hat_ad(
            r_hat, z_hat, a_hat_init, shared_data, tol=1e-8, max_iter=20)

        if not ms_converged:
            print("\n  WARNING: Moore-Spence (AD) did not converge.")
        else:
            a_phys_bif = a_bif * L_c
            print(f"\n  Bifurcation point summary:")
            print(f"        r_off_hat = {r_bif:.10f}")
            print(f"        z_off_hat = {z_bif:.10f}")
            print(f"        a_hat     = {a_bif:.10f}  (a = {a_phys_bif * 1e6:.4f} um)")
            print(f"        phi       = ({phi_bif[0]:.8f}, {phi_bif[1]:.8f})")
    elif BIFURCATION_METHOD in ('moore_spence_broyden', 'moore_spence_exact'):
        _ms_update = ('broyden' if BIFURCATION_METHOD == 'moore_spence_broyden'
                      else 'exact')
        r_bif, z_bif, a_bif, phi_bif, ms_converged = moore_spence_solve_hat(
             r_hat, z_hat, a_hat_init, shared_data, tol=1e-8, max_iter=20, eps_fd=1e-4,
             jacobian_update=_ms_update)

        if not ms_converged:
            print("\n  WARNING: Moore-Spence did not converge.")
        else:
             a_phys_bif = a_bif * L_c
             print(f"\n  Bifurcation point summary:")
             print(f"        r_off_hat = {r_bif:.10f}")
             print(f"        z_off_hat = {z_bif:.10f}")
             print(f"        a_hat     = {a_bif:.10f}  (a = {a_phys_bif * 1e6:.4f} um)")
             print(f"        phi       = ({phi_bif[0]:.8f}, {phi_bif[1]:.8f})")
    elif BIFURCATION_METHOD == 'bisection':
        # Bracket on a_hat. One endpoint should sit on the "equilibrium
        # exists" side, the other beyond the bifurcation. Adjust to taste.
        a_hat_lo = 0.13
        a_hat_hi = 0.14
        r_bif, z_bif, a_bif, bis_converged = bisection_bifurcation_hat(
            r_hat, z_hat, a_hat_init, a_hat_lo, a_hat_hi, shared_data,
            tol_a=1e-6, max_bisect=30,
            newton_tol=1e-10, newton_max_iter=15)
        phi_bif = None

        if not bis_converged:
            print("\n  WARNING: Bisection did not converge.")
        else:
            a_phys_bif = a_bif * L_c
            print(f"\n  Bifurcation point summary:")
            print(f"        r_off_hat = {r_bif:.10f}")
            print(f"        z_off_hat = {z_bif:.10f}")
            print(f"        a_hat     = {a_bif:.10f}  (a = {a_phys_bif * 1e6:.4f} um)")
    else:
        raise ValueError(
            f"Unknown BIFURCATION_METHOD: {BIFURCATION_METHOD!r}. "
            "Use 'moore_spence_ad', 'moore_spence_broyden', "
            "'moore_spence_exact', or 'bisection'.")