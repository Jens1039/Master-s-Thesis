import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import math
from copy import deepcopy

from firedrake import *
from firedrake.adjoint import stop_annotating, continue_annotation
from pyadjoint import set_working_tape, get_working_tape, Tape, ReducedFunctional, Control

from nondimensionalization import first_nondimensionalisation, second_nondimensionalisation
from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable
from perturbed_flow_return_UFL_experimental import perturbed_flow_differentiable
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole
from config_paper_parameters import R, H, W, Q, rho, mu, particle_maxh_rel, global_maxh_rel


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

    J_00 = float(dFx[0].dat.data_ro[0])
    J_01 = float(dFx[1].dat.data_ro[0])

    dFz = Jhat_z.derivative()

    J_10 = float(dFz[0].dat.data_ro[0])
    J_11 = float(dFz[1].dat.data_ro[0])

    J = np.array([[J_00, J_01],
                  [J_10, J_11]])

    stop_annotating()
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
            F_trial = evaluate_forces(dr_trial, dz_trial, mesh_data, compute_jacobian=False)
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

            F_try, J_try, _ = _moore_spence_eval(r_try, z_try, a_try, shared_data)
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


if __name__ == "__main__":

    r_off_hat_init = 0.61
    z_off_hat_init = 0.0
    a_hat_init = 0.135

    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)

    G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat)

    a_phys = a_hat_init * L_c

    (R_hh, H_hh, W_hh, a_hh, _, L_c_p, _, _, _, _) = second_nondimensionalisation(R_hat, H_hat, W_hat, a_phys, L_c, U_c, G_hat, Re,
                                                                                  u_bar_2d_hat, p_bar_tilde_2d_hat, U_m_hat, print_values=False)

    # ============================================================
    # Iterative equilibrium refinement with mesh rebuilds
    # Phase 1 finds equilibrium via mesh displacement, but when
    # the mesh is rebuilt at the found position, |F| != 0.
    # We iterate until the equilibrium is mesh-consistent.
    # ============================================================
    r_hat, z_hat = r_off_hat_init, z_off_hat_init
    for outer in range(10):
        r_hat_new, z_hat_new, converged, mesh_data = newton_root_refine(
            r_hat, z_hat, a_hat_init, shared_data, tol=1e-10, max_iter=15)
        if not converged:
            raise ValueError("Newton refinement did not converge.")
        shift = abs(r_hat_new - r_hat) + abs(z_hat_new - z_hat)
        print(f"  Outer iter {outer}: r_hat={r_hat_new:.10f}, z_hat={z_hat_new:.10f}, shift={shift:.4e}")
        r_hat, z_hat = r_hat_new, z_hat_new
        if shift < 1e-8:
            print(f"  -> Equilibrium mesh-consistent after {outer+1} outer iterations.\n")
            break
    else:
        print("  WARNING: outer equilibrium loop did not converge.\n")
    '''
    # ============================================================
    # PHASE 2: Moore-Spence bifurcation refinement
    # ============================================================

    r_bif, z_bif, a_bif, phi_bif, ms_converged = moore_spence_solve(
        r_hat, z_hat, a_hat_init, shared_data, tol=1e-8, max_iter=20, eps_fd=1e-5)

    if not ms_converged:
        print("\n  WARNING: Moore-Spence did not converge.")
    else:
        a_phys_bif = a_bif * L_c
        print(f"\n  Bifurcation point summary:")
        print(f"    r_off_hat = {r_bif:.10f}")
        print(f"    z_off_hat = {z_bif:.10f}")
        print(f"    a_hat     = {a_bif:.10f}  (a = {a_phys_bif * 1e6:.4f} um)")
        print(f"    phi       = ({phi_bif[0]:.8f}, {phi_bif[1]:.8f})")

    '''
    # ============================================================
    # DIAGNOSTIC TESTS: Moore-Spence DG Jacobian verification
    # ============================================================

    r, z, a = r_hat, z_hat, a_hat_init

    print("\n" + "=" * 65)
    print("  DIAGNOSTIC: Building base evaluation at equilibrium")
    print("=" * 65)

    F_base, J_base, md_base = _moore_spence_eval(r, z, a, shared_data)
    scale = md_base['scale']

    phi = estimate_null_vector(J_base)
    l_vec = phi.copy()
    phi = phi / np.dot(l_vec, phi)

    G1 = F_base
    G2 = J_base @ phi
    G3_val = np.dot(l_vec, phi) - 1.0
    G_base = np.concatenate([G1, G2, [G3_val]])

    print(f"  |F_base|   = {np.linalg.norm(F_base):.6e}")
    print(f"  |J*phi|    = {np.linalg.norm(G2):.6e}")
    print(f"  |G_base|   = {np.linalg.norm(G_base):.6e}")
    print(f"  phi        = {phi}")
    print(f"  scale      = {scale:.4f}")

    # ------------------------------------------------------------------
    # Helper: compute G at perturbed state
    # ------------------------------------------------------------------
    def compute_G_spatial(dr_hh, dz_hh, phi_loc):
        """Compute G using mesh displacement (same mesh, no re-meshing)."""
        F_p, J_p = evaluate_forces(dr_hh, dz_hh, md_base, compute_jacobian=True)
        g1 = F_p
        g2 = J_p @ phi_loc
        g3 = np.dot(l_vec, phi_loc) - 1.0
        return np.concatenate([g1, g2, [g3]]), F_p, J_p

    def compute_G_with_a(r_loc, z_loc, a_loc, phi_loc):
        """Compute G with re-meshing (for a perturbation)."""
        F_p, J_p, _ = _moore_spence_eval(r_loc, z_loc, a_loc, shared_data)
        g1 = F_p
        g2 = J_p @ phi_loc
        g3 = np.dot(l_vec, phi_loc) - 1.0
        return np.concatenate([g1, g2, [g3]]), F_p, J_p

    # ------------------------------------------------------------------
    # TEST 1 & 2: DG column verification with multiple eps values
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  TEST 1 & 2: DG column verification (FD of full G vs DG columns)")
    print("=" * 65)

    eps_values = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

    for eps_fd in eps_values:

        print(f"\n  --- eps_fd = {eps_fd:.0e} ---")

        # Assemble DG exactly like moore_spence_solve does
        eps_hh = eps_fd

        _, J_rp = evaluate_forces(eps_hh, 0.0, md_base, compute_jacobian=True)
        dJphi_dr = (J_rp @ phi - J_base @ phi) / eps_hh

        _, J_zp = evaluate_forces(0.0, eps_hh, md_base, compute_jacobian=True)
        dJphi_dz = (J_zp @ phi - J_base @ phi) / eps_hh

        eps_a = eps_fd * max(abs(a), 1e-4)
        F_ap, J_ap, _ = _moore_spence_eval(r, z, a + eps_a, shared_data)
        dF_da = (F_ap - F_base) / eps_a
        dJphi_da = (J_ap @ phi - J_base @ phi) / eps_a

        DG = np.zeros((5, 5))
        DG[0:2, 0:2] = J_base
        DG[0:2, 2]   = dF_da
        DG[2:4, 0]   = dJphi_dr
        DG[2:4, 1]   = dJphi_dz
        DG[2:4, 2]   = dJphi_da
        DG[2:4, 3:5] = J_base
        DG[4, 3:5]   = l_vec

        # Now verify each column via FD of the full residual G
        col_names = ["dr_hh", "dz_hh", "da", "dphi0", "dphi1"]

        for col_idx in range(5):
            eps_col = eps_fd

            if col_idx == 0:  # perturb r_hh
                G_pert, _, _ = compute_G_spatial(eps_col, 0.0, phi)
                dG_fd = (G_pert - G_base) / eps_col
            elif col_idx == 1:  # perturb z_hh
                G_pert, _, _ = compute_G_spatial(0.0, eps_col, phi)
                dG_fd = (G_pert - G_base) / eps_col
            elif col_idx == 2:  # perturb a (re-mesh)
                eps_a_col = eps_col * max(abs(a), 1e-4)
                G_pert, _, _ = compute_G_with_a(r, z, a + eps_a_col, phi)
                dG_fd = (G_pert - G_base) / eps_a_col
            elif col_idx == 3:  # perturb phi[0]
                phi_pert = phi.copy()
                phi_pert[0] += eps_col
                G_pert, _, _ = compute_G_spatial(0.0, 0.0, phi_pert)
                dG_fd = (G_pert - G_base) / eps_col
            elif col_idx == 4:  # perturb phi[1]
                phi_pert = phi.copy()
                phi_pert[1] += eps_col
                G_pert, _, _ = compute_G_spatial(0.0, 0.0, phi_pert)
                dG_fd = (G_pert - G_base) / eps_col

            dG_analytic = DG[:, col_idx]
            abs_err = np.linalg.norm(dG_fd - dG_analytic)
            rel_err = abs_err / (np.linalg.norm(dG_analytic) + 1e-30)

            print(f"    col {col_idx} ({col_names[col_idx]:6s}): "
                  f"|DG_fd - DG_analytic| = {abs_err:.4e}  "
                  f"rel = {rel_err:.4e}")
            if rel_err > 0.1:
                print(f"      DG_analytic = {dG_analytic}")
                print(f"      DG_fd       = {dG_fd}")

        print(f"    cond(DG) = {np.linalg.cond(DG):.2e}")

    # ------------------------------------------------------------------
    # TEST 3: Predicted vs actual decrease (one Newton step)
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  TEST 3: Predicted vs actual decrease (single Newton step)")
    print("=" * 65)

    # Use best eps from above (we'll use 1e-5 as default)
    eps_fd = 1e-5
    eps_hh = eps_fd

    _, J_rp = evaluate_forces(eps_hh, 0.0, md_base, compute_jacobian=True)
    dJphi_dr = (J_rp @ phi - J_base @ phi) / eps_hh

    _, J_zp = evaluate_forces(0.0, eps_hh, md_base, compute_jacobian=True)
    dJphi_dz = (J_zp @ phi - J_base @ phi) / eps_hh

    eps_a = eps_fd * max(abs(a), 1e-4)
    F_ap, J_ap, _ = _moore_spence_eval(r, z, a + eps_a, shared_data)
    dF_da = (F_ap - F_base) / eps_a
    dJphi_da = (J_ap @ phi - J_base @ phi) / eps_a

    DG = np.zeros((5, 5))
    DG[0:2, 0:2] = J_base
    DG[0:2, 2]   = dF_da
    DG[2:4, 0]   = dJphi_dr
    DG[2:4, 1]   = dJphi_dz
    DG[2:4, 2]   = dJphi_da
    DG[2:4, 3:5] = J_base
    DG[4, 3:5]   = l_vec

    dy = np.linalg.solve(DG, -G_base)

    predicted_G = DG @ dy + G_base  # should be ~0
    print(f"  |G_base|       = {np.linalg.norm(G_base):.6e}")
    print(f"  |predicted residual| (DG*dy + G, should be ~0) = {np.linalg.norm(predicted_G):.6e}")
    print(f"  dy = {dy}")
    print(f"  dy in hat coords: dr_hat={dy[0]/scale:.6e}, dz_hat={dy[1]/scale:.6e}, da={dy[2]:.6e}")

    # Actual G at full step (alpha=1)
    r_new = r + dy[0] / scale
    z_new = z + dy[1] / scale
    a_new = a + dy[2]
    phi_new = phi + dy[3:5]

    G_actual_full, _, _ = compute_G_with_a(r_new, z_new, a_new, phi_new)
    ratio_full = np.linalg.norm(G_actual_full) / np.linalg.norm(G_base)

    print(f"\n  Full step (alpha=1.0):")
    print(f"    |G_new|      = {np.linalg.norm(G_actual_full):.6e}")
    print(f"    |G_new|/|G_base| = {ratio_full:.6e}  (quadratic: should be << 1)")
    print(f"    G_new        = {G_actual_full}")

    # Also test half step
    for alpha_test in [0.5, 0.25, 0.125]:
        r_t = r + alpha_test * dy[0] / scale
        z_t = z + alpha_test * dy[1] / scale
        a_t = a + alpha_test * dy[2]
        phi_t = phi + alpha_test * dy[3:5]
        G_t, _, _ = compute_G_with_a(r_t, z_t, a_t, phi_t)
        print(f"    alpha={alpha_test:.4f}: |G| = {np.linalg.norm(G_t):.6e}  "
              f"ratio = {np.linalg.norm(G_t)/np.linalg.norm(G_base):.4e}")

    print("\n" + "=" * 65)
    print("  DIAGNOSTIC TESTS COMPLETE")
    print("=" * 65)

    # ------------------------------------------------------------------
    # TEST 4: Continuation scan — sigma_min(J) vs a_hat
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  TEST 4: Continuation scan — sigma_min(J) vs a_hat")
    print("=" * 65)

    a_scan_values = [0.13, 0.131, 0.132, 0.133, 0.134, 0.135, 0.136, 0.137, 0.138, 0.139, 0.14]

    # Use last converged equilibrium as initial guess, update as we go
    r_guess = r_off_hat_init
    z_guess = z_off_hat_init

    scan_results = []

    for a_test in a_scan_values:
        r_eq, z_eq, conv, _ = newton_root_refine(
            r_guess, z_guess, a_test, shared_data, tol=1e-8, max_iter=15)

        if conv:
            _, J_eq, _ = _moore_spence_eval(r_eq, z_eq, a_test, shared_data)
            sv = np.linalg.svd(J_eq, compute_uv=False)
            sigma_min = sv.min()
            sigma_max = sv.max()
            ratio = sigma_min / sigma_max

            print(f"  a_hat={a_test:.4f} | r={r_eq:.6f} z={z_eq:.6f} "
                  f"| sigma_min={sigma_min:.6e} sigma_max={sigma_max:.6e} "
                  f"| ratio={ratio:.6e}")

            scan_results.append((a_test, r_eq, z_eq, sigma_min, sigma_max))

            # Use converged point as next initial guess (continuation)
            r_guess = r_eq
            z_guess = z_eq
        else:
            print(f"  a_hat={a_test:.4f} | Newton did not converge")

    # Summary
    if scan_results:
        print("\n  --- Summary ---")
        print(f"  {'a_hat':>8s}  {'sigma_min':>12s}  {'sigma_max':>12s}  {'ratio':>12s}")
        for a_val, r_val, z_val, smin, smax in scan_results:
            print(f"  {a_val:8.4f}  {smin:12.6e}  {smax:12.6e}  {smin/smax:12.6e}")

        # Find a_hat with smallest sigma_min
        best = min(scan_results, key=lambda x: x[3])
        print(f"\n  Smallest sigma_min at a_hat = {best[0]:.4f} "
              f"(sigma_min = {best[3]:.6e}, r = {best[1]:.6f}, z = {best[2]:.6f})")
        print(f"  -> Use this as starting point for Moore-Spence")