import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import math
from copy import deepcopy

from firedrake import *
from firedrake.adjoint import stop_annotating, continue_annotation
from pyadjoint import set_working_tape, Tape, ReducedFunctional, Control

from nondimensionalization import first_nondimensionalisation, second_nondimensionalisation
from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable
from perturbed_flow_return_UFL import perturbed_flow_differentiable
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
        F_fd, J_fd = evaluate_forces_and_fd_jacobian(delta_r_hh, delta_z_hh, mesh_data, eps=1e-5)

        print(f"  AD  Jacobian:\n    {J}")
        print(f"  FD  Jacobian:\n    {J_fd}")
        print(f"  Difference (AD - FD):\n    {J - J_fd}")
        print(f"  Relative diff: {np.linalg.norm(J - J_fd) / (np.linalg.norm(J_fd) + 1e-30):.4e}")

        dx_ad = np.linalg.solve(J, -F)
        dx_fd = np.linalg.solve(J_fd, -F)
        print(f"  Newton step (AD):  {dx_ad}")
        print(f"  Newton step (FD):  {dx_fd}")
        print(f"  F·dx_ad = {np.dot(F, dx_ad):.6e}  (sollte < 0 für Abstieg)")
        print(f"  F·dx_fd = {np.dot(F, dx_fd):.6e}")

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

        dx = np.linalg.solve(J, -F)

        # Armijo step size
        alpha = 1
        for ls_step in range(10):
            dr_trial = delta_r_hh + alpha * dx[0]
            dz_trial = delta_z_hh + alpha * dx[1]
            F_trial = evaluate_forces(dr_trial, dz_trial, mesh_data, compute_jacobian=False)
            if np.linalg.norm(F_trial) < (1.0 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5
        else:
            print(f"Line-Search: alpha = {alpha:.4f} (minimum reached)")

        delta_r_hh += alpha * dx[0]
        delta_z_hh += alpha * dx[1]

    r_off_hat = r_off_hat_init + delta_r_hh / scale
    z_off_hat = z_off_hat_init + delta_z_hh / scale
    print(f"Did not converge after {max_iter} iterations.\n")
    return r_off_hat, z_off_hat, False, mesh_data


def moore_spence(r_off_hat, z_off_hat, a_hat, phi, l_vec, shared_data, *, tol=1e-8, max_iter=20, eps_fd=1e-5):

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

    r_hat, z_hat, converged, mesh_data = newton_root_refine(r_off_hat_init, z_off_hat_init, a_hat_init, shared_data, tol=1e-10, max_iter=15)

    if not converged:
        raise ValueError("Newton refinement did not converge.")