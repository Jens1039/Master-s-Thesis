import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import gc
import numpy as np

from nondimensionalization import first_nondimensionalisation
from background_flow_differentiable import background_flow_differentiable
from perturbed_flow_differentiable import MeshFlippedError, check_mesh_quality, setup_moving_mesh_hat, evaluate_forces, estimate_eigenvectors, _build_xi_hat
from config_paper_parameters import R, H, W, Q, rho, mu, particle_maxh_rel, global_maxh_rel


def newton_root_refine(r_off_hat_init, z_off_hat_init, a_hat, shared_data, *, tol=1e-14, max_iter=15):

    print("\n" + "=" * 65)
    print(f"  Newton Root Refinement (a_hat = {a_hat:.6f})")
    print("=" * 65)

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    md = setup_moving_mesh_hat(r_off_hat_init, z_off_hat_init, a_hat, R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d,
                               particle_maxh_rel, global_maxh_rel)

    dr, dz = 0.0, 0.0

    for k in range(max_iter):
        F, J_full = evaluate_forces(dr, dz, 0.0, md)
        J = J_full[:, :2]

        res = np.linalg.norm(F)
        r_cur = r_off_hat_init + dr
        z_cur = z_off_hat_init + dz

        print(f"  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f} | |F| = {res:.4e}  cond(J) = {np.linalg.cond(J):.2e}")

        if res < tol:
            print(f"  -> Converged after {k} iterations.\n")
            return r_cur, z_cur, md, dr, dz

        dx_n = np.linalg.solve(J, -F)

        alpha = 1.0
        for ls in range(10):
            try:
                F_try = evaluate_forces(dr + alpha * dx_n[0], dz + alpha * dx_n[1], 0.0,
                                            md, jacobian=False)
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
    print(f"  WARNING: Newton refinement did not converge at a_hat = {a_hat} "
          f"after {max_iter} iterations (|F| = {res:.4e}). Continuing anyway.")
    return r_cur, z_cur, md, dr, dz


def _ms_trial(dr_init, dz_init, da_init, phi_start, l_vec, md, r_ref, z_ref, a_ref, L_c, tol, max_iter):
    """Single Moore-Spence trial with a given starting phi.

    Uses trust-region with backtracking (shrink Delta on reject).
    Returns (r, z, a, phi, converged, final_residual).
    """

    dr, dz, da = float(dr_init), float(dz_init), float(da_init)
    phi = phi_start.copy()
    Delta, Delta_max = 1e-2, 1.0
    eta_accept, eta_good = 0.1, 0.75
    res = float('inf')

    for k in range(max_iter):

        # All entries of DG via AD: 1 forward + 2 reverse + 2 H·v
        F_base, J_full, dJphi_dx = evaluate_forces(
            dr, dz, da, md, hessian_phi=phi)
        J_sp = J_full[:, :2]
        dF_da = J_full[:, 2]

        G1 = F_base
        G2 = J_sp @ phi
        G3_val = np.dot(l_vec, phi) - 1.0
        G = np.concatenate([G1, G2, [G3_val]])
        res = np.linalg.norm(G)

        eigs = np.linalg.eigvals(J_sp)
        eigenvalue = eigs[np.argmin(np.abs(eigs))]
        r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
        print(f"\n  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f}  a = {a_cur:.8f}")
        print(f"         | |G| = {res:.4e}  |F| = {np.linalg.norm(G1):.4e}"
              f"  |J*phi| = {np.linalg.norm(G2):.4e}"
              f"  eigenvalue = {eigenvalue:+.4e}")

        if res < tol:
            a_phys = a_cur * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations!")
            print(f"     r_off_hat = {r_cur:.10f}")
            print(f"     z_off_hat = {z_cur:.10f}")
            print(f"     a_hat     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = ({phi[0]:.8f}, {phi[1]:.8f})")
            print(f"     eigenvalue = {eigenvalue:+.6e}")
            return r_cur, z_cur, a_cur, phi, True, res

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
            return r_cur, z_cur, a_cur, phi, False, res

        try:
            dy = np.linalg.solve(DG, -G)
        except np.linalg.LinAlgError:
            print("  !! DG singular — aborting.")
            return r_cur, z_cur, a_cur, phi, False, res

        print(f"         | step: dr={dy[0]:+.4e}  dz={dy[1]:+.4e}  da={dy[2]:+.4e}")

        # ── Helper: evaluate |G| at a candidate step (cheap: only F and J) ──

        def _eval_step(dy_candidate):
            dr_t = dr + dy_candidate[0]
            dz_t = dz + dy_candidate[1]
            da_t = da + dy_candidate[2]
            phi_t = phi + dy_candidate[3:5]
            if a_ref + da_t <= 0:
                return float('nan'), False
            try:
                F_t, J_t = evaluate_forces(dr_t, dz_t, da_t, md)
                G_t = np.concatenate([F_t, J_t[:, :2] @ phi_t,
                                      [np.dot(l_vec, phi_t) - 1.0]])
                return float(np.linalg.norm(G_t)), True
            except MeshFlippedError:
                return float('nan'), False

        def _compute_rho(dy_try, res_try, step_ok):
            G_pred = G + DG @ dy_try
            pred = res**2 - np.linalg.norm(G_pred)**2
            if step_ok and abs(pred) > 1e-30:
                return (res**2 - res_try**2) / pred
            return -1.0 if not step_ok else (1.0 if res_try <= res else -1.0)

        accepted, dy_try, rho, Delta = _globalize_tr(dy, Delta, eta_accept, eta_good, Delta_max, _eval_step, _compute_rho)

        if accepted:
            dr += dy_try[0]
            dz += dy_try[1]
            da += dy_try[2]
            phi_new = phi + dy_try[3:5]
            phi = phi_new / np.dot(l_vec, phi_new)
            print(f"         | step ACCEPTED")
        else:
            print(f"         | step REJECTED")

        gc.collect()

    r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
    return r_cur, z_cur, a_cur, phi, False, res


def _globalize_tr(dy, Delta, eta_accept, eta_good, Delta_max, _eval_step, _compute_rho):
    """Trust-region with backtracking: shrink Delta until step is accepted.

    The Newton direction dy is computed once (expensive).  The inner loop
    only clips and re-evaluates (cheap: only F and J per attempt).
    Returns (accepted, dy_try, rho, Delta_updated).
    """

    for attempt in range(20):
        step_norm_phys = np.linalg.norm(dy[:3])
        at_boundary = step_norm_phys > Delta
        if at_boundary:
            dy_try = dy * (Delta / step_norm_phys)
        else:
            dy_try = dy.copy()

        res_try, step_ok = _eval_step(dy_try)
        rho = _compute_rho(dy_try, res_try, step_ok)

        print(f"         | TR-BT attempt {attempt}: |G_try|={res_try if step_ok else float('nan'):.4e}"
              f"  rho={rho:+.4f}  Delta={Delta:.4e}")

        if rho >= eta_accept and step_ok:
            if rho > eta_good and at_boundary:
                Delta = min(Delta * 2.0, Delta_max)
            return True, dy_try, rho, Delta

        Delta *= 0.5
        if Delta < 1e-12:
            print("         | TR-BT: Delta too small — giving up.")
            break

    return False, dy.copy(), -1.0, Delta


def moore_spence_solve(r_off_hat_eq, z_off_hat_eq, a_hat_init, shared_data, *, tol=1e-14, max_iter=20, md=None,
                       dr_init=0.0, dz_init=0.0, da_init=0.0):

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data
    r_ref = float(r_off_hat_eq) - float(dr_init)
    z_ref = float(z_off_hat_eq) - float(dz_init)
    a_ref = float(a_hat_init)   - float(da_init)

    if md is None:
        md = setup_moving_mesh_hat(r_ref, z_ref, a_ref, R_hat, H_hat, W_hat,
                                   Re, G_hat, U_m_hat, u_2d, p_2d,
                                   particle_maxh_rel, global_maxh_rel)

    F0, J0_full = evaluate_forces(float(dr_init), float(dz_init), float(da_init), md)
    eigpairs = estimate_eigenvectors(J0_full[:, :2])

    # The eigenvector to the smallest-magnitude eigenvalue is the natural
    # starting guess: that eigenvalue is closest to zero and will cross
    # zero at the bifurcation point.
    mu, phi_ev = eigpairs[0]
    l_vec = phi_ev.copy()
    phi_start = phi_ev / np.dot(l_vec, phi_ev)

    print("\n" + "=" * 65)
    print(f"  MOORE-SPENCE (TR)"
          f"  mu={mu:+.4e}  phi=({phi_ev[0]:+.4f}, {phi_ev[1]:+.4f})")
    print(f"  Start: r={r_off_hat_eq:.6f} z={z_off_hat_eq:.6f} a={a_hat_init:.6f}")
    print("=" * 65)

    r_bif, z_bif, a_bif, phi_bif, ok, final_res = _ms_trial(
        float(dr_init), float(dz_init), float(da_init),
        phi_start, l_vec, md, r_ref, z_ref, a_ref, L_c, tol, max_iter)

    print(f"\n  {'=' * 50}")
    print(f"  converged={ok}  |G|={final_res:.4e}  a={a_bif:.8f}")
    print(f"  {'=' * 50}")
    return r_bif, z_bif, a_bif, phi_bif, ok


if __name__ == "__main__":

    # Initial guess from the bifurcation diagramm
    r_off_hat_init = 0.61170000
    z_off_hat_init = 0.00350000
    a_hat_start = 0.135000

    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

    print("\nparticle_maxh_rel = ", particle_maxh_rel)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)

    G_hat, U_m_hat, u_bar_2d, p_bar_tilde_2d = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d, p_bar_tilde_2d)

    # refine initial guess to ensure starting MS with a root
    r_hat, z_hat, md_newton, dr_newton, dz_newton = newton_root_refine(r_off_hat_init, z_off_hat_init, a_hat_start, shared_data,
                                                                        max_iter=10)

    r_bif, z_bif, a_bif, phi_bif, converged = moore_spence_solve(r_hat, z_hat, a_hat_start, shared_data,
                                                md=md_newton, dr_init=dr_newton, dz_init=dz_newton, da_init=0.0)

    if converged:
        print(f"\nBifurcation point: r={r_bif:.10f}  z={z_bif:.10f}  a={a_bif:.10f}")
    else:
        print(f"\nMoore-Spence did not converge.")

    print("\n" + "=" * 65)
    print("Remeshing sanity check for Moore-Spence")
    print("=" * 65)

    md = setup_moving_mesh_hat(r_bif, z_bif, a_bif, R_hat, H_hat, W_hat, Re, G_hat, U_m_hat,
                               u_bar_2d, p_bar_tilde_2d, particle_maxh_rel, global_maxh_rel)

    F, J = evaluate_forces(0.0, 0.0, 0.0, md)
    J_sp = J[:, :2]

    print("r_off: ", r_bif)
    print("z_off: ", z_bif)
    print("a_hat: ", a_bif)

    print(f"\nF_p_x = {F[0]:.6e}")
    print(f"F_p_z = {F[1]:.6e}")
    print(f"|F|   = {np.linalg.norm(F):.6e}")

    print(f"\nJacobian (2x2 spatial part):")
    print(f"  dF_x/dr = {J_sp[0, 0]:+.6e}   dF_x/dz = {J_sp[0, 1]:+.6e}")
    print(f"  dF_z/dr = {J_sp[1, 0]:+.6e}   dF_z/dz = {J_sp[1, 1]:+.6e}")

    eigpairs = estimate_eigenvectors(J_sp)
    mu_min = eigpairs[0][0]

    print(f"|mu_min| = {abs(mu_min):.6e}")