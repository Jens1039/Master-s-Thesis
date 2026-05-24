"""Reduced 2x2 Moore-Spence solver for the pitchfork bifurcation on the
z-symmetric branch (z=0 hardcoded, phi=(0,1) fixed by symmetry).

This is the "Variante B" reduction discussed in the design conversation:
at the channel-mid-plane equilibrium z=0, the geometry's z-reflection
symmetry forces
    F_z(r, 0, a) ≡ 0,
    ∂F_r/∂z(r, 0, a) = ∂F_z/∂r(r, 0, a) = 0,
so the 2D position-Jacobian J_sp is diagonal: diag(J_rr, J_zz). The
bifurcation eigenmode is (0, 1) with eigenvalue J_zz; no phi-normalization
equation is needed. The full 5x5 Moore-Spence system therefore collapses
to the 2x2 system in (r, a):
    G1 = F_r (r, 0, a)  = 0     (radial force balance)
    G2 = J_zz(r, 0, a)  = 0     (antisymmetric eigenvalue crosses zero)

This module is intentionally self-contained from the rest of the
shape-optimization code, so the whole experimental track can be deleted
in one go if it does not pan out. The companion file is
``shape_optimization_symmetric.py``.
"""

import gc
import numpy as np

from perturbed_flow_differentiable import (
    MeshFlippedError, evaluate_forces, setup_moving_mesh_hat,
)
from config_paper_parameters import particle_maxh_rel, global_maxh_rel


# At z=0 with z-reflection symmetry of the geometry, the bifurcating
# eigenvector of J_sp is exactly (0, 1). We hardcode it and never solve
# for it — that's the whole point of the reduction.
PHI_SYMMETRIC = np.array([0.0, 1.0])


def newton_root_refine_symmetric(r_off_hat_init, a_hat, shared_data, *,
                                 tol=1e-9, max_iter=10):
    """1D Newton on F_r(r, 0, a) = 0 with z=0 hardcoded.

    By z-reflection symmetry of the geometry, F_z(r, 0, a) = 0 identically.
    Only the radial force balance is non-trivial, and the Jacobian reduces
    to the scalar J_rr = ∂F_r/∂r.

    Returns
    -------
    r_cur : float
        Converged radial position r_off_hat = r_off_hat_init + dr.
    md : dict
        Moving-mesh data (passable to a downstream Moore-Spence call).
    dr : float
        Total radial displacement from r_off_hat_init.
    """
    print("\n" + "=" * 65)
    print(f"  Newton Root Refinement (SYMMETRIC, z=0)  a_hat = {a_hat:.6f}")
    print("=" * 65)

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    md = setup_moving_mesh_hat(r_off_hat_init, 0.0, a_hat,
                               R_hat, H_hat, W_hat, Re, G_hat, U_m_hat,
                               u_2d, p_2d,
                               particle_maxh_rel, global_maxh_rel)

    dr = 0.0

    stall_window = 3
    stall_ratio = 0.9
    res_history = []

    for k in range(max_iter):
        F, J_full = evaluate_forces(dr, 0.0, 0.0, md)
        F_r = float(F[0])
        F_z = float(F[1])               # should be ~0 by symmetry (FE noise)
        J_rr = float(J_full[0, 0])

        res = abs(F_r)
        r_cur = r_off_hat_init + dr

        print(f"  Iter {k:2d} | r = {r_cur:+.8f}  z = +0.00000000 (fixed) | "
              f"|F_r| = {res:.4e}  J_rr = {J_rr:+.4e}  "
              f"|F_z noise| = {abs(F_z):.2e}")

        if res < tol:
            print(f"  -> Converged after {k} iterations (|F_r| < tol={tol:.1e}).\n")
            return r_cur, md, dr

        res_history.append(res)
        if len(res_history) >= stall_window:
            window = res_history[-stall_window:]
            rel_spread = min(window) / max(window) if max(window) > 0 else 1.0
            if rel_spread > stall_ratio:
                print(f"  -> Newton STALL: |F_r| spread "
                      f"{min(window):.3e}..{max(window):.3e} "
                      f"(ratio {rel_spread:.3f} > {stall_ratio}) — "
                      f"bailing out at FE noise floor.")
                return r_cur, md, dr

        if abs(J_rr) < 1e-12:
            print(f"  !! J_rr singular ({J_rr:.3e}) — aborting.")
            return r_cur, md, dr

        dx = -F_r / J_rr

        alpha = 1.0
        for ls in range(10):
            try:
                F_try = evaluate_forces(dr + alpha * dx, 0.0, 0.0, md,
                                        jacobian=False)
            except MeshFlippedError:
                print(f"         | line search: flipped at alpha={alpha:.4e}, halving")
                alpha *= 0.5
                continue
            if abs(F_try[0]) < (1 - 1e-4 * alpha) * res:
                break
            alpha *= 0.5

        dr += alpha * dx
        gc.collect()

    r_cur = r_off_hat_init + dr
    print(f"  WARNING: symmetric Newton did not converge after {max_iter} iters "
          f"(|F_r| = {res:.4e}). Continuing.")
    return r_cur, md, dr


def _globalize_tr_2d(dy, Delta, eta_accept, eta_good, Delta_max,
                     _eval_step, _compute_rho):
    """Trust-region with backtracking for a 2-vector step dy=(dr,da).

    Mirrors ``locate_bifurcation_points._globalize_tr`` but trims to the
    2-component case: there is no phi slice to skip, so the entire dy is
    clipped against Delta.
    """
    for attempt in range(20):
        step_norm = float(np.linalg.norm(dy))
        at_boundary = step_norm > Delta
        if at_boundary:
            dy_try = dy * (Delta / step_norm)
        else:
            dy_try = dy.copy()

        res_try, step_ok = _eval_step(dy_try)
        rho = _compute_rho(dy_try, res_try, step_ok)

        print(f"         | TR-BT attempt {attempt}: "
              f"|G_try|={res_try if step_ok else float('nan'):.4e}  "
              f"rho={rho:+.4f}  Delta={Delta:.4e}")

        if rho >= eta_accept and step_ok:
            if rho > eta_good and at_boundary:
                Delta = min(Delta * 2.0, Delta_max)
            return True, dy_try, rho, Delta

        Delta *= 0.5
        if Delta < 1e-12:
            print("         | TR-BT: Delta too small — giving up.")
            break

    return False, dy.copy(), -1.0, Delta


def _ms_trial_symmetric(dr_init, da_init, md, r_ref, a_ref, L_c, tol, max_iter):
    """One 2x2 TR-Newton trial on (F_r, J_zz) in unknowns (dr, da)."""
    dr, da = float(dr_init), float(da_init)
    Delta, Delta_max = 1e-2, 1.0
    eta_accept, eta_good = 0.1, 0.75
    res = float('inf')
    res_F = float('inf')

    stall_window = 4
    stall_ratio = 0.95
    res_history = []
    prev_res = None

    for k in range(max_iter):
        # One AD pass with phi=(0,1) gives every block of DG_2x2.
        F_base, J_full, dJphi_dx = evaluate_forces(
            dr, 0.0, da, md, hessian_phi=PHI_SYMMETRIC)

        F_r = float(F_base[0])
        F_z_noise = float(F_base[1])       # ≡0 by symmetry, FE noise here
        J_zz = float(J_full[1, 1])         # = (J_sp @ (0,1))_z
        J_rr = float(J_full[0, 0])         # radial stiffness — TYPE-CHANGE
                                           # diagnostic (must stay < 0)

        G = np.array([F_r, J_zz])
        res = float(np.linalg.norm(G))
        res_F = abs(F_r)

        # Off-diagonal Jacobian entries should be 0 by symmetry — print them
        # as a noise indicator (if they're large, the mesh asymmetry is
        # large and the reduction's validity is reduced).
        J_rz_noise = float(J_full[0, 1])

        r_cur, a_cur = r_ref + dr, a_ref + da

        if prev_res is not None and prev_res > 0:
            ratio_str = f"  ratio = {res/prev_res:.3f}"
        else:
            ratio_str = "  ratio = —    "
        prev_res = res

        print(f"\n  Iter {k:2d} | r = {r_cur:+.8f}  a = {a_cur:.8f}  (z=0 fixed)")
        print(f"         | |G| = {res:.4e}  |F_r| = {abs(F_r):.4e}  "
              f"J_zz = {J_zz:+.4e}{ratio_str}")
        print(f"         | J_rr = {J_rr:+.4e}  (must stay < 0 for radial "
              f"stability — pitchfork reduction valid)")
        print(f"         | noise: |F_z| = {abs(F_z_noise):.2e}  "
              f"|J_rz| = {abs(J_rz_noise):.2e}")

        # TYPE-CHANGE WARNING: pitchfork reduction assumes J_rr < 0.
        # If J_rr crosses zero, the centerline equilibrium has lost radial
        # stability and the 2x2 system is tracking a physically meaningless
        # point. We WARN per iter but do not abort here — let the user
        # decide based on the trajectory.
        if J_rr >= 0.0:
            print(f"         | *** WARNING: J_rr = {J_rr:+.4e} ≥ 0 — "
                  f"radial instability! Reduction assumption violated. ***")

        if res < tol:
            a_phys = a_cur * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations "
                  f"(|G| < tol={tol:.1e}).")
            print(f"     r_off_hat = {r_cur:.10f}")
            print(f"     z_off_hat = +0.0000000000  (fixed by symmetry)")
            print(f"     a_hat     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = (0.0, 1.0)     (fixed by symmetry)")
            print(f"     J_zz      = {J_zz:+.6e}")
            return r_cur, a_cur, True, res, res_F

        # DG_2x2 = [[∂F_r/∂r,   ∂F_r/∂a ],
        #          [∂J_zz/∂r,  ∂J_zz/∂a]]
        # J_full[:, :2] = ∂F/∂(r,z), J_full[:, 2] = ∂F/∂a.
        # dJphi_dx[:, :3] = ∂(J·phi)/∂(r,z,a) with phi=(0,1).
        DG = np.array([
            [float(J_full[0, 0]), float(J_full[0, 2])],
            [float(dJphi_dx[1, 0]), float(dJphi_dx[1, 2])],
        ])

        cond_DG = float(np.linalg.cond(DG))
        print(f"         | cond(DG_2x2) = {cond_DG:.2e}")

        if cond_DG > 1e14:
            print("  !! DG_2x2 ill-conditioned — aborting.")
            return r_cur, a_cur, False, res, res_F

        try:
            dy = np.linalg.solve(DG, -G)
        except np.linalg.LinAlgError:
            print("  !! DG_2x2 singular — aborting.")
            return r_cur, a_cur, False, res, res_F

        print(f"         | step: dr={dy[0]:+.4e}  da={dy[1]:+.4e}")

        def _eval_step(dy_candidate):
            dr_t = dr + dy_candidate[0]
            da_t = da + dy_candidate[1]
            if a_ref + da_t <= 0:
                return float('nan'), False
            try:
                F_t, J_t = evaluate_forces(dr_t, 0.0, da_t, md)
                G_t = np.array([float(F_t[0]), float(J_t[1, 1])])
                return float(np.linalg.norm(G_t)), True
            except MeshFlippedError:
                return float('nan'), False

        def _compute_rho(dy_try, res_try, step_ok):
            G_pred = G + DG @ dy_try
            pred = res ** 2 - float(np.linalg.norm(G_pred)) ** 2
            if step_ok and abs(pred) > 1e-30:
                return (res ** 2 - res_try ** 2) / pred
            return -1.0 if not step_ok else (1.0 if res_try <= res else -1.0)

        accepted, dy_try, rho, Delta = _globalize_tr_2d(
            dy, Delta, eta_accept, eta_good, Delta_max,
            _eval_step, _compute_rho)

        if accepted:
            dr += dy_try[0]
            da += dy_try[1]
            print(f"         | step ACCEPTED")

            # Stall check: same scheme as the full Moore-Spence (FE noise
            # floor on |G|).
            res_history.append(res)
            if len(res_history) >= stall_window:
                window = res_history[-stall_window:]
                rel_spread = min(window) / max(window) if max(window) > 0 else 1.0
                if rel_spread > stall_ratio:
                    print(f"         | STALL: |G| spread "
                          f"{min(window):.3e}..{max(window):.3e} "
                          f"(ratio {rel_spread:.3f} > {stall_ratio}) "
                          f"over last {stall_window} accepted iters — "
                          f"bailing out (likely FE noise floor).")
                    r_cur, a_cur = r_ref + dr, a_ref + da
                    return r_cur, a_cur, False, res, res_F
        else:
            print(f"         | step REJECTED")

        gc.collect()

    r_cur, a_cur = r_ref + dr, a_ref + da
    return r_cur, a_cur, False, res, res_F


def moore_spence_solve_symmetric(r_off_hat_eq, a_hat_init, shared_data, *,
                                 tol=1e-7, max_iter=15, md=None,
                                 dr_init=0.0, da_init=0.0):
    """Reduced 2x2 Moore-Spence solver. Tracks (r_bif, a_bif) on the
    pitchfork manifold with z=0 hardcoded.

    Same calling pattern as ``locate_bifurcation_points.moore_spence_solve``
    minus the z arguments, so it can be swapped into a shape-optimization
    loop with minimal surgery.

    Returns
    -------
    r_bif, a_bif : float
        Converged bifurcation point on the centerline.
    converged : bool
        True iff |G| dropped below ``tol`` (not just hit the noise floor).
    final_F_r : float
        |F_r| at the final iterate — diagnostic of the FE noise floor.
    """
    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data
    r_ref = float(r_off_hat_eq) - float(dr_init)
    a_ref = float(a_hat_init) - float(da_init)

    if md is None:
        md = setup_moving_mesh_hat(r_ref, 0.0, a_ref, R_hat, H_hat, W_hat,
                                   Re, G_hat, U_m_hat, u_2d, p_2d,
                                   particle_maxh_rel, global_maxh_rel)

    print("\n" + "=" * 65)
    print(f"  MOORE-SPENCE SYMMETRIC (2x2, z=0 hardcoded, phi=(0,1))")
    print(f"  Start: r={r_off_hat_eq:.6f}  a={a_hat_init:.6f}")
    print("=" * 65)

    r_bif, a_bif, ok, final_res, final_F = _ms_trial_symmetric(
        float(dr_init), float(da_init),
        md, r_ref, a_ref, L_c, tol, max_iter)

    print(f"\n  {'=' * 50}")
    print(f"  converged={ok}  |G|={final_res:.4e}  |F_r|={final_F:.4e}  "
          f"a={a_bif:.8f}")
    print(f"  {'=' * 50}")
    return r_bif, a_bif, ok, final_F


if __name__ == "__main__":
    # Sanity test: reproduce Step 1 of log_symm.txt / log.txt starting from
    # r0=0.6098, a=0.1375. With z=0 hardcoded, the bif should land near
    # a_hat ≈ 0.1375 (Newton-root) and unchanged in r/a after MS (the
    # initial point already satisfies F_r=J_zz=0 to FE noise).
    from background_flow_differentiable import background_flow_differentiable
    from nondimensionalization import nondimensionalisation
    from config_paper_parameters import R, H, W, a, Q, rho, mu

    R_hat, H_hat, W_hat, _, L_c, U_c, Re = nondimensionalisation(
        R, H, W, a, Q, rho, mu, print_values=True)

    print("\nparticle_maxh_rel =", particle_maxh_rel)
    print("global_maxh_rel   =", global_maxh_rel)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_hat, U_m_hat, u_bar_2d, p_bar_tilde_2d = bg.solve_2D_background_flow()
    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re,
                   G_hat, U_m_hat, u_bar_2d, p_bar_tilde_2d)

    r_init = 0.6098
    a_init = 0.1375

    print("\n" + "#" * 65)
    print(f"  SYMMETRIC PITCHFORK SEARCH")
    print(f"  Input: r={r_init:.6f}  z=0 (fixed)  a={a_init:.6f}")
    print("#" * 65)

    r_eq, md, dr0 = newton_root_refine_symmetric(
        r_init, a_init, shared_data, tol=1e-10, max_iter=15)

    r_bif, a_bif, ok, F_norm = moore_spence_solve_symmetric(
        r_eq, a_init, shared_data, md=md,
        dr_init=dr0, da_init=0.0,
        tol=1e-12, max_iter=20)

    print(f"\n  Result: r_bif = {r_bif:.8f},  a_bif = {a_bif:.8f},  "
          f"converged = {ok}")
