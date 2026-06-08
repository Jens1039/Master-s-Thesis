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
    M1 = F_r (r, 0, a)  = 0     (radial force balance)
    M2 = J_zz(r, 0, a)  = 0     (antisymmetric eigenvalue crosses zero)

This module is intentionally self-contained from the rest of the
shape-optimization code, so the whole experimental track can be deleted
in one go if it does not pan out. The companion file is
``shape_optimization_symmetric.py``.
"""

import gc
import numpy as np

from firedrake.adjoint import stop_annotating

from perturbed_flow_differentiable import (
    MeshFlippedError, evaluate_forces, setup_moving_mesh,
    reset_ale_basis_for_step,
)
from problem_setup import particle_maxh_rel, global_maxh_rel


# At z=0 with z-reflection symmetry of the geometry, the bifurcating
# eigenvector of J_sp is exactly (0, 1). We hardcode it and never solve
# for it — that's the whole point of the reduction.
PHI_SYMMETRIC = np.array([0.0, 1.0])


# Pre-refine the equilibrium with a 1-variable Newton on F_r(r, 0, a)=0
# before starting Moore-Spence. Off → MS gets the raw initial guess and
# has to resolve (r, a) jointly.
USE_NEWTON_REFINEMENT = False

# Build the particle with a mirror-symmetric surface mesh (hemisphere split +
# setPeriodic reflection about z=0). Cancels the z-odd traction noise so the
# discrete F_z / J_rz vanish and J_zz is shielded from odd-mode contamination.
# Flip to False to A/B the effect on the pitchfork (watch whether |F_z noise|
# rises to ~1e-6 and whether J_zz still resolves to ~1e-10 at the crossing).
USE_SYMMETRIC_MESH = True

# (B) Mid-solve ALE re-linearisation thresholds. The linear particle basis
# xi_particle = dr·basis_r + da·basis_a is only accurate for SMALL drift; on an
# aggressive shape step the inner MS can march the particle Δr~0.08 / Δa~0.05 in
# ONE solve, re-staleing the basis mid-march and re-floor-stalling J_zz. When the
# drift SINCE THE LAST linearisation exceeds these, re-centre the basis at the
# current position (drift absorbed into xi_baseline; absolute r/a invariant).
DRIFT_RELIN_TOL_R = 0.02
DRIFT_RELIN_TOL_A = 0.015


def newton_root_refine_symmetric(r_off_init, a, shared_data, *, tol=1e-9, max_iter=10):
    """1D Newton on F_r(r, 0, a) = 0 with z=0 hardcoded.

    By z-reflection symmetry of the geometry, F_z(r, 0, a) = 0 identically.
    Only the radial force balance is non-trivial, and the Jacobian reduces
    to the scalar J_rr = ∂F_r/∂r.

    Returns
    -------
    r_cur : float
        Converged radial position r_off = r_off_init + dr.
    md : dict
        Moving-mesh data (passable to a downstream Moore-Spence call).
    dr : float
        Total radial displacement from r_off_init.
    """
    print("\n" + "=" * 65)
    print(f"  Newton Root Refinement (SYMMETRIC, z=0)  a = {a:.6f}")
    print("=" * 65)

    R, H, W, L_c, U_c, Re, G, U_m, u_2d, p_2d = shared_data

    md = setup_moving_mesh(r_off_init, 0.0, a,
                               R, H, W, Re, G, U_m,
                               u_2d, p_2d,
                               particle_maxh_rel, global_maxh_rel,
                               symmetric_mesh=USE_SYMMETRIC_MESH)

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
        r_cur = r_off_init + dr

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

    r_cur = r_off_init + dr
    print(f"  WARNING: symmetric Newton did not converge after {max_iter} iters "
          f"(|F_r| = {res:.4e}). Continuing.")
    return r_cur, md, dr


def _globalize_tr_2d(M, DM, p_N, Delta, eta_accept, eta_good, Delta_max,
                     _eval_step):
    """Powell dogleg trust-region on the Gauss-Newton model (2D).

    Subproblem:   min_p  m(p) := ½‖M + DM·p‖²    s.t.  ‖p‖ ≤ Δ.

    The dogleg path goes from the origin to the Cauchy point p_C
    (unconstrained minimizer of m along the steepest-descent direction)
    and then on to the Newton step p_N. The dogleg point is the unique
    intersection of this piecewise-linear path with the TR boundary; if
    p_N lies inside the TR we take it, if even p_C lies outside we clip
    along -g. Unlike plain backtracking along p_N, the direction changes
    with Δ — that is what makes this a real TR method.

    Returns (accepted, dy_try, rho, Delta_updated).
    """
    g    = DM.T @ M                       # ∇m(0) = DM^T M
    g_sq = float(g @ g)
    DM_g = DM @ g
    gBg  = float(DM_g @ DM_g)             # g^T (DM^T DM) g

    norm_pN = float(np.linalg.norm(p_N))

    if gBg > 0.0 and g_sq > 0.0:
        tau_C   = g_sq / gBg              # arg min_{τ>0} m(-τ g)
        p_C     = -tau_C * g
        norm_pC = float(np.linalg.norm(p_C))
    else:
        # Pathological: g ∈ ker(DM) or g = 0. Fall through to the
        # Cauchy-clipped branch (gradient-on-boundary) every attempt.
        p_C     = None
        norm_pC = float('inf')

    f0 = float(M @ M)                     # ‖M‖² — used in both pred and ared

    for attempt in range(20):
        if norm_pN <= Delta:
            dy_try, on_boundary, tag = p_N, False, "Newton"
        elif norm_pC >= Delta:
            if g_sq > 0.0:
                dy_try = -(Delta / np.sqrt(g_sq)) * g
            else:
                dy_try = np.zeros_like(M)
            on_boundary = True
            tag         = "Cauchy-clipped"
        else:
            d    = p_N - p_C
            a_q  = float(d @ d)
            b_q  = 2.0 * float(p_C @ d)
            c_q  = float(p_C @ p_C) - Delta * Delta
            disc = b_q * b_q - 4.0 * a_q * c_q
            tau  = (-b_q + np.sqrt(max(disc, 0.0))) / (2.0 * a_q)
            tau  = float(np.clip(tau, 0.0, 1.0))
            dy_try      = p_C + tau * d
            on_boundary = True
            tag         = f"dogleg τ={tau:.3f}"

        res_try, step_ok = _eval_step(dy_try)

        Mp_pred = M + DM @ dy_try
        pred    = f0 - float(Mp_pred @ Mp_pred)
        if step_ok and pred > 1e-30:
            rho = (f0 - res_try * res_try) / pred
        else:
            rho = -1.0

        print(f"         | TR-DL attempt {attempt} [{tag}]: "
              f"|M_try|={(res_try if step_ok else float('nan')):.4e}  "
              f"rho={rho:+.4f}  Delta={Delta:.4e}")

        if rho >= eta_accept and step_ok:
            if rho > eta_good and on_boundary:
                Delta = min(2.0 * Delta, Delta_max)
            return True, dy_try, rho, Delta

        Delta *= 0.5
        if Delta < 1e-12:
            print("         | TR-DL: Delta too small — giving up.")
            break

    return False, np.zeros_like(M), -1.0, Delta


def _ms_trial_symmetric(dr_init, da_init, md, r_ref, a_ref, L_c, tol, max_iter,
                        relinearize_basis=False):
    """One 2x2 TR-Newton trial on (F_r, J_zz) in unknowns (dr, da).

    ``relinearize_basis`` (B): when True, re-centre the linear ALE basis on the
    current particle position whenever the drift since the last linearisation
    exceeds DRIFT_RELIN_TOL_*. Keeps xi_particle accurate as the MS marches a
    long way in one solve (aggressive shape step), preventing the forward J_zz
    noise floor from re-stalling the dogleg. Mesh-state-preserving: r_ref/a_ref
    shift with the absorbed drift, absolute (r,a) are invariant. Caller must
    pass an md whose xi_baseline is private (see moore_spence_solve_symmetric).
    """
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
        # One AD pass with phi=(0,1) gives every block of DM_2x2.
        F_base, J_full, dJphi_dx = evaluate_forces(
            dr, 0.0, da, md, hessian_phi=PHI_SYMMETRIC)
        # Point at which (J_full, dJphi_dx) were evaluated — the FD-Jacobian
        # probe must compare AD vs FD at THIS (dr,da), not the post-step one.
        dr_jac, da_jac = float(dr), float(da)

        # FD-Jacobian consistency probe at the START of a DEFORMED solve.
        # Fires once per deformed MS solve (budget-capped) so we get a DM-vs-AD
        # reading per outer step WITHOUT waiting for a stall — and can watch DM
        # degrade as the accumulated deformation (||xi||) grows. On undeformed
        # geometry DM converges to e-16, so we only probe once xi≠0.
        if MS_FD_JAC_CHECK and k == 0 and _MS_FD_JAC_BUDGET[0] > 0:
            with stop_annotating():
                _xc = (float(np.max(np.abs(np.asarray(md['xi_channel'].dat.data_ro))))
                       if md.get('xi_channel') is not None else 0.0)
                _xb = (float(np.max(np.abs(np.asarray(md['xi_baseline'].dat.data_ro))))
                       if md.get('xi_baseline') is not None else 0.0)
            if _xc > 1e-12 or _xb > 1e-12:
                _MS_FD_JAC_BUDGET[0] -= 1
                _fd_jacobian_check(
                    dr_jac, da_jac, md, J_full, dJphi_dx,
                    tag=f"@iter0 (||xi_ch||={_xc:.2e}, ||xi_base||={_xb:.2e})")

        F_r = float(F_base[0])
        F_z_noise = float(F_base[1])       # ≡0 by symmetry, FE noise here
        J_zz = float(J_full[1, 1])         # = (J_sp @ (0,1))_z
        J_rr = float(J_full[0, 0])         # radial stiffness — TYPE-CHANGE
                                           # diagnostic (must stay < 0)

        M = np.array([F_r, J_zz])
        res = float(np.linalg.norm(M))
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
        print(f"         | |M| = {res:.4e}  |F_r| = {abs(F_r):.4e}  "
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
                  f"(|M| < tol={tol:.1e}).")
            print(f"     r_off = {r_cur:.10f}")
            print(f"     z_off = +0.0000000000  (fixed by symmetry)")
            print(f"     a     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = (0.0, 1.0)     (fixed by symmetry)")
            print(f"     J_zz      = {J_zz:+.6e}")
            return r_cur, a_cur, True, res, res_F, False

        # DM_2x2 = [[∂F_r/∂r,   ∂F_r/∂a ],
        #          [∂J_zz/∂r,  ∂J_zz/∂a]]
        # J_full[:, :2] = ∂F/∂(r,z), J_full[:, 2] = ∂F/∂a.
        # dJphi_dx[:, :3] = ∂(J·phi)/∂(r,z,a) with phi=(0,1).
        DM = np.array([
            [float(J_full[0, 0]), float(J_full[0, 2])],
            [float(dJphi_dx[1, 0]), float(dJphi_dx[1, 2])],
        ])

        cond_DM = float(np.linalg.cond(DM))
        print(f"         | cond(DM_2x2) = {cond_DM:.2e}")

        if cond_DM > 1e14:
            print("  !! DM_2x2 ill-conditioned — aborting.")
            return r_cur, a_cur, False, res, res_F, False

        try:
            dy = np.linalg.solve(DM, -M)
        except np.linalg.LinAlgError:
            print("  !! DM_2x2 singular — aborting.")
            return r_cur, a_cur, False, res, res_F, False

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

        accepted, dy_try, rho, Delta = _globalize_tr_2d(
            M, DM, dy, Delta, eta_accept, eta_good, Delta_max, _eval_step)

        if accepted:
            dr += dy_try[0]
            da += dy_try[1]
            print(f"         | step ACCEPTED")
            # (B) Mid-solve re-linearisation: if the particle has drifted far
            # from where the basis was last linearised, xi_particle = dr·basis_r
            # + da·basis_a becomes a poor mesh motion and the forward J_zz noise
            # floor rises, stalling the dogleg (rho<0, |M_try| frozen). Absorb
            # the drift into xi_baseline and re-solve the basis at the current
            # position; the mesh state (hence F_r, J_zz) is preserved, but the
            # basis — and thus the dogleg's nearby trial evaluations — become
            # accurate again. r_ref/a_ref shift so absolute r_cur/a_cur hold.
            if relinearize_basis and (abs(dr) > DRIFT_RELIN_TOL_R
                                      or abs(da) > DRIFT_RELIN_TOL_A):
                print(f"         | [Re-lin] drift (dr={dr:+.4e}, da={da:+.4e}) "
                      f"> tol — re-centring ALE basis mid-solve.")
                reset_ale_basis_for_step(md, dr, 0.0, da)
                r_ref += dr
                a_ref += da
                dr, da = 0.0, 0.0
        else:
            print(f"         | step REJECTED")

        # Stall check (FE noise floor on |M|). Counts BOTH accepted and
        # rejected iters: at the noise floor the function is flat in (r,a),
        # so TR-BT keeps rejecting at the same |M| — the state never
        # advances and the original "accepted-only" stall detector misses
        # this completely. Including rejected iters lets us bail after
        # ``stall_window`` Newton iters of no progress instead of letting
        # MS waste max_iter × 20 TR-attempts at the floor.
        res_history.append(res)
        if len(res_history) >= stall_window:
            window = res_history[-stall_window:]
            rel_spread = min(window) / max(window) if max(window) > 0 else 1.0
            if rel_spread > stall_ratio:
                print(f"         | STALL: |M| spread "
                      f"{min(window):.3e}..{max(window):.3e} "
                      f"(ratio {rel_spread:.3f} > {stall_ratio}) "
                      f"over last {stall_window} iters — "
                      f"bailing out (likely FE noise floor).")
                # FD-Jacobian consistency probe at the stalled (deformed)
                # state: is DM actually consistent with the forward G here?
                if MS_FD_JAC_CHECK and _MS_FD_JAC_BUDGET[0] > 0:
                    _MS_FD_JAC_BUDGET[0] -= 1
                    _fd_jacobian_check(
                        dr_jac, da_jac, md, J_full, dJphi_dx,
                        tag=f"@stall (|M|={res:.2e}, budget left "
                            f"{_MS_FD_JAC_BUDGET[0]})")
                r_cur, a_cur = r_ref + dr, a_ref + da
                return r_cur, a_cur, False, res, res_F, True

        gc.collect()

    r_cur, a_cur = r_ref + dr, a_ref + da
    return r_cur, a_cur, False, res, res_F, False


# ── Diagnostic: FD-check the reduced MS Jacobian against AD ───────────────
# A mesh-INDEPENDENT Newton stall (|M| floors at the same level regardless of
# FE resolution) is the signature of a G/DM INCONSISTENCY: the dogleg solves
# DM·dy = −G, but if the AD Jacobian DM misses a sensitivity the forward
# residual G actually has, the step is systematically wrong and |M| cannot
# fall below the |DM − dG/dy| level. The F_p Taylor tests verify the FIRST
# derivative of the force; they do NOT cover (a) the SECOND-order J_zz row of
# DM, nor (b) differentiation ON THE DEFORMED geometry. This check FD's every
# DM entry at the stalled (deformed) state and compares to AD, localising the
# missing term. Budget-capped to bound cost (4 force-Jacobian evals each).
MS_FD_JAC_CHECK  = True
_MS_FD_JAC_BUDGET = [8]      # mutable cap on total checks (list → no `global`)


def _fd_jacobian_check(dr, da, md, J_ad, dJphi_ad, *, hs=(1e-4, 1e-6), tag=""):
    """Central-FD the reduced 2×2 MS Jacobian DM and compare to AD, entry by
    entry, to localise a G/DM inconsistency on the deformed geometry.

        DM = [[∂F_r/∂r,  ∂F_r/∂a ]   row 0: first-order force-Jacobian (J_ad)
              [∂J_zz/∂r, ∂J_zz/∂a]]  row 1: J_zz = ∂F_z/∂z Hessian-VP (dJphi_ad)

    Runs the central FD at TWO step sizes ``hs`` to separate a real AD bug
    from FD truncation/floor: an FD that is CONSISTENT across both h but
    disagrees with AD ⇒ genuine AD inconsistency; an FD that drifts with h ⇒
    truncation (J_zz is small near the bifurcation, so its FD needs the larger
    h to carry signal).

    Interpretation:
      • AD ≈ FD (both h)  ⇒ DM is consistent with the forward G; the stall is
        NOT a Jacobian bug (look at the geometry map / residual itself).
      • row 0 mismatch ⇒ the FIRST-order force-Jacobian is wrong on the
        deformed geometry (would contradict the F_p Taylor test ⇒ that test did
        not cover the deformed configuration). Prime suspect: the DECOUPLED_LIFT
        query-position term (∂u_bar_3d/∂xi via the off-tape VertexOnlyMesh),
        a no-op at T=0 but active once xi_channel≠0.
      • row 1 mismatch ⇒ the J_zz Hessian-VP (dJphi_dx) is wrong on the deformed
        geometry (second-order AD through the BG lift — uncovered by any
        first-order test).

    Diagnostic only. Leaves mesh3d at the last FD perturbation; harmless since
    evaluate_forces reassigns mesh3d.coordinates on its next call.
    """
    ad = np.array([
        [float(J_ad[0, 0]),     float(J_ad[0, 2])],
        [float(dJphi_ad[1, 0]), float(dJphi_ad[1, 2])],
    ])
    names = [["dFr/dr", "dFr/da"], ["dJzz/dr", "dJzz/da"]]
    ttag = (' ' + tag) if tag else ''
    print(f"  [FD-Jac{ttag}] AD vs central-FD at h={tuple(f'{h:.0e}' for h in hs)}:")
    print(f"  [FD-Jac]   {'entry':8s} {'AD':>13s}" +
          "".join(f" {'FD(h='+f'{h:.0e}'+')':>15s}" for h in hs) + "   verdict")
    try:
        fd_by_h = []
        for h in hs:
            def _Fr_Jzz(dr_, da_):
                F, J = evaluate_forces(dr_, 0.0, da_, md)   # jacobian=True, no Hessian
                return float(F[0]), float(J[1, 1])
            Frp, Jzzp     = _Fr_Jzz(dr + h, da)
            Frm, Jzzm     = _Fr_Jzz(dr - h, da)
            Fra_p, Jzza_p = _Fr_Jzz(dr, da + h)
            Fra_m, Jzza_m = _Fr_Jzz(dr, da - h)
            fd_by_h.append(np.array([
                [(Frp  - Frm)  / (2 * h), (Fra_p  - Fra_m)  / (2 * h)],
                [(Jzzp - Jzzm) / (2 * h), (Jzza_p - Jzza_m) / (2 * h)],
            ]))
        for i in range(2):
            for j in range(2):
                fds = [fd[i, j] for fd in fd_by_h]
                # consistency across h (truncation indicator)
                fd_spread = (abs(fds[0] - fds[-1])
                             / max(abs(fds[-1]), 1e-30))
                rel = abs(ad[i, j] - fds[-1]) / max(abs(fds[-1]), 1e-30)
                if rel > 1e-2 and fd_spread < 1e-1:
                    verdict = "<<< AD MISMATCH (FD h-consistent)"
                elif rel > 1e-2:
                    verdict = "ambiguous (FD drifts w/ h)"
                else:
                    verdict = "ok"
                print(f"  [FD-Jac]   {names[i][j]:8s} {ad[i,j]:>+13.5e}" +
                      "".join(f" {v:>+15.5e}" for v in fds) +
                      f"   relerr={rel:.1e}  {verdict}")
    except MeshFlippedError:
        print(f"  [FD-Jac] skipped — mesh flipped at an FD perturbation")
    except Exception as e:
        print(f"  [FD-Jac] skipped — {type(e).__name__}: {e}")


def moore_spence_solve_symmetric(r_off_eq, a_init, shared_data, *,
                                 tol=1e-7, max_iter=15, md=None,
                                 dr_init=0.0, da_init=0.0,
                                 relinearize_basis=False):
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
        True iff |M| dropped below ``tol`` (not just hit the noise floor).
    final_F_r : float
        |F_r| at the final iterate — diagnostic of the FE noise floor.
    final_M : float
        |M| = ||(F_r, J_zz)|| at the final iterate.
    stalled_at_floor : bool
        True iff termination was triggered by the FE-noise-floor stall
        detector (|M| flat over ``stall_window`` consecutive iters).
        Lets the caller distinguish "MS failed to find the bif" from
        "MS reached the FE discretisation's resolution limit".
    """
    R, H, W, L_c, U_c, Re, G, U_m, u_2d, p_2d = shared_data
    r_ref = float(r_off_eq) - float(dr_init)
    a_ref = float(a_init) - float(da_init)

    if md is None:
        md = setup_moving_mesh(r_ref, 0.0, a_ref, R, H, W,
                                   Re, G, U_m, u_2d, p_2d,
                                   particle_maxh_rel, global_maxh_rel,
                                   symmetric_mesh=USE_SYMMETRIC_MESH)
    elif relinearize_basis:
        # (A) Re-linearise the particle ALE basis on THIS call's actual mesh
        # configuration before iterating. The passed-in md carries a basis
        # solved at a DIFFERENT config (the reference / last-accepted step),
        # but a TR-backtrack trial has set a fresh xi_channel, so on the
        # deformed mesh xi_particle = dr·basis_r + da·basis_a is an inaccurate
        # mesh motion from MS-iteration 0 — the forward J_zz then carries a
        # noise floor (~1e-6, see probe_jzz_deformed) that the MS cannot drive
        # below, and the dogleg thrashes. Re-solving the basis here on
        # X_ref + xi_baseline + xi_channel removes that floor.
        #
        # Work on an ISOLATED md: the (B) mid-solve re-linearisations inside
        # _ms_trial_symmetric absorb drift INTO xi_baseline *in place*, so give
        # this call its own xi_baseline copy to avoid mutating the caller's
        # shared md_try/live md. basis_*_data and a_init are REBOUND (not
        # mutated in place) by reset_ale_basis_for_step, so they need no copy.
        # See project_ms_stall_rootcause.
        md = dict(md)
        md['xi_baseline'] = md['xi_baseline'].copy(deepcopy=True)
        reset_ale_basis_for_step(md, 0.0, 0.0, 0.0)

    print("\n" + "=" * 65)
    print(f"  MOORE-SPENCE SYMMETRIC (2x2, z=0 hardcoded, phi=(0,1))")
    print(f"  Start: r={r_off_eq:.6f}  a={a_init:.6f}")
    print("=" * 65)

    r_bif, a_bif, ok, final_res, final_F, stalled = _ms_trial_symmetric(
        float(dr_init), float(da_init),
        md, r_ref, a_ref, L_c, tol, max_iter,
        relinearize_basis=relinearize_basis)

    print(f"\n  {'=' * 50}")
    print(f"  converged={ok}  |M|={final_res:.4e}  |F_r|={final_F:.4e}  "
          f"a={a_bif:.8f}")
    print(f"  {'=' * 50}")
    return r_bif, a_bif, ok, final_F, final_res, stalled


if __name__ == "__main__":
    # Sanity test: reproduce Step 1 of log_symm.txt / log.txt starting from
    # r0=0.6098, a=0.1375. With z=0 hardcoded, the bif should land near
    # a ≈ 0.1375 (Newton-root) and unchanged in r/a after MS (the
    # initial point already satisfies F_r=J_zz=0 to FE noise).
    from background_flow_differentiable import background_flow_differentiable
    from problem_setup import R, H, W, a, L_c, U_c, Re

    bg = background_flow_differentiable(R, H, W, Re)
    G, U_m, u_bar_2d, p_bar_tilde_2d = bg.solve_2D_background_flow()
    shared_data = (R, H, W, L_c, U_c, Re,
                   G, U_m, u_bar_2d, p_bar_tilde_2d)

    r_init = 0.61
    a_init = 0.137

    header = ("SYMMETRIC PITCHFORK SEARCH" if USE_NEWTON_REFINEMENT
              else "SYMMETRIC PITCHFORK SEARCH (no Newton pre-refine)")
    print("\n" + "#" * 65)
    print(f"  {header}")
    print(f"  Input: r={r_init:.6f}  z=0 (fixed)  a={a_init:.6f}")
    print("#" * 65)

    if USE_NEWTON_REFINEMENT:
        r_eq, md, dr0 = newton_root_refine_symmetric(
            r_init, a_init, shared_data, tol=1e-10, max_iter=15)
    else:
        # Build MS mesh straight at the initial guess; MS resolves the
        # (r, a) shift jointly.
        r_eq = r_init
        md = setup_moving_mesh(r_eq, 0.0, a_init,
                                   R, H, W, Re, G, U_m,
                                   u_bar_2d, p_bar_tilde_2d,
                                   particle_maxh_rel, global_maxh_rel,
                                   symmetric_mesh=USE_SYMMETRIC_MESH)
        dr0 = 0.0

    r_bif, a_bif, ok, F_norm, M_norm, stalled = moore_spence_solve_symmetric(
        r_eq, a_init, shared_data, md=md,
        dr_init=dr0, da_init=0.0,
        tol=1e-12, max_iter=20)

    print(f"\n  Result: r_bif = {r_bif:.8f},  a_bif = {a_bif:.8f},  "
          f"converged = {ok}")
