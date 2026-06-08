import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import gc
import numpy as np

from background_flow_differentiable import background_flow_differentiable
from perturbed_flow_differentiable import MeshFlippedError, check_mesh_quality, setup_moving_mesh, evaluate_forces, estimate_eigenvectors, _build_xi
from problem_setup import R, H, W, a, L_c, U_c, Re, particle_maxh_rel, global_maxh_rel


# Pre-refine the equilibrium with a 2-variable Newton on F(r,z)=0 before
# starting Moore-Spence. Off → MS gets the raw initial guess and has to
# resolve (r,z) and (a, φ) jointly.
USE_NEWTON_REFINEMENT = False


def _log_mesh_quality(md, label, cycle, *, dr=None, dz=None, da=None):
    """Compute per-cell quality metrics on the current (deformed) mesh and
    print summary statistics. Metrics:
      - oriented det(J): cell volume * reference sign; <= 0 means flipped.
      - edge ratio: longest edge / shortest edge (regular tet = 1).
      - dihedral angles [deg]: regular tet = 70.53°, degenerate -> 0 or 180.
      - |xi|: per-node displacement magnitude vs. reference mesh X_ref.
    """
    mesh3d = md['mesh3d']

    # Displacement field xi = current_coords - X_ref (the bump-driven deformation).
    coords = np.asarray(mesh3d.coordinates.dat.data_ro)
    X_ref = np.asarray(md['X_ref'].dat.data_ro)
    xi = coords - X_ref
    xi_mag = np.linalg.norm(xi, axis=1)
    a = md['a_init']

    # Oriented detJ
    DG0 = FunctionSpace(mesh3d, "DG", 0)
    detJ_fn = Function(DG0)
    detJ_fn.interpolate(JacobianDeterminant(mesh3d))
    detJ_oriented = np.asarray(detJ_fn.dat.data_ro) * np.asarray(md['ref_signs'])

    # Per-cell corner vertices via a CG1 scratch space (robust to mesh order)
    V_lin = VectorFunctionSpace(mesh3d, "CG", 1)
    verts_fn = Function(V_lin)
    verts_fn.interpolate(SpatialCoordinate(mesh3d))
    cnm = V_lin.cell_node_map().values
    verts = np.asarray(verts_fn.dat.data_ro)[cnm]   # (n_cells, 4, 3)

    # Edge lengths (6 edges per tet)
    edge_idx = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    edges = verts[:, edge_idx[:, 1]] - verts[:, edge_idx[:, 0]]
    edge_lens = np.linalg.norm(edges, axis=2)
    edge_ratio = edge_lens.max(axis=1) / np.maximum(edge_lens.min(axis=1), 1e-300)

    # Dihedral angles (one per edge)
    edge_data = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2),
                 (1, 2, 0, 3), (1, 3, 0, 2), (2, 3, 0, 1)]
    dihedrals = np.empty((verts.shape[0], 6))
    for k, (i, j, a, b) in enumerate(edge_data):
        e = verts[:, j] - verts[:, i]
        e_hat = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-300)
        va = verts[:, a] - verts[:, i]
        vb = verts[:, b] - verts[:, i]
        va_p = va - np.sum(va * e_hat, axis=1, keepdims=True) * e_hat
        vb_p = vb - np.sum(vb * e_hat, axis=1, keepdims=True) * e_hat
        denom = np.maximum(np.linalg.norm(va_p, axis=1) * np.linalg.norm(vb_p, axis=1), 1e-300)
        cos_t = np.sum(va_p * vb_p, axis=1) / denom
        dihedrals[:, k] = np.arccos(np.clip(cos_t, -1.0, 1.0))
    dihedrals_deg = np.degrees(dihedrals)

    n_cells = detJ_oriented.size
    n_flipped = int(np.sum(detJ_oriented <= 0))

    print(f"\n  Mesh quality after {label} (cycle {cycle}):  n_cells={n_cells}  flipped={n_flipped}")
    if dr is not None or dz is not None or da is not None:
        print(f"    accumulated:     dr={dr if dr is not None else 0.0:+.6e}  "
              f"dz={dz if dz is not None else 0.0:+.6e}  "
              f"da={da if da is not None else 0.0:+.6e}   (a_ref={a:.6f})")
    print(f"    |xi| (node):     max={xi_mag.max():.4e}  p99={np.percentile(xi_mag, 99):.4e}"
          f"  median={np.median(xi_mag):.4e}   (relative to a_ref: max={xi_mag.max()/a:.3e})")
    print(f"    detJ_oriented:   min={detJ_oriented.min():+.3e}  median={np.median(detJ_oriented):+.3e}  max={detJ_oriented.max():+.3e}")
    print(f"    edge ratio:      median={np.median(edge_ratio):.3f}  p90={np.percentile(edge_ratio, 90):.3f}"
          f"  p99={np.percentile(edge_ratio, 99):.3f}  max={edge_ratio.max():.3f}   (regular tet = 1)")
    print(f"    min dihedral°:   min={dihedrals_deg.min():.3f}  p1={np.percentile(dihedrals_deg.min(axis=1), 1):.3f}"
          f"  p10={np.percentile(dihedrals_deg.min(axis=1), 10):.3f}  median={np.median(dihedrals_deg.min(axis=1)):.3f}   (regular tet ≈ 70.53)")
    print(f"    max dihedral°:   median={np.median(dihedrals_deg.max(axis=1)):.3f}"
          f"  p90={np.percentile(dihedrals_deg.max(axis=1), 90):.3f}"
          f"  p99={np.percentile(dihedrals_deg.max(axis=1), 99):.3f}  max={dihedrals_deg.max():.3f}")


def newton_root_refine(r_off_init, z_off_init, a, shared_data, *,
                       tol=1e-9, max_iter=10):

    print("\n" + "=" * 65)
    print(f"  Newton Root Refinement (a = {a:.6f})")
    print("=" * 65)

    R, H, W, L_c, U_c, Re, G, U_m, u_2d, p_2d = shared_data

    md = setup_moving_mesh(r_off_init, z_off_init, a, R, H, W, Re, G, U_m, u_2d, p_2d,
                               particle_maxh_rel, global_maxh_rel)

    dr, dz = 0.0, 0.0

    # Stall-detector state: same idea as in _ms_trial. Newton on F=0 hits
    # the FE noise floor of the surface-integrated forces; further iterations
    # only chase noise gradients, drift the iterate (often toward the bif
    # point and then *past* it), and eventually tangle the mesh.
    stall_window = 3
    stall_ratio  = 0.9
    res_history  = []

    for k in range(max_iter):
        F, J_full = evaluate_forces(dr, dz, 0.0, md)
        J = J_full[:, :2]

        res = np.linalg.norm(F)
        r_cur = r_off_init + dr
        z_cur = z_off_init + dz

        print(f"  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f} | |F| = {res:.4e}  cond(J) = {np.linalg.cond(J):.2e}")

        if res < tol:
            print(f"  -> Converged after {k} iterations (|F| < tol={tol:.1e}).\n")
            return r_cur, z_cur, md, dr, dz

        # Bail when |F| stops shrinking — usually means we have reached the
        # FE noise floor of the residual, and further Newton steps will only
        # diffuse the iterate while burning compute.
        res_history.append(res)
        if len(res_history) >= stall_window:
            window = res_history[-stall_window:]
            rel_spread = min(window) / max(window) if max(window) > 0 else 1.0
            if rel_spread > stall_ratio:
                print(f"  -> Newton STALL: |F| spread {min(window):.3e}..{max(window):.3e} "
                      f"(ratio {rel_spread:.3f} > {stall_ratio}) over last {stall_window} "
                      f"iters — bailing out at FE noise floor.")
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

    r_cur = r_off_init + dr
    z_cur = z_off_init + dz
    print(f"  WARNING: Newton refinement did not converge at a = {a} "
          f"after {max_iter} iterations (|F| = {res:.4e}). Continuing anyway.")
    return r_cur, z_cur, md, dr, dz


def _ms_trial(dr_init, dz_init, da_init, phi_start, l_vec, md, r_ref, z_ref, a_ref, L_c, tol, max_iter):
    """Single Moore-Spence trial with a given starting phi.

    Globalization: Powell dogleg trust region on the Gauss-Newton model
    (see ``_globalize_tr``).
    Returns (r, z, a, phi, converged, final_|M|, final_|F|, stalled_at_floor).
    """

    dr, dz, da = float(dr_init), float(dz_init), float(da_init)
    phi = phi_start.copy()
    Delta, Delta_max = 1e-1, 1.0
    eta_accept, eta_good = 0.1, 0.75
    res = float('inf')
    res_F = float('inf')   # final |F| = |M1| — used as a noise-floor diagnostic

    # Stall-detector state: ring buffer of |M| values.
    # Counts BOTH accepted and rejected iters: at the noise floor the
    # function is flat in (dr,dz,da,phi), so the TR keeps rejecting at the
    # same |M| — the state never advances and an accepted-only stall
    # detector misses this completely. Including rejected iters lets us
    # bail after ``stall_window`` Newton iters of no progress instead of
    # letting MS waste max_iter × 20 TR-attempts at the floor.
    stall_window = 4          # # of iters to look back over
    stall_ratio  = 0.95       # require min(window)/max(window) ≤ stall_ratio
    res_history = []
    prev_res    = None        # for per-iter reduction-ratio diagnostic

    for k in range(max_iter):

        # All entries of DM via AD: 1 forward + 2 reverse + 2 H·v
        F_base, J_full, dJphi_dx = evaluate_forces(
            dr, dz, da, md, hessian_phi=phi)
        J_sp = J_full[:, :2]
        dF_da = J_full[:, 2]

        M1 = F_base
        M2 = J_sp @ phi
        M3_val = np.dot(l_vec, phi) - 1.0
        M = np.concatenate([M1, M2, [M3_val]])
        res = np.linalg.norm(M)
        res_F = float(np.linalg.norm(M1))

        eigs = np.linalg.eigvals(J_sp)
        eigenvalue = eigs[np.argmin(np.abs(eigs))]
        r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
        # Per-iter reduction ratio: res / prev_res.  < 0.5 ≈ quadratic Newton,
        # < 0.9 still linear-ish, > 0.9 stalling (Newton fighting noise).
        if prev_res is not None and prev_res > 0:
            iter_ratio = res / prev_res
            ratio_str = f"  ratio = {iter_ratio:.3f}"
        else:
            ratio_str = "  ratio = —    "
        prev_res = res

        print(f"\n  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f}  a = {a_cur:.8f}")
        print(f"         | |M| = {res:.4e}  |F| = {np.linalg.norm(M1):.4e}"
              f"  |J*phi| = {np.linalg.norm(M2):.4e}"
              f"  eigenvalue = {eigenvalue:+.4e}"
              f"{ratio_str}")

        if res < tol:
            a_phys = a_cur * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations (|M| < tol={tol:.1e}).")
            print(f"     r_off = {r_cur:.10f}")
            print(f"     z_off = {z_cur:.10f}")
            print(f"     a     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = ({phi[0]:.8f}, {phi[1]:.8f})")
            print(f"     eigenvalue = {eigenvalue:+.6e}")
            return r_cur, z_cur, a_cur, phi, True, res, res_F, False

        # Assemble DM (5x5) — every block from AD
        DM = np.zeros((5, 5))
        DM[0:2, 0:2] = J_sp                # dM1/d(r,z)
        DM[0:2, 2]   = dF_da               # dM1/da
        DM[2:4, 0]   = dJphi_dx[:, 0]      # d(J·phi)/dr  via H·phi
        DM[2:4, 1]   = dJphi_dx[:, 1]      # d(J·phi)/dz  via H·phi
        DM[2:4, 2]   = dJphi_dx[:, 2]      # d(J·phi)/da  via H·phi
        DM[2:4, 3:5] = J_sp                # dM2/dphi
        DM[4, 3:5]   = l_vec               # dM3/dphi

        cond_DM = np.linalg.cond(DM)
        print(f"         | cond(DM) = {cond_DM:.2e}")

        if cond_DM > 1e14:
            print("  !! DM ill-conditioned — aborting.")
            return r_cur, z_cur, a_cur, phi, False, res, res_F, False

        try:
            p_N = np.linalg.solve(DM, -M)
        except np.linalg.LinAlgError:
            print("  !! DM singular — aborting.")
            return r_cur, z_cur, a_cur, phi, False, res, res_F, False

        print(f"         | Newton step: dr={p_N[0]:+.4e}  dz={p_N[1]:+.4e}  da={p_N[2]:+.4e}"
              f"  |p_N|={np.linalg.norm(p_N):.4e}")

        # ── Helper: evaluate |M| at a candidate step (cheap: only F and J) ──

        def _eval_step(dy_candidate):
            dr_t = dr + dy_candidate[0]
            dz_t = dz + dy_candidate[1]
            da_t = da + dy_candidate[2]
            phi_t = phi + dy_candidate[3:5]
            if a_ref + da_t <= 0:
                return float('nan'), False
            try:
                F_t, J_t = evaluate_forces(dr_t, dz_t, da_t, md)
                M_t = np.concatenate([F_t, J_t[:, :2] @ phi_t,
                                      [np.dot(l_vec, phi_t) - 1.0]])
                return float(np.linalg.norm(M_t)), True
            except MeshFlippedError:
                return float('nan'), False

        accepted, dy_try, rho, Delta = _globalize_tr(
            M, DM, p_N, Delta, eta_accept, eta_good, Delta_max, _eval_step)

        if accepted:
            dr += dy_try[0]
            dz += dy_try[1]
            da += dy_try[2]
            phi_new = phi + dy_try[3:5]
            phi = phi_new / np.dot(l_vec, phi_new)
            print(f"         | step ACCEPTED")
        else:
            print(f"         | step REJECTED")

        # Stall check (FE noise floor on |M|). Counts BOTH accepted and
        # rejected iters — see the comment at the top of this function:
        # at the noise floor the function is flat and TR-BT rejects
        # forever at the same |M|, so an accepted-only detector misses it.
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
                r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
                return r_cur, z_cur, a_cur, phi, False, res, res_F, True

        gc.collect()

    r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
    return r_cur, z_cur, a_cur, phi, False, res, res_F, False


def _globalize_tr(M, DM, p_N, Delta, eta_accept, eta_good, Delta_max, _eval_step):
    """Powell dogleg trust-region on the Gauss-Newton model.

    Subproblem at each iter:   min_p  m(p) := ½‖M + DM·p‖²    s.t.  ‖p‖ ≤ Δ.

    The dogleg path goes from the origin to the Cauchy point p_C
    (unconstrained minimizer of m along the steepest-descent direction)
    and then on to the full Newton step p_N. The dogleg point is the
    unique intersection of this piecewise-linear path with the TR boundary;
    if p_N lies inside the TR we take it, if even p_C lies outside we clip
    along -g. Unlike plain backtracking along p_N, the *direction* changes
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
            # Even the unconstrained Cauchy point is outside TR — go to
            # the boundary along -g (steepest descent on the model).
            if g_sq > 0.0:
                dy_try = -(Delta / np.sqrt(g_sq)) * g
            else:
                dy_try = np.zeros_like(M)
            on_boundary = True
            tag         = "Cauchy-clipped"
        else:
            # Dogleg segment p(τ) = p_C + τ·(p_N - p_C), τ ∈ [0, 1].
            # Solve ‖p(τ)‖² = Δ² for the positive root.
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

        # Reduction ratio: pred from the GN model, ared from the real |M|.
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


def moore_spence_solve(r_off_eq, z_off_eq, a_init, shared_data, *, tol=1e-7, max_iter=15, md=None,
                       dr_init=0.0, dz_init=0.0, da_init=0.0):

    R, H, W, L_c, U_c, Re, G, U_m, u_2d, p_2d = shared_data
    r_ref = float(r_off_eq) - float(dr_init)
    z_ref = float(z_off_eq) - float(dz_init)
    a_ref = float(a_init)   - float(da_init)

    if md is None:
        md = setup_moving_mesh(r_ref, z_ref, a_ref, R, H, W,
                                   Re, G, U_m, u_2d, p_2d,
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
    print(f"  Start: r={r_off_eq:.6f} z={z_off_eq:.6f} a={a_init:.6f}")
    print("=" * 65)

    r_bif, z_bif, a_bif, phi_bif, ok, final_res, final_F, stalled = _ms_trial(
        float(dr_init), float(dz_init), float(da_init),
        phi_start, l_vec, md, r_ref, z_ref, a_ref, L_c, tol, max_iter)

    print(f"\n  {'=' * 50}")
    print(f"  converged={ok}  |M|={final_res:.4e}  |F|={final_F:.4e}  a={a_bif:.8f}")
    print(f"  {'=' * 50}")
    return r_bif, z_bif, a_bif, phi_bif, ok, final_F, final_res, stalled


if __name__ == "__main__":

    # Initial guess from the bifurcation diagramm
    r_off_init = 0.6098
    z_off_init = 0.0274
    a_start = 0.1375

    bg = background_flow_differentiable(R, H, W, Re)

    G, U_m, u_bar_2d, p_bar_tilde_2d = bg.solve_2D_background_flow()

    shared_data = (R, H, W, L_c, U_c, Re, G, U_m, u_bar_2d, p_bar_tilde_2d)

    header = "Newton + Moore-Spence" if USE_NEWTON_REFINEMENT else "Moore-Spence (no Newton pre-refine)"
    print("\n" + "#" * 65)
    print(f"  {header}")
    print(f"  Input: r={r_off_init:.10f}  z={z_off_init:.10f}  "
          f"a={a_start:.10f}")
    print("#" * 65)

    if USE_NEWTON_REFINEMENT:
        r, z, md_ms, dr_init_ms, dz_init_ms = newton_root_refine(
            r_off_init, z_off_init, a_start, shared_data, max_iter=10)

        _log_mesh_quality(md_ms, "newton", 0, dr=dr_init_ms, dz=dz_init_ms, da=0.0)

        print("\n" + "-" * 65)
        print("  Sanity remesh after NEWTON (fresh mesh @ converged position)")
        print("-" * 65)
        md_fresh_newton = setup_moving_mesh(
            r, z, a_start, R, H, W, Re, G, U_m,
            u_bar_2d, p_bar_tilde_2d, particle_maxh_rel, global_maxh_rel,
            a_mesh_size_res_ref=a_start)
        F_fresh = evaluate_forces(0.0, 0.0, 0.0, md_fresh_newton, jacobian=False)
        print(f"  Newton point on fresh mesh:  |F| = {np.linalg.norm(F_fresh):.4e}"
              f"  (F = [{F_fresh[0]:+.4e}, {F_fresh[1]:+.4e}])")
        _log_mesh_quality(md_fresh_newton, "newton-fresh", 0, dr=0.0, dz=0.0, da=0.0)
        del md_fresh_newton
        gc.collect()
    else:
        # Build the MS mesh straight at the initial guess; MS will resolve
        # the (r, z) shift jointly with (a, φ).
        r, z = r_off_init, z_off_init
        md_ms = setup_moving_mesh(
            r, z, a_start, R, H, W, Re, G, U_m,
            u_bar_2d, p_bar_tilde_2d, particle_maxh_rel, global_maxh_rel)
        dr_init_ms, dz_init_ms = 0.0, 0.0

    # Moore-Spence on the prepared mesh (single-shot, passes md_ms through
    # and continues from any pre-deformation set above).
    r_bif, z_bif, a_bif, phi_bif, converged, _F, _M, _stalled = moore_spence_solve(
        r, z, a_start, shared_data,
        md=md_ms, dr_init=dr_init_ms, dz_init=dz_init_ms, da_init=0.0)

    # md_ms was built around (r_off_init, z_off_init, a_start),
    # so xi at MS-end encodes the full offset from those reference values.
    dr_total = float(r_bif - r_off_init)
    dz_total = float(z_bif - z_off_init)
    da_total = float(a_bif - a_start)
    _log_mesh_quality(md_ms, "ms", 0, dr=dr_total, dz=dz_total, da=da_total)

    del md_ms
    gc.collect()

    # ---- Sanity remesh after Moore-Spence: build a fresh mesh centered at
    # the bifurcation point and re-evaluate the MS residual components.
    # If |F|, |J*phi|, eigenvalue are not ~0 on the fresh mesh, the bifurcation
    # point was a mesh-deformation artefact rather than a physical zero.
    print("\n" + "-" * 65)
    print("  Sanity remesh after MOORE-SPENCE (fresh mesh @ bifurcation point)")
    print("-" * 65)
    md_fresh_ms = setup_moving_mesh(
        r_bif, z_bif, a_bif, R, H, W, Re, G, U_m,
        u_bar_2d, p_bar_tilde_2d, particle_maxh_rel, global_maxh_rel,
        a_mesh_size_res_ref=a_start)
    F_ms_fresh, J_ms_fresh = evaluate_forces(0.0, 0.0, 0.0, md_fresh_ms)
    Jsp_fresh = J_ms_fresh[:, :2]
    Jphi_fresh = Jsp_fresh @ phi_bif
    eigs_fresh = np.linalg.eigvals(Jsp_fresh)
    eig_min_fresh = eigs_fresh[np.argmin(np.abs(eigs_fresh))]
    print(f"  MS point on fresh mesh:")
    print(f"    |F|        = {np.linalg.norm(F_ms_fresh):.4e}   (F = [{F_ms_fresh[0]:+.4e}, {F_ms_fresh[1]:+.4e}])")
    print(f"    |J*phi|    = {np.linalg.norm(Jphi_fresh):.4e}   (phi from MS, l_vec-normalized)")
    print(f"    eig(J)_min = {eig_min_fresh:+.4e}   (should cross 0 at the true bifurcation)")
    _log_mesh_quality(md_fresh_ms, "ms-fresh", 0, dr=0.0, dz=0.0, da=0.0)
    del md_fresh_ms
    gc.collect()

    print(f"\nBifurcation point: r={r_bif:.10f}  z={z_bif:.10f}  a={a_bif:.10f}")
    print(f"phi = ({phi_bif[0]:+.8f}, {phi_bif[1]:+.8f})")
    print(f"converged = {converged}")