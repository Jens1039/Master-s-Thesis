"""Shape optimization restricted to the z-symmetric pitchfork branch.

Companion to ``locate_bifurcation_points_symmetric.py``: the geometry's
mirror symmetry about the channel mid-plane forces the physics

    F_p_x(s, t) = +F_p_x(s, H_hat - t)        (radial:  t-symmetric)
    F_p_z(s, t) = -F_p_z(s, H_hat - t)        (axial:   t-antisymmetric)

on the cross-section (s, t) ∈ [0, W_hat] × [0, H_hat]. Equivalently, in
particle coords (with z=0 at the channel centerline),

    F_p_x(x, +z) = +F_p_x(x, -z)
    F_p_z(x, +z) = -F_p_z(x, -z).

Consequences exploited here:
  - F_p_z(r, 0, a) ≡ 0  ⇒  the z-equilibrium drops out and the bif
    point lives at z=0 to all orders.
  - ∂F_r/∂z = ∂F_z/∂r = 0 at z=0  ⇒  the position-Jacobian J_sp is
    diagonal and the unstable eigenvector is exactly (0, 1).
  - The full 5×5 Moore-Spence collapses to 2×2 in (r, a) with phi=(0,1)
    hardcoded (see ``locate_bifurcation_points_symmetric``).
  - The shape-adjoint multiplier is a 2-vector; the shape gradient is
    the sum of one linear derivative (λ_0 · F_p_x) and one Hessian-VP
    (λ_1 · F_p_z with TLM direction phi=(0,1)).

This module deliberately re-uses the heavy AD plumbing from
``shape_optimization`` (xi-channel lifting, Riesz solver, plotting,
checkpointing) so any fix in the asymmetric track propagates here.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import gc
from datetime import datetime
from firedrake import *
from firedrake.adjoint import (
    stop_annotating, continue_annotation,
    get_working_tape, set_working_tape, Tape,
    ReducedFunctional, Control,
)

from background_flow_differentiable import background_flow_differentiable
from perturbed_flow_differentiable import (
    _build_xi_hat, evaluate_forces, reset_ale_basis_for_step,
)
from locate_bifurcation_points_symmetric import (
    newton_root_refine_symmetric, moore_spence_solve_symmetric, PHI_SYMMETRIC,
)
from config_paper_parameters import *
from nondimensionalization import nondimensionalisation

# Re-use the asymmetric helpers — they are physics-agnostic, and the
# GenericSolveBlock Hessian-guard monkey-patch installed at module load of
# ``shape_optimization`` is applied transitively on this import.
from shape_optimization import (
    _solve_bg_on_mesh, build_xi_channel_from_T2d, eval_forces_with_bg_on_tape,
    project_z_symmetric, riesz_representative, check_2d_mesh_quality,
    save_cross_section_plot, extract_deformed_boundary, save_optimized_section,
)


def compute_DG_at_bif_symmetric(r_bif, a_bif, mesh_data, r_ref, a_ref):
    """2×2 linearisation of G = (F_r, J_zz) at (r_bif, 0, a_bif), phi=(0,1).

        DG = [[∂F_r/∂r,   ∂F_r/∂a ],
              [∂J_zz/∂r,  ∂J_zz/∂a]]

    The cross-block ∂F_r/∂z, ∂F_z/∂r vanishes *exactly* by z-reflection
    symmetry (not just to FE noise), so this 2×2 is the full linearisation
    on the symmetric branch.
    """
    dr = float(r_bif - r_ref)
    da = float(a_bif - a_ref)

    F_base, J_full, dJphi_dx = evaluate_forces(
        dr, 0.0, da, mesh_data, hessian_phi=PHI_SYMMETRIC)

    # phi=(0,1) ⇒ J·phi = ∂F/∂z; its z-component is J_zz = ∂F_z/∂z.
    DG = np.array([
        [float(J_full[0, 0]),   float(J_full[0, 2])],     # ∂F_r/∂r, ∂F_r/∂a
        [float(dJphi_dx[1, 0]), float(dJphi_dx[1, 2])],   # ∂J_zz/∂r, ∂J_zz/∂a
    ])
    return DG, F_base


def compute_shape_gradient_symmetric(r_bif, a_bif, a_target, DG_2x2,
                                      T_2d, mesh2d, X_ref_2d,
                                      R_hat, H_hat, W_hat, Re_float,
                                      mesh_data, r_ref, a_ref):
    """Shape gradient for the symmetric reduction.

        L  = J(y) + λ^T G(y, T),   y = (r, a),   G = (F_r, J_zz)
        ∂L/∂y = 0   ⇒   DG_y^T λ = ∂J/∂y = (0, 2(a − a_target))
        dJ/dT       = λ_0 · ∂F_r/∂T   +   λ_1 · ∂J_zz/∂T
                    = ∂(λ_0 F_p_x)/∂T + ∂² (λ_1 F_p_z)/(∂T ∂z)
                                          ^ Hessian-VP in TLM dir (0,1)
    """
    rhs_adj = np.array([0., 2. * float(a_bif - a_target)])
    cond_DG = np.linalg.cond(DG_2x2)
    print(f"  [Shape] cond(DG_2x2) = {cond_DG:.3e}")

    lambda_adj = np.linalg.solve(DG_2x2.T, rhs_adj)
    lam_str = ', '.join(f'{v:.4e}' for v in lambda_adj)
    print(f"  [Shape] lambda_adj = [{lam_str}]")

    lam0 = float(lambda_adj[0])    # multiplies F_r → linear derivative
    lam1 = float(lambda_adj[1])    # multiplies J_zz → mixed Hessian d²F_z/(dT ∂z)

    set_working_tape(Tape())
    continue_annotation()

    c_T = Control(T_2d)
    mesh2d.coordinates.assign(X_ref_2d + T_2d)

    G_hat_val, _, u_bar_2d_new, p_bar_2d_new = _solve_bg_on_mesh(
        mesh2d, R_hat, H_hat, W_hat, Re_float)
    print(f"  [Shape] G_hat = {G_hat_val:.6e}")

    print(f"  [Shape] Evaluating forces at y* = "
          f"(r={r_bif:.4f}, z=0, a={a_bif:.4f})...")
    F_p_x, F_p_z, dr_fn, dz_fn = eval_forces_with_bg_on_tape(
        r_bif, 0.0, a_bif, u_bar_2d_new, p_bar_2d_new, G_hat_val,
        mesh_data, r_ref, 0.0, a_ref,
        T_2d=T_2d, mesh2d=mesh2d, rz_on_tape=True)
    print(f"  [Shape] F(y*): F_x = {float(F_p_x):.4e}, "
          f"F_z = {float(F_p_z):.4e}  (F_z = FE noise; ≡0 in exact arithmetic)")

    c_r = Control(dr_fn)
    c_z = Control(dz_fn)        # kept ONLY to carry the phi_z TLM input
    controls = [c_r, c_z, c_T]

    # ── Term 1: ∂(λ_0 F_r)/∂T  via plain reverse-mode derivative ──
    Lambda1 = lam0 * F_p_x
    print("  [Shape] Computing G1 shape gradient (linear derivative)...")
    rf_1 = ReducedFunctional(Lambda1, controls)
    d1   = rf_1.derivative()
    shape_grad = d1[2]

    # ── Term 2: ∂(λ_1 · ∂F_z/∂z)/∂T via Hessian-VP, TLM dir phi=(0,1) ──
    # m_dot[1] = 1.0 propagates ∂/∂z; m_dot[2] = None tells pyadjoint to
    # skip the T-tape entirely (we want d²/(dT ∂z), NOT d²/dT² — passing
    # a zero Function would force every block downstream of T_2d to be
    # walked, and expand_derivatives chokes on CoordinateDerivative(zero)
    # inside the BG NS SolveBlock). Same trick as the asymmetric path.
    with stop_annotating():
        R_space  = dr_fn.function_space()
        phi_r_fn = Function(R_space).assign(0.0)
        phi_z_fn = Function(R_space).assign(1.0)
    m_dot = [phi_r_fn, phi_z_fn, None]

    Lambda2 = lam1 * F_p_z
    print("  [Shape] Computing G2 shape gradient (Hessian-VP, phi=(0,1))...")
    rf_2 = ReducedFunctional(Lambda2, controls)
    rf_2.derivative()                  # pyadjoint requires .derivative()
    H2   = rf_2.hessian(m_dot)         #   before .hessian()
    shape_grad.dat.data[:] += np.asarray(H2[2].dat.data_ro)

    stop_annotating()
    get_working_tape().clear_tape()
    gc.collect()

    # Restore meshes — the next LS trial will reassign, but a clean state
    # here avoids surprises in any in-between diagnostic reading coords.
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)
        mesh3d_r = mesh_data['mesh3d']
        R_sp = FunctionSpace(mesh3d_r, "R", 0)
        dr_r = Function(R_sp).assign(float(r_bif - r_ref))
        dz_r = Function(R_sp).assign(0.0)
        da_r = Function(R_sp).assign(float(a_bif - a_ref))
        xi_restore = _build_xi_hat(dr_r, dz_r, da_r, mesh_data)
        mesh3d_r.coordinates.assign(
            mesh_data['X_ref'] + mesh_data['xi_baseline']
            + xi_restore + mesh_data['xi_channel'])

    return shape_grad, lambda_adj


def run_shape_optimization_symmetric(a_target, shared_data, mesh_data_init,
                                      bif_result_init,
                                      *,
                                      r_ref, a_ref,
                                      max_steps=50, tol_J=1e-8,
                                      alpha_step=0.1, alpha_min=1e-8,
                                      alpha_backtrack=0.5, max_backtrack=12,
                                      Delta_max=1e-1, branch_C=1.0,
                                      riesz_alpha=1.0, riesz_beta=1e-2,
                                      riesz_mu_cr=100.0,
                                      ms_tol=1e-12, ms_max_iter=30,
                                      n_grid_2d=128,
                                      plot_dir=None):
    """Algorithm 4.1, z-symmetric branch.

    Identical control flow to ``run_shape_optimization`` but with:
      - state (r_bif, a_bif); z_bif ≡ 0, phi_bif ≡ (0, 1) fixed by symmetry,
      - 2×2 DG and 2-vector adjoint multiplier λ,
      - inner MS via the 2×2 symmetric solver,
      - z-symmetric projection of V_rep at every step (MANDATORY here —
        the reduction's premise is that T_2d stays mirror-symmetric).
    """
    R_hat, H_hat, W_hat, L_c, U_c, Re, _, _, _, _ = shared_data

    if plot_dir is None:
        plot_dir = ("images/shape_opt_sym_run_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"  [ShapeOpt-Sym] Cross-section snapshots → {plot_dir}/")

    mesh2d = RectangleMesh(n_grid_2d, n_grid_2d, W_hat, H_hat,
                            quadrilateral=False, diagonal="crossed",
                            comm=COMM_WORLD)
    V_2d   = VectorFunctionSpace(mesh2d, "CG", 1)
    T_2d   = Function(V_2d, name="T_2d")
    with stop_annotating():
        X_ref_2d = Function(V_2d, name="X_ref_2d")
        X_ref_2d.interpolate(SpatialCoordinate(mesh2d))

    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d)
        V_dg0_ref = FunctionSpace(mesh2d, "DG", 0)
        J_ref     = Function(V_dg0_ref)
        J_ref.interpolate(JacobianDeterminant(mesh2d))
        ref_signs_2d = np.sign(np.asarray(J_ref.dat.data_ro)).copy()

    r_bif = float(bif_result_init['r'])
    a_bif = float(bif_result_init['a'])

    md         = mesh_data_init
    shared_cur = shared_data

    print(f"\n{'#'*70}")
    print(f"#  SHAPE OPTIMISATION  (Algorithm 4.1, z-SYMMETRIC)")
    print(f"#  Target:  a_hat* = {a_target:.4f}")
    print(f"#  Initial: a_hat* = {a_bif:.6f}  (z = 0 fixed, phi = (0, 1))")
    print(f"{'#'*70}")

    history   = []
    converged = False

    eta_accept = 0.1
    # eta_good lowered from 0.75: in log_2 the first accepted steps had
    # ρ ≈ 0.49, sitting in the "no growth" band (between eta_accept and
    # eta_good) and clamping Delta forever at alpha_step. With eta_good=0.4
    # the TR auto-promotes Delta on these steady-good steps, and the
    # eta_accept safety net still catches over-aggressive trials. Pure
    # control-flow change, no compute pipeline modification.
    eta_good   = 0.4
    alpha_seed = min(float(alpha_step), float(Delta_max))

    prev_r_bif = prev_a_bif = None

    for step in range(max_steps):
        J = (a_bif - a_target) ** 2

        print(f"\n{'='*65}")
        print(f"  STEP {step+1:3d}  |  a_bif = {a_bif:.8f}  |  J = {J:.6e}")
        if prev_a_bif is not None:
            print(f"  Δ since prev step:  "
                  f"da = {a_bif - prev_a_bif:+.3e}  "
                  f"dr = {r_bif - prev_r_bif:+.3e}")
        print(f"{'='*65}")
        prev_r_bif, prev_a_bif = float(r_bif), float(a_bif)

        history.append({
            'step':  step,
            'a_bif': a_bif,
            'J':     J,
            'alpha': None,
            'trials': [],
        })

        if plot_dir is not None:
            save_cross_section_plot(step + 1, mesh2d, X_ref_2d, T_2d,
                                    W_hat, H_hat, a_bif, J, plot_dir)

        if J < tol_J:
            print(f"\n  CONVERGED: J = {J:.4e} < tol = {tol_J:.4e}")
            converged = True
            break

        DG_2x2, F_at_bif = compute_DG_at_bif_symmetric(
            r_bif, a_bif, md, r_ref, a_ref)
        F_r_val = float(F_at_bif[0])
        F_z_val = float(F_at_bif[1])
        print(f"  |F_r(y*)| = {abs(F_r_val):.4e}  "
              f"|F_z noise| = {abs(F_z_val):.4e}  "
              f"(should be ~0 and pure FE noise resp.)")
        if abs(F_r_val) > 1e-4:
            print("  WARNING: large |F_r| — bifurcation point may have drifted")

        # ── Drift diagnostics (no dz — fixed at 0) ──
        with stop_annotating():
            xi_b_data = np.asarray(md['xi_baseline'].dat.data_ro)
            xi_c_data = np.asarray(md['xi_channel'].dat.data_ro)
            xi_b_max  = float(np.max(np.linalg.norm(xi_b_data, axis=1)))
            xi_c_max  = float(np.max(np.linalg.norm(xi_c_data, axis=1)))

            V_dg0_3d = FunctionSpace(md['mesh3d'], "DG", 0)
            F_def    = Identity(3) + grad(md['xi_baseline'] + md['xi_channel'])
            detF     = Function(V_dg0_3d)
            detF.interpolate(det(F_def))
            detF_arr = np.asarray(detF.dat.data_ro)
            detF_min = float(detF_arr.min())
            detF_max = float(detF_arr.max())
            n_inv    = int(np.sum(detF_arr <= 0.0))

        print(f"  [Drift] ||xi_baseline||_max = {xi_b_max:.3e}  "
              f"||xi_channel||_max = {xi_c_max:.3e}")
        print(f"  [Drift] 3D det(I+∇xi) range = "
              f"[{detF_min:+.3e}, {detF_max:+.3e}]  inverted cells = {n_inv}")

        shape_grad, lambda_adj = compute_shape_gradient_symmetric(
            r_bif, a_bif, a_target, DG_2x2,
            T_2d, mesh2d, X_ref_2d, R_hat, H_hat, W_hat, Re, md,
            r_ref, a_ref)

        V_rep = riesz_representative(
            shape_grad, mesh2d,
            alpha_elast=riesz_alpha, beta_l2=riesz_beta,
            mu_cr=riesz_mu_cr,
            fix_corners=True, W_hat=W_hat, H_hat=H_hat)

        # MANDATORY z-symmetric projection. The shape gradient is in the
        # symmetric subspace by physics (F_r symmetric, J_zz symmetric in T),
        # so this only kills FE-noise antisymmetric components. Without it,
        # FE noise accumulates over outer steps and eventually breaks the
        # F_z ≡ 0 / J_off-diag = 0 premise of the 2×2 reduction.
        V_rep, max_anti, mirror_dist = project_z_symmetric(
            V_rep, mesh2d, H_hat)
        print(f"  [Symm] z-symmetric projection: "
              f"max antisym killed = {max_anti:.3e}  "
              f"max mirror-lookup dist = {mirror_dist:.2e}")

        with stop_annotating():
            grad_norm_sq_L2 = float(assemble(inner(V_rep, V_rep) * dx))
            Bv = as_vector([V_rep[0].dx(0) - V_rep[1].dx(1),
                            V_rep[0].dx(1) + V_rep[1].dx(0)])
            grad_norm_sq_M = float(assemble(
                riesz_alpha  * inner(grad(V_rep), grad(V_rep)) * dx
              + riesz_mu_cr  * inner(Bv, Bv) * dx
              + riesz_beta   * inner(V_rep, V_rep) * dx))
        grad_norm_L2 = float(grad_norm_sq_L2 ** 0.5)
        grad_norm_M  = float(grad_norm_sq_M  ** 0.5)
        print(f"  ||V_rep||_L2 = {grad_norm_L2:.4e}   "
              f"||V_rep||_M = {grad_norm_M:.4e}  (CR metric)")

        if grad_norm_L2 < 1e-14:
            print("  Gradient effectively zero — cannot proceed.")
            break

        with stop_annotating():
            mesh2d.coordinates.assign(X_ref_2d + T_2d)
            try:
                V_dg0 = FunctionSpace(mesh2d, "DG", 0)
                J_pre = Function(V_dg0)
                J_pre.interpolate(JacobianDeterminant(mesh2d))
                j_arr   = np.asarray(J_pre.dat.data_ro)
                abs_min = float(np.abs(j_arr).min())
                abs_max = float(np.abs(j_arr).max())
                n_flip  = int(np.sum(np.sign(j_arr) != ref_signs_2d))
                print(f"  [pre-LS] mesh2d |J| range = "
                      f"[{abs_min:.3e}, {abs_max:.3e}]  flipped cells={n_flip}  "
                      f"||T_2d||={float(norm(T_2d)):.3e}")
            except Exception as e:
                print(f"  [pre-LS] Jacobian check skipped ({e})")

        print(f"\n  TR backtracking (Delta_0 = {alpha_seed:.3e}, "
              f"Delta_max = {Delta_max:.3e}, "
              f"eta_accept = {eta_accept}, eta_good = {eta_good}, "
              f"branch_C = {branch_C:.2g})...")
        Delta    = float(alpha_seed)
        accepted = False
        accepted_rho = float('nan')

        for bt in range(max_backtrack):
            if Delta < alpha_min:
                print(f"  [TR-BT] Delta {Delta:.3e} < alpha_min — aborting.")
                break

            alpha = Delta

            with stop_annotating():
                T_2d_try = T_2d.copy(deepcopy=True)
                T_2d_try.dat.data[:] += alpha * V_rep.dat.data_ro

            with stop_annotating():
                mesh2d.coordinates.assign(X_ref_2d + T_2d_try)
                mesh_ok = check_2d_mesh_quality(mesh2d,
                                                ref_signs=ref_signs_2d)

            if not mesh_ok:
                print(f"  [TR-BT bt={bt}] Mesh quality failed, shrinking Delta.")
                Delta *= alpha_backtrack
                continue

            try:
                with stop_annotating():
                    G_try, U_m_try, u_bar_try, p_bar_try = \
                        _solve_bg_on_mesh(
                            mesh2d, R_hat, H_hat, W_hat, Re)
            except Exception as e:
                print(f"  [TR-BT bt={bt}] Background flow failed: {e}")
                Delta *= alpha_backtrack
                continue

            shared_try = (R_hat, H_hat, W_hat, L_c, U_c, Re,
                          G_try, U_m_try, u_bar_try, p_bar_try)

            md_try = dict(md)
            md_try['u_bar_2d_hat']       = u_bar_try
            md_try['p_bar_tilde_2d_hat'] = p_bar_try
            md_try['G_hat']              = G_try
            md_try['U_m_hat']            = U_m_try

            # build_xi_channel_from_T2d does a VOM lookup at query points
            # clipped to the REFERENCE rectangle [0, W_hat] x [0, H_hat]. The
            # BG NS solve above set mesh2d.coords to the DEFORMED cross-section
            # — with larger T_2d_try the deformed wall moves out of the
            # reference rectangle and VOM raises VertexOnlyMeshMissingPointsError.
            # Swap mesh2d coords to reference for the xi_channel build; coords
            # will be re-assigned at the next iter's pre-LS check anyway.
            with stop_annotating():
                mesh2d.coordinates.assign(X_ref_2d)
                xi_ch_try = build_xi_channel_from_T2d(
                    T_2d_try, md['mesh3d'], md['X_ref'], mesh2d,
                    md['R_hat'], md['W_hat'], md['H_hat'])
                xi_ch_static = Function(md['V_def'], name="xi_channel_static")
                xi_ch_static.dat.data[:] = xi_ch_try.dat.data_ro
            md_try['xi_channel'] = xi_ch_static

            print(f"  [TR-BT bt={bt}] alpha = {alpha:.3e} → running symmetric MS...")
            try:
                with stop_annotating():
                    r_try, a_try, conv_try, F_norm_try = \
                        moore_spence_solve_symmetric(
                            r_bif, a_bif, shared_try,
                            tol=ms_tol, max_iter=ms_max_iter,
                            md=md_try,
                            dr_init=float(r_bif - r_ref),
                            da_init=float(a_bif - a_ref))
            except Exception as e:
                print(f"  [TR-BT bt={bt}] Symmetric MS failed: {e}")
                history[-1]['trials'].append({
                    'alpha': alpha, 'conv': False,
                    'F_norm': float('nan'), 'a_try': float('nan'),
                    'note': f'exception: {e}',
                })
                Delta *= alpha_backtrack
                continue

            history[-1]['trials'].append({
                'alpha':  alpha,
                'conv':   bool(conv_try),
                'F_norm': float(F_norm_try),    # |F_r| at MS terminal iterate
                'a_try':  float(a_try),
            })

            if not conv_try:
                print(f"  [TR-BT bt={bt}] Symmetric MS did not converge.")
                Delta *= alpha_backtrack
                continue

            # ── Branch tracking (Boullé–Farrell–Paganini eq. 4.3) ──
            try:
                u_old_arr = np.asarray(shared_cur[8].dat.data_ro, dtype=float)
                u_new_arr = np.asarray(u_bar_try.dat.data_ro,     dtype=float)
                if u_old_arr.shape == u_new_arr.shape:
                    delta_u = float(np.linalg.norm(u_new_arr - u_old_arr))
                    base_u  = float(np.linalg.norm(u_new_arr))
                    branch_ratio = delta_u / max(base_u, 1e-30)
                    print(f"  [Branch] ‖u_new − u_old‖/‖u_new‖ = "
                          f"{branch_ratio:.3e}  (C = {branch_C:.2g})")
                    if branch_ratio > branch_C:
                        print(f"  [TR-BT bt={bt}] Branch-tracking FAILED "
                              f"(ratio {branch_ratio:.3e} > {branch_C:.2g}) "
                              f"— shrinking Delta.")
                        Delta *= alpha_backtrack
                        continue
                else:
                    print(f"  [Branch] DOF-shape mismatch "
                          f"({u_old_arr.shape} vs {u_new_arr.shape}) — check skipped.")
            except Exception as e:
                print(f"  [Branch] check skipped ({e})")

            J_try = (float(a_try) - float(a_target)) ** 2

            pred = alpha * grad_norm_sq_M
            if pred > 1e-30:
                rho = (J - J_try) / pred
            else:
                rho = -1.0 if J_try > J else 1.0
            print(f"  [TR-BT bt={bt}] a_try = {a_try:.6f},  "
                  f"J_try = {J_try:.4e}  (vs J = {J:.4e})  rho = {rho:+.4f}")

            if rho >= eta_accept:
                print(f"  [TR-BT] Step ACCEPTED (bt={bt}, alpha={alpha:.4e}, "
                      f"rho={rho:+.4f})")
                T_2d.assign(T_2d_try)
                r_bif      = float(r_try)
                a_bif      = float(a_try)
                md         = md_try
                shared_cur = shared_try
                history[-1]['alpha'] = alpha
                history[-1]['rho']   = rho
                accepted = True
                accepted_rho = rho
                break
            else:
                Delta *= alpha_backtrack

        if not accepted:
            print(f"  [ShapeOpt-Sym] TR backtracking failed — stopping.")
            break

        prev_seed = alpha_seed
        if accepted_rho > eta_good:
            alpha_seed = min(2.0 * Delta, float(Delta_max))
        else:
            alpha_seed = min(Delta, float(Delta_max))
        if alpha_seed != prev_seed:
            print(f"  [TR-BT] Delta_next: {prev_seed:.3e} → {alpha_seed:.3e}  "
                  f"(rho={accepted_rho:+.4f})")

        trials = history[-1]['trials']
        failed_F   = [t['F_norm'] for t in trials if not t['conv']]
        accepted_F = [t['F_norm'] for t in trials if t['conv']]
        drift_dr = float(r_bif - r_ref)
        drift_da = float(a_bif - a_ref)
        if failed_F:
            failed_str = ", ".join(f"{x:.2e}" for x in failed_F)
        else:
            failed_str = "—"
        accepted_str = f"{accepted_F[-1]:.2e}" if accepted_F else "—"
        print(f"\n  [DIAG step {step+1}] drift (dr, da) from r_ref: "
              f"({drift_dr:+.2e}, {drift_da:+.2e})  (dz fixed at 0)")
        print(f"  [DIAG step {step+1}] plateau |F_r| of failed trials: [{failed_str}]"
              f"  | accepted |F_r|: {accepted_str}")

        # ── Snap-and-reset: absorb (dr, 0, da) into xi_baseline and
        # re-solve basis around the new bif.
        print(f"  [Snap] Re-solving ALE basis around new bif point...")
        reset_ale_basis_for_step(md, drift_dr, 0.0, drift_da)
        r_ref, a_ref = float(r_bif), float(a_bif)

        with stop_annotating():
            mesh2d.coordinates.assign(X_ref_2d + T_2d)

    J_final = (float(a_bif) - float(a_target)) ** 2

    print(f"\n{'#'*70}")
    print(f"#  SHAPE OPTIMISATION FINISHED (SYMMETRIC)")
    print(f"#  Steps run:    {step + 1}")
    print(f"#  a_bif_final:  {a_bif:.8f}")
    print(f"#  J_final:      {J_final:.6e}")
    print(f"#  Converged:    {converged}")
    print(f"{'#'*70}")

    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)

    if plot_dir is not None:
        save_cross_section_plot(step + 2, mesh2d, X_ref_2d, T_2d,
                                W_hat, H_hat, a_bif, J_final, plot_dir)

    return {
        'T_2d':      T_2d,
        'mesh2d':    mesh2d,
        'X_ref_2d':  X_ref_2d,
        'a_bif':     a_bif,
        'r_bif':     r_bif,
        'z_bif':     0.0,                       # fixed by symmetry
        'phi_bif':   np.array([0.0, 1.0]),      # fixed by symmetry
        'J_final':   J_final,
        'history':   history,
        'converged': converged,
        'shared_data_final': shared_cur,
        'mesh_data_final':   md,
    }


def run_from_main_symmetric(r0, a_target, max_steps=50, tol_J=1e-8,
                             alpha_step=0.05, riesz_alpha=1.0,
                             riesz_beta=1e-2, riesz_mu_cr=100.0,
                             ms_tol=1e-12, ms_max_iter=15, n_grid_2d=128,
                             a0=0.1375):
    """Top-level driver matching ``shape_optimization.run_from_main`` but
    without z arguments (z=0 fixed by symmetry) and using the symmetric
    Newton + Moore-Spence path.
    """
    R_hat, H_hat, W_hat, _, L_c, U_c, Re = nondimensionalisation(
        R, H, W, a, Q, rho, mu, print_values=True)

    with stop_annotating():
        bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
        G_hat, U_m_hat, u_bar_2d, p_bar_tilde = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re,
                   G_hat, U_m_hat, u_bar_2d, p_bar_tilde)

    r_eq, md0, dr0 = newton_root_refine_symmetric(
        r0, a0, shared_data, tol=1e-10, max_iter=15)

    r_ref, a_ref = r0, a0

    with stop_annotating():
        r_bif, a_bif, conv_ms, F_norm0 = moore_spence_solve_symmetric(
            r_eq, a0, shared_data,
            tol=ms_tol, max_iter=ms_max_iter,
            md=md0, dr_init=dr0, da_init=0.0)
        print(f"  Initial bifurcation residual: |F_r| = {F_norm0:.3e}")

    if not conv_ms:
        raise RuntimeError(
            "Symmetric Moore-Spence did not converge for initial domain")

    bif_init = {'r': r_bif, 'a': a_bif, 'converged': True}

    print(f"\n  Initial bifurcation: a_hat = {a_bif:.6f} (target = {a_target:.4f})")

    result = run_shape_optimization_symmetric(
        a_target, shared_data, md0, bif_init,
        r_ref=r_ref, a_ref=a_ref,
        max_steps=max_steps, tol_J=tol_J,
        alpha_step=alpha_step,
        riesz_alpha=riesz_alpha, riesz_beta=riesz_beta,
        riesz_mu_cr=riesz_mu_cr,
        ms_tol=ms_tol, ms_max_iter=ms_max_iter,
        n_grid_2d=n_grid_2d)

    T_final  = result['T_2d']
    out_path = "output_shape_T2d_symmetric.h5"
    try:
        with CheckpointFile(out_path, "w") as chk:
            chk.save_function(T_final, name="T_2d")
        print(f"\n  Saved T_2d to {out_path}")
    except Exception as e:
        print(f"  Warning: could not save checkpoint: {e}")

    try:
        save_optimized_section(
            result,
            R=R, H=H, W=W, Q=Q, rho=rho, mu=mu,
            R_hat=R_hat, H_hat=H_hat, W_hat=W_hat,
            L_c=L_c, U_c=U_c, Re=Re,
            a_target=a_target,
            n_grid_2d=n_grid_2d,
            filepath="output_optimized_section_symmetric.pkl",
        )
    except Exception as e:
        print(f"  Warning: could not save optimized-section pickle: {e}")

    return result


if __name__ == "__main__":

    a_target = 0.1

    # Initial guess from the bifurcation diagram, z=0 hardcoded.
    r_off_hat_init = 0.6098
    a_hat_init     = 0.1375

    print("\nparticle_maxh_rel = ", particle_maxh_rel)
    print("global_maxh_rel = ",   global_maxh_rel)

    result = run_from_main_symmetric(
        a_target=a_target,
        r0=r_off_hat_init,
        a0=a_hat_init,
        # At ~5e-4 da/step (observed in log_2 step 1), going from a≈0.136
        # to a=0.1 takes ~72 steps. max_steps=100 is a defensive budget.
        max_steps=100,
        tol_J=1e-8,
        # Conservative parameters for the direction-validation run. The
        # aggressive variant (riesz_alpha=1, alpha_step=5e-2) caused 50× too
        # large T_2d_try and MS chased the new bif across (r,a) for many
        # TR-clipped Newton steps. Back to original values first to clean
        # up the direction question.
        alpha_step=1e-2,
        riesz_alpha=10.0,
        riesz_beta=1e-2,
        riesz_mu_cr=100.0,
        ms_tol=1e-12,
        ms_max_iter=30,
        n_grid_2d=120,
    )

    print("\nFinal result:")
    print(f"  a_bif = {result['a_bif']:.8f}")
    print(f"  J     = {result['J_final']:.6e}")
    print(f"  steps = {len(result['history'])}")
