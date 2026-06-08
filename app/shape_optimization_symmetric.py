"""Shape optimization restricted to the z-symmetric pitchfork branch.

Companion to ``locate_bifurcation_points_symmetric.py``: the geometry's
mirror symmetry about the channel mid-plane forces the physics

    F_p_x(s, t) = +F_p_x(s, H - t)        (radial:  t-symmetric)
    F_p_z(s, t) = -F_p_z(s, H - t)        (axial:   t-antisymmetric)

on the cross-section (s, t) ∈ [0, W] × [0, H]. Equivalently, in
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
from perturbed_flow_differentiable import _build_xi, evaluate_forces, reset_ale_basis_for_step
from locate_bifurcation_points_symmetric import newton_root_refine_symmetric, moore_spence_solve_symmetric, PHI_SYMMETRIC
from problem_setup import *

# Re-use the asymmetric helpers — they are physics-agnostic, and the
# GenericSolveBlock Hessian-guard monkey-patch installed at module load of
# ``shape_optimization`` is applied transitively on this import.
import shape_optimization as _sopt
from shape_optimization import (
    _solve_bg_on_mesh, eval_forces_with_bg_on_tape,
    riesz_representative, riesz_metric_form,
    check_2d_mesh_quality, mesh2d_cell_quality,
    save_cross_section_plot, extract_deformed_boundary, save_optimized_section,
)


def project_z_symmetric(V_rep, mesh2d, H):
    """Project a 2D shape-deformation field onto the z-symmetric subspace
    about the channel mid-plane z=H/2.

    A z-symmetric deformation satisfies
        V_x(x, z) = +V_x(x, H - z)   (x-component symmetric)
        V_z(x, z) = -V_z(x, H - z)   (z-component antisymmetric: walls bulge in
                                      mirror image, channel-symmetric shape)

    Rationale: the underlying problem has a (z, -z) Bif-pair. Any deformation
    that breaks this symmetry drags the +z Bif toward z=0, where it
    coalesces with its mirror — a degeneracy that limits step-size and
    eventually breaks MS convergence. The CR-Riesz metric M is block-
    diagonal between sym/antisym subspaces (all M-terms are products of
    sym×antisym → integral over a symmetric domain vanishes), so projecting
    after the unrestricted Riesz solve is equivalent to solving Riesz on
    the symmetric subspace directly.

    Returns (V_sym, max_drift_killed) where max_drift_killed is the maximum
    per-node L²-norm of the antisymmetric component that was discarded — a
    diagnostic of how asymmetric V_rep was before projection.
    """
    from scipy.spatial import cKDTree
    with stop_annotating():
        coords = np.asarray(mesh2d.coordinates.dat.data_ro)
        V_arr  = np.asarray(V_rep.dat.data_ro).copy()
        mirror_coords = coords.copy()
        mirror_coords[:, 1] = H - coords[:, 1]
        _, idx = cKDTree(coords).query(mirror_coords)
        # Sanity: in a crossed-diagonal RectangleMesh the mirror lookup is
        # exact. Distances should be ~machine eps.
        max_mirror_dist = float(np.max(np.linalg.norm(
            coords[idx] - mirror_coords, axis=1)))

        V_sym = V_arr.copy()
        V_sym[:, 0] = (V_arr[:, 0] + V_arr[idx, 0]) / 2.0
        V_sym[:, 1] = (V_arr[:, 1] - V_arr[idx, 1]) / 2.0

        # Antisymmetric component (what we throw away) — for diagnostics.
        V_anti = V_arr - V_sym
        max_anti = float(np.max(np.linalg.norm(V_anti, axis=1)))

        V_out = Function(V_rep.function_space())
        V_out.dat.data[:] = V_sym
    return V_out, max_anti, max_mirror_dist

# ──────────────────────────────────────────────────────────────────────────
# Monkey-patch ``build_xi_channel_from_T2d`` so its VOM construction can be
# made robust against deformed mesh2d.coords without touching the asymmetric
# module. ``eval_forces_with_bg_on_tape`` looks the function up via the
# ``shape_optimization`` module globals at call time, so overwriting the
# module attribute here intercepts both that path and the local line-search
# call site (re-exported below).
# ──────────────────────────────────────────────────────────────────────────
from contextlib import contextmanager

_original_build_xi_channel = _sopt.build_xi_channel_from_T2d
_xi_channel_X_ref_2d = None       # set via the xi_channel_ref_swap CM
_xi_channel_restore  = True       # whether the wrapper restores mesh2d after


def _build_xi_channel_with_ref_swap(T_2d, mesh3d, X_ref, mesh2d,
                                     R, W, H):
    """Patched build_xi_channel_from_T2d. When the ``xi_channel_ref_swap``
    context manager is active *and* ``T_2d`` is non-trivially non-zero,
    performs an off-tape ``mesh2d.coordinates.assign(X_ref_2d)`` before
    delegating to the original implementation, then restores via
    .assign(saved_coords). The original VOM construction (off-tape, inside
    its own ``with stop_annotating()``) then runs on the reference
    cross-section — its element/local-coord lookup is frozen at that
    moment, so the subsequent on-tape interpolate is safe after the coord
    restoration.

    The ``T_2d``-based early-out matters: at step 1 of the shape-opt loop
    T_2d is exactly zero, so mesh2d.coords (= X_ref_2d + T_2d) are
    *logically* at reference and the swap is a no-op semantically. But the
    swap STILL triggers firedrake internal-state mutations (cache
    invalidation on the coord Function, possibly a halo sync), which
    empirically flipped the resulting shape-gradient direction at step 1
    in earlier attempts. Comparing via T_2d.dat.data is robust to function-
    space-instance differences between X_ref_2d and mesh2d.coordinates
    (which may have different DOF orderings — a naive
    np.array_equal(mesh2d.coords, X_ref_2d) check returned False in
    that case, defeating the early-out).
    """
    if _xi_channel_X_ref_2d is None:
        print("    [xi-patch] context inactive → passthrough to original")
        return _original_build_xi_channel(
            T_2d, mesh3d, X_ref, mesh2d, R, W, H)

    # Skip the swap when T_2d is effectively zero — mesh2d.coords are
    # already at reference and any state mutation here corrupts the
    # downstream AD.
    with stop_annotating():
        t_max = float(np.max(np.abs(np.asarray(T_2d.dat.data_ro))))
    if t_max < 1e-14:
        print(f"    [xi-patch] context active, T_2d max={t_max:.2e} < 1e-14 "
              f"→ no-op passthrough")
        return _original_build_xi_channel(
            T_2d, mesh3d, X_ref, mesh2d, R, W, H)

    mode = "swap+restore" if _xi_channel_restore else "swap (no restore)"
    print(f"    [xi-patch] context active, T_2d max={t_max:.2e} → {mode}")

    if _xi_channel_restore:
        with stop_annotating():
            saved_coords = mesh2d.coordinates.copy(deepcopy=True)
            mesh2d.coordinates.assign(_xi_channel_X_ref_2d)
        xi = _original_build_xi_channel(
            T_2d, mesh3d, X_ref, mesh2d, R, W, H)
        with stop_annotating():
            mesh2d.coordinates.assign(saved_coords)
    else:
        with stop_annotating():
            mesh2d.coordinates.assign(_xi_channel_X_ref_2d)
        xi = _original_build_xi_channel(
            T_2d, mesh3d, X_ref, mesh2d, R, W, H)
        # Leave mesh2d at reference — caller manages subsequent state.

    return xi


# Install the patch in the shape_optimization module globals so callers
# inside that module (eval_forces_with_bg_on_tape) pick it up.
_sopt.build_xi_channel_from_T2d = _build_xi_channel_with_ref_swap

# Re-export the patched symbol locally for direct call sites in this module
# (e.g. the line-search path in run_shape_optimization_symmetric).
build_xi_channel_from_T2d = _build_xi_channel_with_ref_swap


@contextmanager
def xi_channel_ref_swap(X_ref_2d, restore=True):
    """Context manager: while active, every build_xi_channel_from_T2d call
    swaps mesh2d.coordinates to X_ref_2d (off-tape) for its VOM construction.

    The ``restore`` parameter controls what happens to mesh2d.coords AFTER
    the build_xi_channel call returns:

    - ``restore=True`` (default, for on-tape callers): mesh2d.coords are
      restored to their pre-swap state via ``.assign(saved_coords)``. Use
      this in ``compute_shape_gradient_symmetric`` — the BG NS solve that
      ran BEFORE the swap captured mesh2d at the deformed state, and the
      pyadjoint AD backward pass of that solve evaluates forms using LIVE
      mesh.coords. If we leave mesh2d at reference, the backward solve
      assembles the wrong (reference-frame) adjoint matrix and the
      gradient is corrupted.

    - ``restore=False`` (for off-tape callers): mesh2d.coords stay at
      X_ref_2d. Use this in the line-search — the subsequent
      ``moore_spence_solve_symmetric`` evaluates forces with mesh2d in
      its current state, and log_test.txt empirically showed that MS at
      reference gives the correct bif location whereas MS at deformed
      (after a restore) gives the wrong rho sign. This matches the
      agg.txt manual swap pattern. The next outer iteration's pre-LS
      check re-assigns mesh2d.coords regardless.

    Outside the context the patched function is a transparent passthrough.
    """
    global _xi_channel_X_ref_2d, _xi_channel_restore
    prev_x, prev_r = _xi_channel_X_ref_2d, _xi_channel_restore
    _xi_channel_X_ref_2d = X_ref_2d
    _xi_channel_restore  = restore
    try:
        yield
    finally:
        _xi_channel_X_ref_2d = prev_x
        _xi_channel_restore  = prev_r


def inner_M_form(u, v, riesz_alpha, riesz_beta,
                 metric="h1", mu=None, lam=None):
    """M-bilinear pairing between two V_2d Functions in the Riesz metric.

    Mirrors EXACTLY the metric used by ``riesz_representative`` (via the
    shared ``riesz_metric_form``) so diagnostic cosines and norms live in
    the same Hilbert space as V_a itself — and so a metric switch (H¹ ↔
    elasticity) can never desynchronise the diagnostics from the solve.
    """
    return float(assemble(riesz_metric_form(
        u, v, metric=metric, alpha=riesz_alpha, beta=riesz_beta,
        mu=mu, lam=lam)))


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

    # ── Eigenmode diagnostic on the position 2×2 block ────────────────
    # J_pos = [[∂F_r/∂r, ∂F_r/∂z], [∂F_z/∂r, ∂F_z/∂z]]
    # By z-reflection symmetry the cross-terms ∂F_r/∂z, ∂F_z/∂r must be 0
    # in exact arithmetic. At a clean pitchfork: J_zz ≈ 0, J_rr stays
    # nonzero → eigenvector (0, 1) is the unstable mode. Any of:
    #   • |J_rz|, |J_zr| comparable to |J_rr|  → broken symmetry / branch-jump
    #   • |J_rr| ≈ 0 with |J_zz| nonzero       → MS has tracked a different bif
    #   • eigenvector strongly tilted from (0,1) → physics has reorganised
    # are signals that the optimiser has drifted onto a different mode.
    J_pos = np.array([
        [float(J_full[0, 0]), float(J_full[0, 1])],
        [float(J_full[1, 0]), float(J_full[1, 1])],
    ])
    j_scale = max(np.max(np.abs(J_pos)), 1e-30)
    asym_rz = abs(J_pos[0, 1]) / j_scale
    asym_zr = abs(J_pos[1, 0]) / j_scale
    eigvals, eigvecs = np.linalg.eig(J_pos)
    # Unstable mode = eigvalue closest to 0 (the one that vanishes at bif)
    i_unst = int(np.argmin(np.abs(eigvals)))
    v_unst = eigvecs[:, i_unst]
    v_unst = v_unst / np.linalg.norm(v_unst)
    if v_unst[1] < 0:
        v_unst = -v_unst                                  # canonical sign
    tilt_deg = float(np.degrees(np.arctan2(v_unst[0], v_unst[1])))
    print(f"  [Eigen] J_pos =")
    print(f"          [[{J_pos[0,0]:+.4e}, {J_pos[0,1]:+.4e}],")
    print(f"           [{J_pos[1,0]:+.4e}, {J_pos[1,1]:+.4e}]]")
    print(f"  [Eigen] symmetry-break: |J_rz|/||J|| = {asym_rz:.2e}, "
          f"|J_zr|/||J|| = {asym_zr:.2e}  (should be ≪ 1)")
    print(f"  [Eigen] eigvals = [{eigvals[0]:+.4e}, {eigvals[1]:+.4e}]  "
          f"(unstable: {eigvals[i_unst]:+.4e})")
    print(f"  [Eigen] unstable eigvec = ({v_unst[0]:+.4f}, {v_unst[1]:+.4f}), "
          f"tilt from (0,1) = {tilt_deg:+.2f}°")
    if abs(tilt_deg) > 10.0:
        print(f"  [Eigen] WARNING: unstable mode tilted >10° from (0,1) — "
              f"PHI_SYMMETRIC reduction may no longer fit the true mode")

    # phi=(0,1) ⇒ J·phi = ∂F/∂z; its z-component is J_zz = ∂F_z/∂z.
    DG = np.array([
        [float(J_full[0, 0]),   float(J_full[0, 2])],     # ∂F_r/∂r, ∂F_r/∂a
        [float(dJphi_dx[1, 0]), float(dJphi_dx[1, 2])],   # ∂J_zz/∂r, ∂J_zz/∂a
    ])
    return DG, F_base


def compute_shape_gradient_symmetric(r_bif, a_bif, a_target, DG_2x2,
                                      T_2d, mesh2d, X_ref_2d,
                                      R, H, W, Re_float,
                                      mesh_data, r_ref, a_ref,
                                      return_terms=False):
    """Shape derivative of the *bifurcation parameter* ``a_bif`` w.r.t. T.

    Returned cofunction is ∇a (not ∇J). The caller assembles the Gauss-
    Newton step (a_bif − a_target) · V_a / ‖V_a‖²_M from the Riesz lift
    V_a — this is the pseudo-inverse of the rank-1 leading-order Hessian
    2·∇a⊗∇a against the ε-scaled gradient 2ε·∇a. Doing it explicitly has
    two payoffs: (i) precision — the ε-free factor ∇a is fully resolved
    by AD whereas the ε-scaled ∇J = 2ε·∇a sinks into the noise floor
    near the optimum; and (ii) a natural unit step (α=1 places a_bif
    at a_target in one shot of the linear-a model), giving the outer
    line search a fixed scale. Convergence: GN is quadratic at a zero-
    residual problem; plain SD on J = (a − a_target)² with fixed step is
    *linear* at the metric-dependent rate (1 − 2s·g_a²), not cubic — both
    "ε³ stagnation" and "Hessian ∝ ε" are sloppy folklore that survived
    earlier drafts.

    Derivation (implicit function on the bifurcation constraint G = 0):

        a = a(T), defined implicitly by G(y, T) = (F_r, J_zz) = 0 with y=(r,a).
        L_a = a + μ^T G ,    ∂L_a/∂y = 0  ⇒  DG_y^T μ = (0, 1)
        da/dT = μ_0 · ∂F_r/∂T + μ_1 · ∂J_zz/∂T
              = ∂(μ_0 F_p_x)/∂T + ∂²(μ_1 F_p_z)/(∂T ∂z)
                                       ^ Hessian-VP in TLM dir (0,1)

    Note: ``a_target`` is kept in the signature only for callers that
    still need it for ε bookkeeping; the gradient itself no longer
    depends on it.
    """
    # ── IFT sign fix ── (was [0., 1.0])
    # Lagrange/IFT: DG_y^T μ = -e_a  and  da/dT = μ^T ∂G/∂T. The old +(0,1)
    # convention computed -da/dT (Taylor test confirmed the sign flip). Use
    # -(0,1) so μ = -DG^{-T} e_a and the gradient carries the correct sign.
    rhs_adj = np.array([0., -1.0])
    cond_DG = np.linalg.cond(DG_2x2)
    print(f"  [Shape] cond(DG_2x2) = {cond_DG:.3e}")

    lambda_adj = np.linalg.solve(DG_2x2.T, rhs_adj)
    lam_str = ', '.join(f'{v:.4e}' for v in lambda_adj)
    print(f"  [Shape] lambda_adj = [{lam_str}]")

    lam0 = float(lambda_adj[0])    # multiplies F_r → linear derivative
    lam1 = float(lambda_adj[1])    # multiplies J_zz → mixed Hessian d²F_z/(dT ∂z)

    # ══════════════════════════════════════════════════════════════════
    #  G1 — linear shape derivative ∂(λ_0 F_r)/∂T.  BG-NS solve ON tape so
    #  the ∂u_bar/∂T sensitivity is captured (DifferentiableFieldEvalBlock
    #  registers field_2d as a dependency). This part is Taylor-verified.
    # ══════════════════════════════════════════════════════════════════
    set_working_tape(Tape())
    continue_annotation()

    c_T = Control(T_2d)
    mesh2d.coordinates.assign(X_ref_2d + T_2d)

    G_val, _, u_bar_2d_new, p_bar_2d_new = _solve_bg_on_mesh(
        mesh2d, R, H, W, Re_float)
    print(f"  [Shape] G = {G_val:.6e}")

    print(f"  [Shape] Evaluating forces at y* = "
          f"(r={r_bif:.4f}, z=0, a={a_bif:.4f})...")
    with xi_channel_ref_swap(X_ref_2d, restore=False):
        F_p_x, _F_p_z_g1, dr_fn, dz_fn = eval_forces_with_bg_on_tape(
            r_bif, 0.0, a_bif, u_bar_2d_new, p_bar_2d_new, G_val,
            mesh_data, r_ref, 0.0, a_ref,
            T_2d=T_2d, mesh2d=mesh2d, rz_on_tape=True)
    print(f"  [Shape] F(y*): F_x = {float(F_p_x):.4e}, "
          f"F_z = {float(_F_p_z_g1):.4e}  (F_z = FE noise; ≡0 in exact arithmetic)")

    print("  [Shape] Computing G1 shape gradient (linear derivative)...")
    rf_1 = ReducedFunctional(lam0 * F_p_x, [Control(dr_fn), Control(dz_fn), c_T])
    shape_grad = rf_1.derivative()[2]

    g1_term = None
    if return_terms:
        with stop_annotating():
            g1_term = shape_grad.copy(deepcopy=True)

    # Snapshot the frozen BG fields — they seed the field-map tape (Stage B).
    with stop_annotating():
        Vu = u_bar_2d_new.function_space()
        Vp = p_bar_2d_new.function_space()
        ubar_data = np.asarray(u_bar_2d_new.dat.data_ro).copy()
        pbar_data = np.asarray(p_bar_2d_new.dat.data_ro).copy()

    stop_annotating()
    get_working_tape().clear_tape()

    # ══════════════════════════════════════════════════════════════════
    #  G2 — mixed shape derivative ∂J_zz/∂T = ∂²F_z/(∂T ∂z), split into
    #  two FIRST-order stages. The single-tape Hessian-VP cannot cross the
    #  BG-NS SolveBlock's mesh-coordinate Hessian: in the z-TLM direction
    #  the mesh coords carry no TLM, so they are off the Hessian-markings
    #  path and the ∂u_bar/∂T contribution is silently dropped. Splitting
    #  it uses only operations that are individually verified (force-side
    #  Hessian-VP + BG-side first-order adjoint = the G1 mechanism).
    #
    #    Stage A (force tape, u_bar/p_bar as CONTROLS, BG off-tape):
    #      H2[T]    = geometry-only G2 (xi_channel path)   → g2_geom
    #      H2[ubar] = λ1·∂J_zz/∂u_bar  (cofunction seed)   → u_adj
    #      H2[pbar] = λ1·∂J_zz/∂p_bar                       → p_adj
    #    Stage B (BG tape, T as control, BG on-tape):
    #      g2_field = ∇_T⟨u_adj, u_bar(T)⟩ + ∇_T⟨p_adj, p_bar(T)⟩
    #               = (∂u_bar/∂T)ᵀ u_adj + (∂p_bar/∂T)ᵀ p_adj
    #    G2 = g2_geom + g2_field.
    # ══════════════════════════════════════════════════════════════════
    print("  [Shape] Computing G2 shape gradient (two-stage split)...")

    # ── Stage A: force tape with frozen-BG controls ──
    set_working_tape(Tape())
    continue_annotation()

    c_T_A = Control(T_2d)
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)
        ubar_ctrl = Function(Vu, name="ubar_ctrl"); ubar_ctrl.dat.data[:] = ubar_data
        pbar_ctrl = Function(Vp, name="pbar_ctrl"); pbar_ctrl.dat.data[:] = pbar_data
    c_ubar = Control(ubar_ctrl)
    c_pbar = Control(pbar_ctrl)

    with xi_channel_ref_swap(X_ref_2d, restore=False):
        _F_x_A, F_p_z_A, dr_fn_A, dz_fn_A = eval_forces_with_bg_on_tape(
            r_bif, 0.0, a_bif, ubar_ctrl, pbar_ctrl, G_val,
            mesh_data, r_ref, 0.0, a_ref,
            T_2d=T_2d, mesh2d=mesh2d, rz_on_tape=True)

    with stop_annotating():
        R_space  = dr_fn_A.function_space()
        phi_r_fn = Function(R_space).assign(0.0)
        phi_z_fn = Function(R_space).assign(1.0)
    m_dot_A = [phi_r_fn, phi_z_fn, None, None, None]

    rf_2 = ReducedFunctional(
        lam1 * F_p_z_A,
        [Control(dr_fn_A), Control(dz_fn_A), c_T_A, c_ubar, c_pbar])
    rf_2.derivative()                  # pyadjoint requires .derivative()
    H2 = rf_2.hessian(m_dot_A)         #   before .hessian()
    with stop_annotating():
        g2_geom    = np.asarray(H2[2].dat.data_ro).copy()
        u_adj_data = np.asarray(H2[3].dat.data_ro).copy()
        p_adj_data = np.asarray(H2[4].dat.data_ro).copy()
    print(f"    [G2-split] ‖g2_geom‖={np.linalg.norm(g2_geom):.3e}  "
          f"‖u_adj‖={np.linalg.norm(u_adj_data):.3e}  "
          f"‖p_adj‖={np.linalg.norm(p_adj_data):.3e}")

    stop_annotating()
    get_working_tape().clear_tape()

    # ── Stage B: BG tape — map the field seeds back to T via the (verified)
    #    first-order BG-NS shape adjoint. ──
    set_working_tape(Tape())
    continue_annotation()

    c_T_B = Control(T_2d)
    mesh2d.coordinates.assign(X_ref_2d + T_2d)
    _Gb, _, ubar_B, pbar_B = _solve_bg_on_mesh(mesh2d, R, H, W, Re_float)

    with stop_annotating():
        u_adj_cof = Cofunction(ubar_B.function_space().dual())
        u_adj_cof.dat.data[:] = u_adj_data
        p_adj_cof = Cofunction(pbar_B.function_space().dual())
        p_adj_cof.dat.data[:] = p_adj_data

    J_bg = assemble(action(u_adj_cof, ubar_B)) + assemble(action(p_adj_cof, pbar_B))
    g2_field_cof = ReducedFunctional(J_bg, c_T_B).derivative()
    with stop_annotating():
        g2_field = np.asarray(g2_field_cof.dat.data_ro).copy()
    print(f"    [G2-split] ‖g2_field‖={np.linalg.norm(g2_field):.3e}")

    stop_annotating()
    get_working_tape().clear_tape()
    gc.collect()

    # ── Assemble G2 = geometry + field, fold into ∇a ──
    g2_total = g2_geom + g2_field
    g2_term = None
    if return_terms:
        with stop_annotating():
            g2_term = shape_grad.copy(deepcopy=True)
            g2_term.dat.data[:] = g2_total
    shape_grad.dat.data[:] += g2_total

    # Restore meshes — the next LS trial will reassign, but a clean state
    # here avoids surprises in any in-between diagnostic reading coords.
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)
        mesh3d_r = mesh_data['mesh3d']
        R_sp = FunctionSpace(mesh3d_r, "R", 0)
        dr_r = Function(R_sp).assign(float(r_bif - r_ref))
        dz_r = Function(R_sp).assign(0.0)
        da_r = Function(R_sp).assign(float(a_bif - a_ref))
        xi_restore = _build_xi(dr_r, dz_r, da_r, mesh_data)
        mesh3d_r.coordinates.assign(
            mesh_data['X_ref'] + mesh_data['xi_baseline']
            + xi_restore + mesh_data['xi_channel'])

    if return_terms:
        return shape_grad, lambda_adj, g1_term, g2_term
    return shape_grad, lambda_adj


# ══════════════════════════════════════════════════════════════════════════
# Taylor test for the shape gradient ∇a = ∂a_bif/∂T_2d
# ══════════════════════════════════════════════════════════════════════════

def _prepare_md_try_at_T(T_eval, *, mesh2d, X_ref_2d, shared_data, md):
    """Off-tape geometry+BG setup at a given T_2d, shared by the a_bif and
    the force-only evaluators.

    Deform mesh2d → BG NS solve → build xi_channel off-tape with the ref-swap
    (restore=False) → return (md_try, shared_try) carrying the perturbed
    background flow and channel deformation. This is the common front half of
    the line-search inner body.
    """
    R, H, W, L_c, U_c, Re = shared_data[:6]

    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_eval)
        G_try, U_m_try, u_bar_try, p_bar_try = _solve_bg_on_mesh(
            mesh2d, R, H, W, Re)

    shared_try = (R, H, W, L_c, U_c, Re,
                  G_try, U_m_try, u_bar_try, p_bar_try)

    md_try = dict(md)
    md_try['u_bar_2d']       = u_bar_try
    md_try['p_bar_tilde_2d'] = p_bar_try
    md_try['G']              = G_try
    md_try['U_m']            = U_m_try

    with stop_annotating():
        with xi_channel_ref_swap(X_ref_2d, restore=False):
            xi_ch_try = build_xi_channel_from_T2d(
                T_eval, md['mesh3d'], md['X_ref'], mesh2d,
                md['R'], md['W'], md['H'])
        xi_ch_static = Function(md['V_def'], name="xi_channel_static")
        xi_ch_static.dat.data[:] = xi_ch_try.dat.data_ro
        # Leave mesh2d at the REFERENCE cross-section (the build_xi_channel
        # ref-swap restore=False already left it there). With DECOUPLED_LIFT
        # the u_bar_2d sampling positions are channel-undeformed, so they must
        # locate in the reference mesh2d; u_bar_2d's deformed-solution dofs are
        # then read material-frame. (Previously this re-assigned mesh2d to the
        # deformed state for the COUPLED lift; the channel-decoupled lift needs
        # reference — consistent with the gradient's lift.)
    md_try['xi_channel'] = xi_ch_static

    return md_try, shared_try


def _evaluate_a_bif_at_T(T_eval, *, mesh2d, X_ref_2d, shared_data, md,
                         r_bif, a_bif, r_ref, a_ref,
                         ms_tol, ms_max_iter, ms_floor_F=1e-10):
    """Re-evaluate the bifurcation parameter a_bif at a given T_2d.

    Front half via ``_prepare_md_try_at_T``, then a symmetric Moore-Spence
    solve around the (r_ref, a_ref) frame. Using the same map for base and
    perturbed points makes the Taylor residual cancel the evaluation's own
    discretisation bias.

    Returns (a_try, r_try, converged, |F_r|).
    """
    md_try, shared_try = _prepare_md_try_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d,
        shared_data=shared_data, md=md)

    with stop_annotating():
        r_try, a_try, conv_try, F_norm_try, G_norm_try, stalled_try = \
            moore_spence_solve_symmetric(
                r_bif, a_bif, shared_try,
                tol=ms_tol, max_iter=ms_max_iter, md=md_try,
                dr_init=float(r_bif - r_ref), da_init=float(a_bif - a_ref))

    # Same FE-noise-floor stall acceptance as the optimiser: a stalled MS
    # whose |F_r| is already at the discretisation limit is a valid bif
    # estimate, not a failed solve.
    if (not conv_try) and stalled_try and float(F_norm_try) < ms_floor_F:
        conv_try = True

    return float(a_try), float(r_try), bool(conv_try), float(F_norm_try)


def _eval_forces_at_T(T_eval, *, mesh2d, X_ref_2d, shared_data, md,
                      r_bif, a_bif, r_ref, a_ref):
    """Evaluate the building-block forces (F_r, J_zz) at a given T_2d, with
    the bifurcation state FROZEN at (r_bif, 0, a_bif).

    Same off-tape geometry+BG setup as the a_bif evaluator, but instead of
    re-solving Moore-Spence it calls ``evaluate_forces`` once at the fixed
    state — so finite-differencing this over T isolates ∂F_r/∂T and
    ∂J_zz/∂T (the AD building blocks that feed G1 and G2), independent of the
    IFT multiplier algebra.

    Returns (F_r, J_zz).
    """
    md_try, _shared_try = _prepare_md_try_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d,
        shared_data=shared_data, md=md)

    dr = float(r_bif - r_ref)
    da = float(a_bif - a_ref)
    with stop_annotating():
        F_base, J_full, _dJphi = evaluate_forces(
            dr, 0.0, da, md_try, hessian_phi=PHI_SYMMETRIC)
    F_r  = float(F_base[0])
    J_zz = float(J_full[1, 1])
    return F_r, J_zz


def _eval_ontape_Fr_at_T(T_eval, *, mesh2d, X_ref_2d, shared_data, md,
                         r_bif, a_bif, r_ref, a_ref, lift_on_tape=False):
    """Forward-evaluate the radial force F_r on the SAME map the GRADIENT
    differentiates: ``eval_forces_with_bg_on_tape`` with the ref-swap at
    restore=True (the on-tape path), run purely forward under
    ``stop_annotating``.

    Finite-differencing this over T isolates the on-tape ∂F_r/∂T — to be
    compared against both the AD value (G1/λ_0) and the off-tape FD
    (``_eval_forces_at_T``). If on-tape-FD ≈ AD but ≠ off-tape-FD, the AD is
    a correct derivative of its own map and the bug is the on-tape↔off-tape
    map inconsistency (restore=True vs restore=False), not the AD itself.

    Only F_r — J_zz = ∂F_z/∂z is too close to the F_z noise floor (~2e-7) to
    finite-difference cleanly on this path (which is exactly why the gradient
    uses an AD Hessian-VP for it).
    """
    R, H, W, L_c, U_c, Re = shared_data[:6]
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_eval)
        G_val, _Um, u_bar_new, p_bar_new = _solve_bg_on_mesh(
            mesh2d, R, H, W, Re)
        with xi_channel_ref_swap(X_ref_2d, restore=False):   # reference mesh2d (gradient path)
            F_p_x, F_p_z = eval_forces_with_bg_on_tape(
                r_bif, 0.0, a_bif, u_bar_new, p_bar_new, G_val,
                md, r_ref, 0.0, a_ref,
                T_2d=T_eval, mesh2d=mesh2d, rz_on_tape=False,
                lift_on_tape=lift_on_tape)
    return float(F_p_x)


def _forward_pipeline_diff(T_eval, *, mesh2d, X_ref_2d, shared_data, md,
                           r_bif, a_bif, r_ref, a_ref):
    """Localize WHERE the two force evaluators diverge forward, at fixed T.

    Compares, at the same T_eval:
      • the channel deformation xi_channel built each way
          - gradient path:  build_xi_channel_from_T2d, restore=True (fresh)
          - eval path:      md_try['xi_channel'] from _prepare_md_try (restore=False)
      • the resulting F_r
          - gradient: eval_forces_with_bg_on_tape  (the AD twin, → -2.97 slope)
          - eval:     evaluate_forces              (the MS objective, → -4.69 slope)

    ‖Δxi_channel‖ ≈ 0 but ΔF_r large  ⇒  geometry identical, divergence is
    downstream (lifting branch / perturbed-flow / F_p). ‖Δxi_channel‖ large
    ⇒  the geometry build itself diverges (restore / fresh-vs-static).
    """
    # ── eval path (restore=False): md_try carries its xi_channel + BG ──
    md_try, _shared_try = _prepare_md_try_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data, md=md)
    xi_ch_eval = md_try['xi_channel']

    # ── gradient path (restore=True): fresh xi_channel ──
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_eval)
        with xi_channel_ref_swap(X_ref_2d):              # restore=True
            xi_ch_grad = build_xi_channel_from_T2d(
                T_eval, md['mesh3d'], md['X_ref'], mesh2d,
                md['R'], md['W'], md['H'])
        n_eval = float(norm(xi_ch_eval))
        n_grad = float(norm(xi_ch_grad))
        d = Function(xi_ch_eval.function_space())
        d.dat.data[:] = (np.asarray(xi_ch_grad.dat.data_ro)
                         - np.asarray(xi_ch_eval.dat.data_ro))
        n_diff = float(norm(d))

    Fr_grad = _eval_ontape_Fr_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
        md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref,
        lift_on_tape=False)                         # Branch 2 (off-tape VOM)
    Fr_grad_b1 = _eval_ontape_Fr_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
        md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref,
        lift_on_tape=True)                          # Branch 1 (differentiable_field_eval)
    Fr_eval, _Jzz = _eval_forces_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
        md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref)

    # The only differing input to perturbed_flow_differentiable is `a`:
    #   gradient:  float(a) = a_bif
    #   eval:      (md['a_init'], delta_a) → a_val = a_init + (a_bif - a_ref)
    # These match iff md['a_init'] == a_ref. Print to check.
    a_val_grad = float(a_bif)
    a_val_eval = float(md['a_init']) + float(a_bif - a_ref)

    print(f"  [fwd-diff] ‖xi_channel‖: grad(restore=T)={n_grad:.6e}  "
          f"eval(restore=F)={n_eval:.6e}  ‖Δ‖={n_diff:.3e}")
    print(f"  [fwd-diff] F_r: grad/branch2={Fr_grad:+.6e}  "
          f"grad/branch1={Fr_grad_b1:+.6e}  "
          f"eval(evaluate_forces)={Fr_eval:+.6e}")
    print(f"  [fwd-diff]   Δ(branch1-branch2)={Fr_grad_b1 - Fr_grad:+.3e}  "
          f"Δ(branch1-eval)={Fr_grad_b1 - Fr_eval:+.3e}  "
          f"Δ(branch2-eval)={Fr_grad - Fr_eval:+.3e}")
    print(f"  [fwd-diff] a_val: grad(float a)={a_val_grad:.8f}  "
          f"eval(a_init+δa)={a_val_eval:.8f}  Δ={a_val_grad - a_val_eval:+.3e}")
    print(f"  [fwd-diff]   md['a_init']={float(md['a_init']):.6f}  "
          f"a_ref={a_ref:.6f}  a_bif={a_bif:.6f}  "
          f"(grad uses a_bif; eval uses a_init+δa — equal iff a_init==a_ref)")


def _compare_F_p_components(T_eval, *, mesh2d, X_ref_2d, shared_data, md,
                            r_bif, a_bif, r_ref, a_ref):
    """Side-by-side F_p component breakdown for the two evaluators at fixed T.

    Geometry/lifting/a_val are all identical, yet F_r differs — so one of the
    physics terms (Stokes drag F_s, fluid_stress, inertial, centrifugal, or
    the rotation rate T_adj that feeds them) must diverge. This prints each
    component from both pipelines to pinpoint which.
    """
    R, H, W, L_c, U_c, Re = shared_data[:6]

    # ── gradient path: eval_forces_with_bg_on_tape ──
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_eval)
        G_val, _Um, u_bar_new, p_bar_new = _solve_bg_on_mesh(
            mesh2d, R, H, W, Re)
        with xi_channel_ref_swap(X_ref_2d):
            _Fx, _Fz, comps_grad = eval_forces_with_bg_on_tape(
                r_bif, 0.0, a_bif, u_bar_new, p_bar_new, G_val,
                md, r_ref, 0.0, a_ref,
                T_2d=T_eval, mesh2d=mesh2d, rz_on_tape=False,
                return_components=True)

    # ── eval path: evaluate_forces ──
    # _prepare_md_try uses restore=False → leaves mesh2d at REFERENCE. Evaluate
    # once as-is (mesh2d@ref, what MS sees) and once with mesh2d reset to the
    # DEFORMED state (matching the gradient). differentiable_field_eval samples
    # u_bar_2d via a VOM on mesh2d, so its coord state changes the sampling.
    md_try, _shared_try = _prepare_md_try_at_T(
        T_eval, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data, md=md)
    dr = float(r_bif - r_ref)
    da = float(a_bif - a_ref)
    with stop_annotating():
        _F1, comps_eval_ref = evaluate_forces(
            dr, 0.0, da, md_try, return_components=True)       # mesh2d @ reference
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_eval)            # → deformed
        _F2, comps_eval_def = evaluate_forces(
            dr, 0.0, da, md_try, return_components=True)

    def _Fr(c):  # F_p_x = (1/Re)·F_s_x + fluid_stress_x + inertial_x + centrifugal_x
        return (c['F_s_x'] / float(Re) + c['fluid_stress_x']
                + c['inertial_x'] + c['centrifugal_x'])
    print(f"  [comp] F_r: grad={_Fr(comps_grad):+.6e}  "
          f"eval@ref={_Fr(comps_eval_ref):+.6e}  "
          f"eval@deformed={_Fr(comps_eval_def):+.6e}")
    print(f"  [comp]   Δ(grad-eval@ref)={_Fr(comps_grad)-_Fr(comps_eval_ref):+.3e}  "
          f"Δ(grad-eval@deformed)={_Fr(comps_grad)-_Fr(comps_eval_def):+.3e}")

    print(f"  [comp] {'component':24s} {'grad':>15s} {'eval@ref':>15s} "
          f"{'eval@def':>15s}")
    for k in sorted(comps_grad):
        g = comps_grad[k]
        er = comps_eval_ref.get(k, float('nan'))
        ed = comps_eval_def.get(k, float('nan'))
        print(f"  [comp] {k:24s} {g:>+15.6e} {er:>+15.6e} {ed:>+15.6e}")


def _smooth_symmetric_direction(V_2d, mesh2d, X_ref_2d, W, H, *,
                                seed=0, smooth=0.05, target_max=1.0):
    """Build a smooth, z-symmetric, corner-pinned, boundary-moving Taylor
    direction dT.

    A meaningful test of a *shape* gradient needs dT to move the domain
    boundary (Hadamard: ∂a/∂T is supported on ∂Ω, so an interior-only bump
    would pair against pure FE noise). We start from a random nodal field,
    Helmholtz-smooth it (so finite-ε deformed meshes stay valid), pin the
    four corners (consistent with the Riesz metric's kernel removal in
    ``riesz_representative``), then project onto the z-symmetric subspace
    the 2×2 reduction lives on. Finally normalise to ‖dT‖_∞ = target_max.
    """
    rng = np.random.default_rng(seed)
    v = TrialFunction(V_2d)
    w = TestFunction(V_2d)
    a_form = (smooth * inner(grad(v), grad(w)) + inner(v, w)) * dx

    rand = Function(V_2d)
    with stop_annotating():
        rand.dat.data[:] = rng.standard_normal(
            np.asarray(rand.dat.data_ro).shape)
    L_form = inner(rand, w) * dx

    out = Function(V_2d, name="taylor_dir")
    with stop_annotating():
        X_arr = np.asarray(X_ref_2d.dat.data_ro)
        atol  = 1e-6
        is_corner = (
            (np.isclose(X_arr[:, 0], 0.0, atol=atol)
             | np.isclose(X_arr[:, 0], W, atol=atol))
            & (np.isclose(X_arr[:, 1], 0.0, atol=atol)
               | np.isclose(X_arr[:, 1], H, atol=atol))
        )
        corner_nodes = np.where(is_corner)[0]

        A = assemble(a_form)
        b = assemble(L_form)
        n_comp = V_2d.value_size
        corner_dofs = np.concatenate([
            n_comp * corner_nodes + c for c in range(n_comp)
        ]).astype(np.int32)
        A.M.handle.zeroRows(corner_dofs, diag=1.0)
        b.dat.data[corner_nodes, :] = 0.0
        LinearSolver(A, solver_parameters={
            "ksp_type": "cg", "pc_type": "hypre",
            "ksp_rtol": 1e-10, "ksp_atol": 1e-14,
        }).solve(out, b)

    # Project onto the z-symmetric subspace (keeps the test inside the
    # reduction's valid manifold; the antisymmetric part it removes is small).
    out, max_anti, _mdist = project_z_symmetric(out, mesh2d, H)

    with stop_annotating():
        arr = np.asarray(out.dat.data_ro)
        m   = float(np.max(np.linalg.norm(arr, axis=1)))
        if m > 0:
            out.dat.data[:] = arr * (target_max / m)
    print(f"  [Taylor-dir] smooth random z-sym direction: "
          f"‖dT‖_∞ = {target_max:.3g}, antisym removed = {max_anti:.2e}")
    return out


def taylor_test_shape_gradient_symmetric(grad_a, g1_term, g2_term, lambda_adj,
                                         T_2d, mesh2d, X_ref_2d, V_2d,
                                         shared_data, md,
                                         r_bif, a_bif, r_ref, a_ref, W, H,
                                         *,
                                         ms_tol=1e-12, ms_max_iter=30,
                                         eps0=4e-3, n_eps=4, factor=0.5,
                                         dir_seed=0, dir_smooth=0.05,
                                         dir_max=1.0, perturbation=None):
    """Finite-difference Taylor test for ∇a = ∂a_bif/∂T_2d.

    Called from inside ``run_shape_optimization_symmetric`` at step 1 using
    the LIVE objects (``grad_a``, its G1/G2 split, ``T_2d``, ``mesh2d``,
    ``md`` …) the optimiser already computed — no duplicate setup, no second
    gradient solve. Verifies the shape gradient against re-evaluations of
    a_bif(T) through the SAME BG-flow + symmetric Moore-Spence pipeline the
    line-search uses (``_evaluate_a_bif_at_T``). For a direction dT,

        a(T + ε·dT) = a(T) + ε·⟨∇a, dT⟩ + O(ε²).

    Reported per ε:
      • r0 = |a(T+ε dT) − a(T)|                 → must shrink O(ε)   (rate≈1)
      • r1 = |a(T+ε dT) − a(T) − ε·⟨∇a, dT⟩|    → must shrink O(ε²)  (rate≈2)

    The directional derivative uses the l2 dof pairing ⟨∇a, dT⟩ =
    Σ_i (∇a)_i (dT)_i — the correct contraction for the cotangent (Cofunction)
    that ``.derivative()`` returns. This is the RAW gradient, NOT its Riesz
    lift V_a. ``g1_term``/``g2_term`` are the linear (λ_0·F_x) and Hessian-VP
    (λ_1·F_z) pieces, paired with dT separately to localise any error.
    """
    print(f"\n{'#'*70}")
    print(f"#  TAYLOR TEST  ∇a = ∂a_bif/∂T_2d  (z-SYMMETRIC, inline @ step 1)")
    print(f"#  base bif: r = {r_bif:.6f}, a = {a_bif:.6f}  (z=0, phi=(0,1))")
    print(f"#  eps0 = {eps0:.2e}, factor = {factor}, n_eps = {n_eps}")
    print(f"{'#'*70}")

    # ── Perturbation direction (reset coords to reference first) ──
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d)
    if perturbation is None:
        dT = _smooth_symmetric_direction(
            V_2d, mesh2d, X_ref_2d, W, H,
            seed=dir_seed, smooth=dir_smooth, target_max=dir_max)
    else:
        dT = perturbation

    # ── Directional derivative under BOTH duality conventions ──
    # pyadjoint's .derivative() may return either the RAW cotangent
    # (riesz_representation=None → a Cofunction; pair via l2 dof dot) or the
    # L2-Riesz gradient (a primal Function; pair via the L2 inner product
    # ∫∇a·dT dx). We report both so a rerun pins down which one gives the
    # rate-2 column; the other is then the wrong contraction, not a wrong ∇a.
    is_dual = grad_a.function_space().dual() != grad_a.function_space()
    print(f"  [Taylor] type(grad_a) = {type(grad_a).__name__}  "
          f"(dual/cofunction space = {is_dual})")

    dadT_l2 = float(np.sum(np.asarray(grad_a.dat.data_ro)
                           * np.asarray(dT.dat.data_ro)))
    with stop_annotating():
        g_prim = Function(V_2d, name="grad_a_primal")
        g_prim.dat.data[:] = np.asarray(grad_a.dat.data_ro)
        dadT_L2 = float(assemble(inner(g_prim, dT) * dx))
    print(f"  [Taylor] ⟨∇a, dT⟩_l2  (dof dot)      = {dadT_l2:+.6e}")
    print(f"  [Taylor] ⟨∇a, dT⟩_L2  (∫∇a·dT dx)    = {dadT_L2:+.6e}")

    # ── G1 (linear λ_0·F_x) vs G2 (Hessian-VP λ_1·F_z) decomposition ──
    # l2 dof pairing (the correct one for a cofunction); their sum = dadT_l2.
    dadT_g1 = float(np.sum(np.asarray(g1_term.dat.data_ro)
                           * np.asarray(dT.dat.data_ro)))
    dadT_g2 = float(np.sum(np.asarray(g2_term.dat.data_ro)
                           * np.asarray(dT.dat.data_ro)))
    print(f"  [Taylor] ⟨G1, dT⟩_l2  (linear  λ_0·F_x)  = {dadT_g1:+.6e}")
    print(f"  [Taylor] ⟨G2, dT⟩_l2  (Hessian λ_1·F_z)  = {dadT_g2:+.6e}")
    print(f"  [Taylor]   sum = {dadT_g1 + dadT_g2:+.6e}  (= dadT_l2 check)")

    # ── Building-block FD test: verify ∂F_r/∂T and ∂J_zz/∂T directly ──
    # Strip the IFT multiplier to expose the raw AD shape-derivatives:
    #   AD ⟨∂F_r/∂T,  dT⟩ = ⟨G1,dT⟩ / λ_0     (linear reverse-mode)
    #   AD ⟨∂J_zz/∂T, dT⟩ = ⟨G2,dT⟩ / λ_1     (mixed Hessian-VP)
    # then finite-difference F_r(T) and J_zz(T) at FROZEN (r_bif, 0, a_bif).
    # A clean rate-1 match means the building block is correct → any error
    # in da/dT is then pure IFT algebra (λ sign / DG). A mismatch (esp. in
    # ∂J_zz/∂T) localises the bug to that AD derivative.
    lam0 = float(lambda_adj[0])
    lam1 = float(lambda_adj[1])
    ad_dFr  = dadT_g1 / lam0 if abs(lam0) > 0 else float('nan')
    ad_dJzz = dadT_g2 / lam1 if abs(lam1) > 0 else float('nan')
    print(f"  [Taylor] λ = [{lam0:+.4e}, {lam1:+.4e}]")
    print(f"  [Taylor] AD ⟨∂F_r/∂T,  dT⟩ = {ad_dFr:+.6e}   (= G1/λ_0)")
    print(f"  [Taylor] AD ⟨∂J_zz/∂T, dT⟩ = {ad_dJzz:+.6e}   (= G2/λ_1)")

    with stop_annotating():
        Fr0, Jzz0 = _eval_forces_at_T(
            T_2d, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
            md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref)
    print(f"  [Taylor] base forces: F_r = {Fr0:+.6e}, J_zz = {Jzz0:+.6e}")

    print(f"\n  ── building-block FD (frozen bif state) ──")
    print(f"  {'eps':>10} {'dFr/eps (FD)':>15} {'AD ∂F_r/∂T':>13} "
          f"{'dJzz/eps (FD)':>15} {'AD ∂J_zz/∂T':>13}")
    bb_eps = float(eps0)
    for _k in range(max(2, n_eps - 1)):
        with stop_annotating():
            T_bb = T_2d.copy(deepcopy=True)
            T_bb.dat.data[:] += bb_eps * np.asarray(dT.dat.data_ro)
            Fr_e, Jzz_e = _eval_forces_at_T(
                T_bb, mesh2d=mesh2d, X_ref_2d=X_ref_2d,
                shared_data=shared_data, md=md,
                r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref)
        fd_dFr  = (Fr_e  - Fr0)  / bb_eps
        fd_dJzz = (Jzz_e - Jzz0) / bb_eps
        print(f"  {bb_eps:>10.3e} {fd_dFr:>15.6e} {ad_dFr:>13.6e} "
              f"{fd_dJzz:>15.6e} {ad_dJzz:>13.6e}")
        bb_eps *= factor
    print(f"  (off-tape FD column → AD column as eps→0 ⇒ that building block "
          f"is correct; persistent mismatch localises the bug)")

    # ── ON-TAPE FD: ∂F_r/∂T on the exact map the gradient differentiates ──
    # off-tape (above) uses evaluate_forces / restore=False — the map MS &
    # the line search evaluate. on-tape uses eval_forces_with_bg_on_tape /
    # restore=True — the map the AD gradient differentiates. Comparing both
    # FD columns against the single AD value decides between:
    #   on-tape-FD ≈ AD, ≠ off-tape-FD  → AD correct for its map; bug = the
    #                                      on↔off map inconsistency (+ sign).
    #   on-tape-FD ≠ AD                  → the on-tape AD itself is wrong.
    with stop_annotating():
        Fr0_ot = _eval_ontape_Fr_at_T(
            T_2d, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
            md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref)
    print(f"\n  ── ∂F_r/∂T : on-tape FD vs off-tape FD vs AD ──")
    print(f"  {'eps':>10} {'on-tape FD':>15} {'off-tape FD':>15} "
          f"{'AD (G1/λ_0)':>15}")
    ot_eps = float(eps0)
    for _k in range(max(2, n_eps - 1)):
        with stop_annotating():
            T_ot = T_2d.copy(deepcopy=True)
            T_ot.dat.data[:] += ot_eps * np.asarray(dT.dat.data_ro)
            Fr_e_ot = _eval_ontape_Fr_at_T(
                T_ot, mesh2d=mesh2d, X_ref_2d=X_ref_2d,
                shared_data=shared_data, md=md,
                r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref)
            T_off = T_2d.copy(deepcopy=True)
            T_off.dat.data[:] += ot_eps * np.asarray(dT.dat.data_ro)
            Fr_e_off, _ = _eval_forces_at_T(
                T_off, mesh2d=mesh2d, X_ref_2d=X_ref_2d,
                shared_data=shared_data, md=md,
                r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref)
        fd_ot  = (Fr_e_ot  - Fr0_ot) / ot_eps
        fd_off = (Fr_e_off - Fr0)    / ot_eps
        print(f"  {ot_eps:>10.3e} {fd_ot:>15.6e} {fd_off:>15.6e} "
              f"{ad_dFr:>15.6e}")
        ot_eps *= factor
    print(f"  (on-tape FD ≈ AD & ≠ off-tape FD ⇒ map inconsistency is the "
          f"bug; on-tape FD ≠ AD ⇒ the on-tape AD itself is wrong)")

    # (Debug diagnostics _forward_pipeline_diff / _compare_F_p_components
    #  removed from the live run — they did explicit mesh2d.assign(deformed)
    #  that is inconsistent with the channel-decoupled (reference-mesh2d) lift.
    #  The building-block FD above + the main table below are the numerical
    #  tests; the helpers remain defined for ad-hoc use.)

    # ── 3. Baseline a_bif(T=0) through the evaluation map ──
    a_base, r_base, conv_base, F_base = _evaluate_a_bif_at_T(
        T_2d, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
        md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref,
        ms_tol=ms_tol, ms_max_iter=ms_max_iter)
    print(f"  [Taylor] a(T=0) = {a_base:.10f}  "
          f"(conv={conv_base}, |F_r|={F_base:.2e})")

    # ── 4. ε sweep ──
    rows = []
    eps = float(eps0)
    for k in range(n_eps):
        with stop_annotating():
            T_pert = T_2d.copy(deepcopy=True)
            T_pert.dat.data[:] += eps * np.asarray(dT.dat.data_ro)
        a_eps, r_eps, conv_eps, F_eps = _evaluate_a_bif_at_T(
            T_pert, mesh2d=mesh2d, X_ref_2d=X_ref_2d, shared_data=shared_data,
            md=md, r_bif=r_bif, a_bif=a_bif, r_ref=r_ref, a_ref=a_ref,
            ms_tol=ms_tol, ms_max_iter=ms_max_iter)
        r0    = abs(a_eps - a_base)
        r1_l2 = abs(a_eps - a_base - eps * dadT_l2)
        r1_L2 = abs(a_eps - a_base - eps * dadT_L2)
        slope = (a_eps - a_base) / eps
        rows.append({'eps': eps, 'a': a_eps, 'conv': conv_eps, 'F': F_eps,
                     'r0': r0, 'r1_l2': r1_l2, 'r1_L2': r1_L2, 'slope': slope})
        print(f"  [Taylor] eps={eps:.3e}  a={a_eps:.10f}  conv={conv_eps}  "
              f"slope={slope:+.4f}  r0={r0:.3e}  "
              f"r1_l2={r1_l2:.3e}  r1_L2={r1_L2:.3e}")
        eps *= factor

    # ── 5. Convergence-rate table ──
    def _rate(prev, cur, key):
        if prev is None or cur[key] <= 0 or prev[key] <= 0:
            return float('nan')
        return np.log(prev[key] / cur[key]) / np.log(prev['eps'] / cur['eps'])

    print(f"\n{'─'*86}")
    print(f"  {'eps':>11} {'a(T+eps dT)':>15} {'slope':>9} {'cv':>3} "
          f"{'r0':>10} {'r1_l2':>10} {'rt_l2':>6} {'r1_L2':>10} {'rt_L2':>6}")
    print(f"  {'─'*84}")
    prev = None
    rates_l2, rates_L2 = [], []
    for row in rows:
        rt0    = _rate(prev, row, 'r0')
        rt_l2  = _rate(prev, row, 'r1_l2')
        rt_L2  = _rate(prev, row, 'r1_L2')
        if np.isfinite(rt_l2):
            rates_l2.append(rt_l2)
        if np.isfinite(rt_L2):
            rates_L2.append(rt_L2)
        print(f"  {row['eps']:>11.3e} {row['a']:>15.10f} {row['slope']:>+9.4f} "
              f"{'Y' if row['conv'] else 'N':>3} "
              f"{row['r0']:>10.3e} {row['r1_l2']:>10.3e} {rt_l2:>6.3f} "
              f"{row['r1_L2']:>10.3e} {rt_L2:>6.3f}")
        prev = row
    print(f"  {'─'*84}")

    best_l2 = max(rates_l2) if rates_l2 else float('nan')
    best_L2 = max(rates_L2) if rates_L2 else float('nan')
    obs_slope = rows[-1]['slope']          # smallest-ε slope ≈ true ⟨∇a, dT⟩
    print(f"  observed ⟨∇a, dT⟩ (smallest-ε slope) = {obs_slope:+.6e}")
    print(f"  AD ⟨∇a, dT⟩_l2 = {dadT_l2:+.6e}   (best rate {best_l2:.3f})")
    print(f"  AD ⟨∇a, dT⟩_L2 = {dadT_L2:+.6e}   (best rate {best_L2:.3f})")
    best_rate1 = max(best_l2, best_L2)
    verdict = "PASS" if np.isfinite(best_rate1) and best_rate1 > 1.8 else "CHECK"
    print(f"  → {verdict}   (rate→2 in EITHER column confirms ∇a under that "
          f"pairing; rate→1 in BOTH means ∇a itself is wrong)")
    print(f"{'#'*70}\n")

    return {
        'dadT_l2':   dadT_l2,
        'dadT_L2':   dadT_L2,
        'obs_slope': obs_slope,
        'a_base':    a_base,
        'rows':      rows,
        'best_rate_l2': best_l2,
        'best_rate_L2': best_L2,
        'grad_a':    grad_a,
        'dT':        dT,
    }


def diagnose_basis_staleness(md_try, shared_try, r_try, a_try, r_ref, a_ref,
                             *, ms_tol, ms_max_iter, stalled_F, stalled_M):
    """One-shot probe: is a stalled inner-MS trial caused by the FROZEN
    (last-accepted-centred) ALE basis, or by pitchfork-flattening / F_z mesh
    noise?

    During a line-search trial the inner ``moore_spence_solve_symmetric``
    uses the ALE basis ``(basis_r, basis_z, basis_a)`` linearised at the LAST
    ACCEPTED bifurcation (r_ref, a_ref). For a large-alpha trial the bif sits
    far from that centre (Δa can reach ~0.035), so the linear superposition
    ``xi_particle = dr·b_r + da·b_a`` may be too inaccurate for the
    second-derivative ``J_zz = ∂F_z/∂z`` — the MS Newton then floors |M| above
    tol. This probe RE-CENTRES the basis at the trial's own (r_try, a_try) via
    ``reset_ale_basis_for_step`` and re-runs MS from delta=0:

      • |M| collapses to ~machine-eps  ⇒  the stall WAS within-trial ALE-basis
        staleness (cause 1) → fix: re-linearise the basis mid-solve once the
        drift exceeds a threshold.
      • |M| stays at the stalled floor ⇒  the basis is innocent; the floor is
        set by the flattening pitchfork (J_rr → 0) and/or the F_z mesh-noise
        of the non-mirror-symmetric 3D mesh (cause 2/3).

    Fully NON-DESTRUCTIVE: snapshots and restores every md field that
    ``reset_ale_basis_for_step`` mutates (xi_baseline, a_init, basis_*_data,
    mesh3d coords). ``md_try`` shares these objects with the live ``md`` (it is
    a shallow ``dict(md)``), so the restore is what keeps the optimiser intact.
    """
    md     = md_try
    mesh3d = md['mesh3d']

    with stop_annotating():
        save_xi    = np.asarray(md['xi_baseline'].dat.data_ro).copy()
        save_ainit = float(md['a_init'])
        save_br    = np.asarray(md['basis_r_data']).copy()
        save_bz    = np.asarray(md['basis_z_data']).copy()
        save_ba    = np.asarray(md['basis_a_data']).copy()
        save_xyz   = mesh3d.coordinates.copy(deepcopy=True)

    dr_drift = float(r_try - r_ref)
    da_drift = float(a_try - a_ref)

    print(f"\n  ┌─ [BasisDiag] ALE-basis staleness probe ──────────────────")
    print(f"  │ stalled MS (basis @ a_ref={a_ref:.6f}): "
          f"|M|={stalled_M:.3e}  |F_r|={stalled_F:.3e}  a_try={a_try:.6f}")
    print(f"  │ re-centring basis at trial bif (Δr={dr_drift:+.4f}, "
          f"Δa={da_drift:+.4f}) and re-running MS from delta=0 …")
    try:
        reset_ale_basis_for_step(md, dr_drift, 0.0, da_drift)
        with stop_annotating():
            r2, a2, conv2, F2, M2, stalled2 = moore_spence_solve_symmetric(
                r_try, a_try, shared_try,
                tol=ms_tol, max_iter=ms_max_iter, md=md,
                dr_init=0.0, da_init=0.0)
        print(f"  │ re-centred MS: conv={conv2}  |M|={M2:.3e}  "
              f"|F_r|={F2:.3e}  a={a2:.6f}  (stalled={stalled2})")
        improved = (M2 < 1e-3 * stalled_M) or (conv2 and M2 < 1e-10)
        if improved:
            print(f"  │ VERDICT: |M| dropped {stalled_M:.2e} → {M2:.2e}  "
                  f"⇒ stall WAS within-trial ALE-basis staleness (cause 1).")
            print(f"  │ FIX: re-linearise the basis when the inner-MS drift "
                  f"exceeds a threshold (call reset_ale_basis_for_step mid-solve).")
        else:
            print(f"  │ VERDICT: |M| ~unchanged ({stalled_M:.2e} → {M2:.2e})  "
                  f"⇒ basis is innocent; floor is pitchfork-flattening (J_rr→0) "
                  f"/ F_z mesh noise (cause 2/3).")
    except Exception as e:
        print(f"  │ [BasisDiag] probe failed: {e}")
    finally:
        with stop_annotating():
            md['xi_baseline'].dat.data[:] = save_xi
            md['a_init']       = save_ainit
            md['basis_r_data'] = save_br
            md['basis_z_data'] = save_bz
            md['basis_a_data'] = save_ba
            mesh3d.coordinates.assign(save_xyz)
    print(f"  └───────────────────────────────────────────────────────────")


def run_shape_optimization_symmetric(a_target, shared_data, mesh_data_init,
                                      bif_result_init,
                                      *,
                                      r_ref, a_ref,
                                      max_steps=50, tol_J=1e-8,
                                      alpha_step=0.1, alpha_min=1e-8,
                                      alpha_backtrack=0.5, max_backtrack=12,
                                      step_max=1.0, branch_C=1.0,
                                      riesz_alpha=1.0, riesz_beta=1e-2,
                                      riesz_metric="h1",
                                      riesz_mu=None, riesz_lambda=None,
                                      ms_tol=1e-12, ms_max_iter=30,
                                      n_grid_2d=128,
                                      mesh_quality_floor=0.20,
                                      mesh_quality_warn=0.35,
                                      plot_dir=None,
                                      taylor_check=False, taylor_only=True,
                                      taylor_kw=None,
                                      diagnose_basis=False):
    """Algorithm 4.1, z-symmetric branch.

    ``mesh_quality_floor`` — hard gate on the deformed mesh2d shape quality
    (min over cells of q = 4√3·A/Σℓ² ∈ (0,1]). A line-search trial whose deformed
    mesh2d drops below this is rejected like a sign-flip, so the line search
    shrinks step_len and keeps the Riesz elastic mesh-move well-conditioned.
    ``mesh_quality_warn`` — softer level; only logged (per accepted step) as
    an early-warning that the cross-section mesh is starting to distort.

    ``riesz_metric`` — "h1" (default, vector-Laplacian) or "elasticity"
    (linear elasticity, couples components via ε(V)/div V → better-
    conditioned mesh moves). ``riesz_mu``/``riesz_lambda`` are the Lamé
    parameters for the elasticity metric (default to ``riesz_alpha`` each).
    The chosen metric is used consistently for the Riesz solve, the explicit
    ‖V_a‖²_M, and the [Diag] pairings via the shared ``riesz_metric_form``.

    Identical control flow to ``run_shape_optimization`` but with:
      - state (r_bif, a_bif); z_bif ≡ 0, phi_bif ≡ (0, 1) fixed by symmetry,
      - 2×2 DG and 2-vector adjoint multiplier λ,
      - inner MS via the 2×2 symmetric solver,
      - z-symmetric projection of V_rep at every step (MANDATORY here —
        the reduction's premise is that T_2d stays mirror-symmetric).
    """
    R, H, W, L_c, U_c, Re, _, _, _, _ = shared_data

    if plot_dir is None:
        plot_dir = ("images/shape_opt_sym_run_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S"))
    print(f"  [ShapeOpt-Sym] Cross-section snapshots → {plot_dir}/")

    mesh2d = RectangleMesh(n_grid_2d, n_grid_2d, W, H,
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

    q0_min, q0_mean = mesh2d_cell_quality(mesh2d)
    print(f"  [MeshQ] reference cross-section ({n_grid_2d}×{n_grid_2d}, crossed): "
          f"q_min = {q0_min:.4f}  q_mean = {q0_mean:.4f}  "
          f"(quality floor for line search = {mesh_quality_floor:.2f})")

    r_bif = float(bif_result_init['r'])
    a_bif = float(bif_result_init['a'])

    md         = mesh_data_init
    shared_cur = shared_data

    if riesz_metric == "elasticity":
        _mu  = riesz_alpha if riesz_mu     is None else riesz_mu
        _lam = riesz_alpha if riesz_lambda is None else riesz_lambda
        _metric_str = f"elasticity (μ={_mu:.3g}, λ={_lam:.3g}, β={riesz_beta:.3g})"
    else:
        _metric_str = f"H¹ vector-Laplacian (α={riesz_alpha:.3g}, β={riesz_beta:.3g})"

    print(f"\n{'#'*70}")
    print(f"#  SHAPE OPTIMISATION  (Algorithm 4.1, z-SYMMETRIC)")
    print(f"#  Target:  a* = {a_target:.4f}")
    print(f"#  Initial: a* = {a_bif:.6f}  (z = 0 fixed, phi = (0, 1))")
    print(f"#  Riesz metric: {_metric_str}")
    print(f"{'#'*70}")

    history   = []
    converged = False
    # One-shot guard for the ALE-basis staleness probe (diagnose_basis=True):
    # fire on the FIRST non-converged inner-MS trial, then disable.
    _basis_diag_done = False

    rho_accept = 0.1
    # rho_grow lowered from 0.75: in log_2 the first accepted steps had
    # ρ ≈ 0.49, sitting in the "no growth" band (between rho_accept and
    # rho_grow) and clamping step_len forever at alpha_step. With rho_grow=0.4
    # the line search auto-promotes step_len on these steady-good steps, and the
    # rho_accept safety net still catches over-aggressive trials. Pure
    # control-flow change, no compute pipeline modification.
    rho_grow   = 0.4
    step_init = min(float(alpha_step), float(step_max))

    # No persistent curvature state. The line search uses pred = α·2ε² as the
    # linear-a model prediction and trusts the post-MS rho-check
    # (together with mesh_ok and branch tracking) to catch over-
    # aggressive alpha. H_obs is computed per trial for diagnostic
    # printing only — not fed back into any decision logic.

    # Diagnostic state: V_a snapshot from the previous outer step. Used
    # to test whether V_a is the dominant eigenmode of M⁻¹·∇²a (in which
    # case L-BFGS would only add a magnitude correction, not a direction
    # correction, and the per-step Δa ceiling would persist).
    # Tracked quantities at each step k ≥ 2:
    #   • cos(V_a_{k-1}, V_a_k)_M           — direction-lock indicator
    #   • ||η||_M / ||V_a_{k-1}||_M          — magnitude of the V_a-change
    #     where η := V_a_{k-1} − V_a_k
    #   • cos²(V_a_{k-1}, η)_M               — Rayleigh tightness; equals 1
    #     iff η ∝ V_a_{k-1}, i.e. M⁻¹·∇²a·V_a_{k-1} is pure scaling on
    #     V_a_{k-1} (eigenvector condition). <1 ⇒ V_a rotates between
    #     steps ⇒ L-BFGS could find directions out of the V_a-plane.
    prev_V_a      = None
    prev_g_a_sq_M = None

    # Lag-by-one curvature estimate for the line-search quadratic model. After each
    # accepted step the Curv-Diag block reconstructs
    #   Q := ⟨V_a, ∇²a·V_a⟩_L²
    # from (ε_old, ε_new, α). This Q is then used as the curvature in next
    # iteration's pred. Snap-and-reset keeps the local a-curvature roughly
    # stable between outer steps so a one-step lag is a reasonable estimator.
    # Step 1 (no history) falls back to the linear-a model.
    Q_eff_prev = None

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
                                    W, H, a_bif, J, plot_dir)

        if J < tol_J:
            print(f"\n  CONVERGED: J = {J:.4e} < tol = {tol_J:.4e}")
            converged = True
            break

        # evaluate_forces (called inside compute_DG) builds its VOM with query
        # points CLIPPED to the reference cross-section box [0,W]×[0,H]
        # (differentiable_field_eval). With the decoupled lift the channel
        # deformation lives in md['xi_channel']/u_bar_2d, NOT in mesh2d, so
        # mesh2d must sit at the pure reference here — otherwise the reference-
        # box points fall outside the deformed mesh2d and VertexOnlyMesh raises
        # MissingPointsError. The previous step's tail left mesh2d at the
        # deformed X_ref_2d + T_2d (for plotting/diagnostics); reset it. This
        # mirrors the reference state MS used during the line search.
        with stop_annotating():
            mesh2d.coordinates.assign(X_ref_2d)

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

        _want_terms = taylor_check and step == 0
        _grad_out = compute_shape_gradient_symmetric(
            r_bif, a_bif, a_target, DG_2x2,
            T_2d, mesh2d, X_ref_2d, R, H, W, Re, md,
            r_ref, a_ref, return_terms=_want_terms)
        if _want_terms:
            grad_a, lambda_adj, g1_term, g2_term = _grad_out
        else:
            grad_a, lambda_adj = _grad_out

        # ── Inline Taylor test of ∇a (only at step 1, if requested) ──
        # Reuses the live grad_a / G1 / G2 / mesh objects just computed — no
        # duplicate setup. Validates ∂a_bif/∂T against finite differences of
        # the same BG+MS pipeline the line-search uses.
        if _want_terms:
            taylor_test_shape_gradient_symmetric(
                grad_a, g1_term, g2_term, lambda_adj,
                T_2d, mesh2d, X_ref_2d, V_2d,
                shared_cur, md, r_bif, a_bif, r_ref, a_ref, W, H,
                ms_tol=ms_tol, ms_max_iter=ms_max_iter,
                **(taylor_kw or {}))
            if taylor_only:
                print("  [Taylor] taylor_only=True — stopping after the "
                      "gradient check (no optimisation step taken).")
                return {
                    'T_2d': T_2d, 'mesh2d': mesh2d, 'X_ref_2d': X_ref_2d,
                    'a_bif': a_bif, 'r_bif': r_bif, 'z_bif': 0.0,
                    'phi_bif': np.array([0.0, 1.0]),
                    'J_final': (a_bif - a_target) ** 2,
                    'history': history, 'converged': False,
                    'shared_data_final': shared_cur, 'mesh_data_final': md,
                    'taylor_only': True,
                }

        # ── Riesz lift of ∇a (metric = riesz_metric) ──
        # Same solver as before, but the RHS is now the ε-free ∇a, so
        # V_a has fixed magnitude across outer steps and the Gauss-Newton
        # step picks up ε only via the explicit (a_bif − a_target) factor
        # in the step-direction construction below. The metric (H¹ or
        # elasticity) is selected once here and reused for ‖V_a‖²_M and the
        # [Diag] pairings, all via the shared riesz_metric_form.
        V_a = riesz_representative(
            grad_a, mesh2d,
            alpha_elast=riesz_alpha, beta_l2=riesz_beta,
            fix_corners=True, W=W, H=H,
            metric=riesz_metric, mu=riesz_mu, lam=riesz_lambda)

        # MANDATORY z-symmetric projection. The shape gradient is in the
        # symmetric subspace by physics (F_r symmetric, J_zz symmetric in T),
        # so this only kills FE-noise antisymmetric components. Without it,
        # FE noise accumulates over outer steps and eventually breaks the
        # F_z ≡ 0 / J_off-diag = 0 premise of the 2×2 reduction.
        V_a, max_anti, mirror_dist = project_z_symmetric(
            V_a, mesh2d, H)
        print(f"  [Symm] z-symmetric projection: "
              f"max antisym killed = {max_anti:.3e}  "
              f"max mirror-lookup dist = {mirror_dist:.2e}")

        with stop_annotating():
            g_a_sq_L2 = float(assemble(inner(V_a, V_a) * dx))
            # MUST match the riesz_representative solve metric exactly — the
            # line-search step landing (gn_scale = −ε/g_a_sq_M) and pred slope
            # (grad_norm_sq_M = 2ε²) rely on g_a_sq_M = ⟨∇a, V_a⟩, which holds
            # only when this form == the solve form. Shared via
            # riesz_metric_form so a metric switch can't break the identity.
            g_a_sq_M = float(assemble(riesz_metric_form(
                V_a, V_a, metric=riesz_metric,
                alpha=riesz_alpha, beta=riesz_beta,
                mu=riesz_mu, lam=riesz_lambda)))
        print(f"  ||V_a||_L2 = {g_a_sq_L2 ** 0.5:.4e}   "
              f"||V_a||_M = {g_a_sq_M ** 0.5:.4e}  ({riesz_metric} metric)")

        # ── DIAGNOSTIC: true directional derivative vs g_a_sq_M ──────────────
        # The step scaling gn_scale = −ε/g_a_sq_M assumes the identity
        #     g_a_sq_M = M(V_a, V_a) = ⟨∇a, V_a⟩.
        # That holds ONLY without interior masking (then V_a = M⁻¹∇a, SPD).
        # With mask_interior=True the Riesz RHS is the boundary-only part
        # P_∂Ω·∇a, so V_a = M⁻¹P_∂Ω∇a and the TRUE directional derivative
        #     d_true = ⟨∇a, V_a⟩ = ⟨∇a, M⁻¹P_∂Ω∇a⟩
        # is sign-INDEFINITE (M⁻¹P is not SPD). The actual initial slope of
        # a along the GN step is  da/dα|₀ = gn_scale·d_true = −ε·(d_true/g_a_sq_M),
        # so:
        #   • d_true ≤ 0  ⇒  V_a is NOT a descent direction (step raises a),
        #   • d_true ≠ g_a_sq_M  ⇒  step is mis-scaled (landing ≠ a_target).
        # d_true uses the FULL, UNMASKED grad_a · V_a (l2 dof dot). Purely
        # diagnostic — nothing below consumes these values.
        with stop_annotating():
            d_true = float(np.sum(np.asarray(grad_a.dat.data_ro)
                                  * np.asarray(V_a.dat.data_ro)))
        _eps_diag   = float(a_bif - a_target)
        _ratio_diag = d_true / g_a_sq_M if abs(g_a_sq_M) > 1e-30 else float('nan')
        # Post-Fix-1 the step is scaled by d_true, so da/dα|₀ = −ε ALWAYS
        # (guaranteed descent). What this diagnostic now reports is the
        # "natural sense" of the masked lift: d_true>0 → V_a itself descends
        # (step in −V_a); d_true<0 → V_a ascends, Fix-1 flips the step to +V_a.
        # ratio = d_true/g_a_sq_M ≠ 1 quantifies how badly interior masking
        # broke the would-be g_a_sq_M scaling (ratio<0 = sign flip).
        _sense = "V_a descends (step −V_a)" if d_true > 0 \
            else "V_a ascends → Fix-1 flips step to +V_a"
        print(f"  [DiagDD] d_true=⟨∇a,V_a⟩ = {d_true:+.6e}   "
              f"g_a_sq_M = {g_a_sq_M:+.6e}   ratio = {_ratio_diag:+.4f}")
        print(f"  [DiagDD] step scaled by d_true → da/dα|₀ = −ε = "
              f"{-_eps_diag:+.6e} (descent); natural sense: {_sense}")

        if g_a_sq_M < 1e-14:
            print("  ∇a effectively zero — cannot proceed.")
            break

        # ── L-BFGS pre-flight diagnostic ──
        # Compares V_a against its previous-step snapshot to decide
        # whether a Newton-direction approximation (L-BFGS) would buy
        # anything over GN steepest descent. See the [Diag] explanation
        # at the top of this loop's state init.
        if prev_V_a is not None:
            with stop_annotating():
                cross_M = inner_M_form(prev_V_a, V_a,
                                        riesz_alpha, riesz_beta,
                                        metric=riesz_metric,
                                        mu=riesz_mu, lam=riesz_lambda)
                eta = Function(V_2d, name="V_a_step_diff")
                eta.dat.data[:] = (np.asarray(prev_V_a.dat.data_ro)
                                   - np.asarray(V_a.dat.data_ro))
                sn = inner_M_form(prev_V_a, eta,
                                   riesz_alpha, riesz_beta,
                                   metric=riesz_metric,
                                   mu=riesz_mu, lam=riesz_lambda)
                nn = inner_M_form(eta, eta,
                                   riesz_alpha, riesz_beta,
                                   metric=riesz_metric,
                                   mu=riesz_mu, lam=riesz_lambda)
            cos_VprevV = (cross_M
                          / ((prev_g_a_sq_M * g_a_sq_M) ** 0.5
                             if prev_g_a_sq_M * g_a_sq_M > 0 else float('nan')))
            eta_rel    = ((nn / prev_g_a_sq_M) ** 0.5
                          if prev_g_a_sq_M > 1e-30 else float('nan'))
            if nn > 1e-30 and prev_g_a_sq_M > 1e-30:
                rayleigh_tight = (sn * sn) / (prev_g_a_sq_M * nn)
            else:
                rayleigh_tight = float('nan')
            print(f"  [Diag] cos(V_a_prev, V_a)_M           = {cos_VprevV:+.6f}")
            print(f"  [Diag] ||eta||_M / ||V_a_prev||_M      = {eta_rel:.4e}  "
                  f"(eta = V_a_prev − V_a)")
            print(f"  [Diag] Rayleigh tightness cos²(V_a_prev, eta)_M "
                  f"= {rayleigh_tight:.6f}  "
                  f"(1 = locked eigenmode → L-BFGS only rescales)")

        # ── Gauss-Newton step direction ──  (Fix 1: scale by d_true)
        # Linear-a model:  a(T + s·V_a) ≈ a + s·⟨∇a, V_a⟩ = a + s·d_true,
        # where d_true = ⟨∇a, V_a⟩ is the TRUE directional derivative computed
        # above. We scale by d_true, NOT g_a_sq_M = M(V_a, V_a). The two are
        # equal ONLY without interior masking (then V_a = M⁻¹∇a). With
        # mask_interior=True the Riesz RHS is the boundary part P_∂Ω∇a, so
        # V_a = M⁻¹P_∂Ω∇a and the identity breaks: g_a_sq_M stays > 0 but
        # d_true is sign-INDEFINITE (the DiagDD line measured d_true = −4.93
        # for the elasticity metric → −V_a would ASCEND if scaled by g_a_sq_M;
        # +0.20·g_a_sq_M for H¹ → 5× under-scaled, which poisoned the pred
        # model in step 2). Scaling by d_true:
        #   gn_scale = −ε/d_true   ⇒   da/dα|₀ = gn_scale·d_true = −ε
        # is GUARANTEED descent and lands a at a_target at α=1 in the linear
        # model — for ANY metric, regardless of masking — and restores
        # consistency with the pred slope grad_norm_sq_M = 2ε² (which assumes
        # da/dα|₀ = −ε). The IFT sign fix rhs_adj=[0,−1] still makes ∇a the
        # true +∂a/∂T (Taylor-verified); d_true just carries whatever sign the
        # masked lift actually has, so the step is always downhill.
        eps    = float(a_bif - a_target)
        eps_sq = eps * eps
        # Guard: d_true ≈ 0 means V_a ⟂ ∇a (degenerate direction) — scaling
        # would blow up. Bail like the g_a_sq_M ≈ 0 case.
        if abs(d_true) < 1e-14:
            print(f"  d_true ≈ 0 ({d_true:.2e}) — V_a ⟂ ∇a, cannot scale "
                  f"step; stopping.")
            break
        gn_scale = -eps / d_true
        V_rep = Function(V_2d, name="step_GN")
        V_rep.dat.data[:] = gn_scale * np.asarray(V_a.dat.data_ro)

        step_norm_M  = abs(gn_scale) * (g_a_sq_M  ** 0.5)
        step_norm_L2 = abs(gn_scale) * (g_a_sq_L2 ** 0.5)
        print(f"  [GN] α=1 step: ||step||_L2 = {step_norm_L2:.4e}  "
              f"||step||_M = {step_norm_M:.4e}  "
              f"(ε = {eps:+.4e}, ε² = {eps_sq:.4e}, d_true = {d_true:+.4e})")

        # Under the GN parametrisation the linear J-slope is
        #     -dJ/dα |_{α=0} = +2·ε² ,
        # i.e. pred_lin(α) = α·2ε² is the linear-a model's predicted J
        # reduction. We rely on the actual ρ-check after MS to catch
        # any over-aggressive α (no model-based pre-rejection): MS
        # converges cleanly, mesh-quality is monitored, and rho =
        # (J − J_try)/pred_lin self-corrects via line search.
        grad_norm_sq_M = 2.0 * eps_sq

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

        # ── Safeguarded 1D line search along the fixed Riesz descent direction
        # V_rep (step = T_2d + alpha·V_rep). This is NOT a trust region / dogleg:
        # the outer residual r(T) = a_bif(T) − a_target is SCALAR, so the
        # Gauss-Newton direction is collinear with the gradient (Cauchy)
        # direction — a dogleg's two legs coincide and it degenerates to 1D
        # scaling. So we search only the step length `alpha = step_len`, with a
        # trust-region-style ρ acceptance ratio (actual/predicted J reduction),
        # cross-step adaptation of `step_init` (×2 on ρ>rho_grow, hold else), and
        # quadratic-interpolation backtracking on reject (see the else branch
        # below). The genuine Powell-dogleg trust region lives in the INNER
        # Moore-Spence (2D residual), in locate_bifurcation_points_symmetric.
        print(f"\n  Line search (step_init = {step_init:.3e}, "
              f"step_max = {step_max:.3e}, "
              f"rho_accept = {rho_accept}, rho_grow = {rho_grow}, "
              f"branch_C = {branch_C:.2g})...")
        step_len    = float(step_init)
        accepted = False
        accepted_rho = float('nan')

        for bt in range(max_backtrack):
            if step_len < alpha_min:
                print(f"  [LS] step_len {step_len:.3e} < alpha_min — aborting.")
                break

            alpha = step_len

            with stop_annotating():
                T_2d_try = T_2d.copy(deepcopy=True)
                T_2d_try.dat.data[:] += alpha * V_rep.dat.data_ro

            with stop_annotating():
                mesh2d.coordinates.assign(X_ref_2d + T_2d_try)
                mesh_ok = check_2d_mesh_quality(mesh2d,
                                                ref_signs=ref_signs_2d,
                                                tol_quality=mesh_quality_floor)

            if not mesh_ok:
                print(f"  [LS bt={bt}] Mesh quality failed, shrinking step_len.")
                step_len *= alpha_backtrack
                continue

            try:
                with stop_annotating():
                    G_try, U_m_try, u_bar_try, p_bar_try = \
                        _solve_bg_on_mesh(
                            mesh2d, R, H, W, Re)
            except Exception as e:
                print(f"  [LS bt={bt}] Background flow failed: {e}")
                step_len *= alpha_backtrack
                continue

            shared_try = (R, H, W, L_c, U_c, Re,
                          G_try, U_m_try, u_bar_try, p_bar_try)

            md_try = dict(md)
            md_try['u_bar_2d']       = u_bar_try
            md_try['p_bar_tilde_2d'] = p_bar_try
            md_try['G']              = G_try
            md_try['U_m']            = U_m_try

            # line-search is off-tape — no AD adjoint to keep consistent.
            # The subsequent MS evaluates forces using LIVE mesh2d.coords;
            # log_test.txt empirically showed MS@reference gives correct
            # descent rho whereas MS@deformed (after restore) gives ascent.
            # So: restore=False — mesh2d stays at reference after the call,
            # matching the agg.txt manual-swap pattern.
            with stop_annotating():
                with xi_channel_ref_swap(X_ref_2d, restore=False):
                    xi_ch_try = build_xi_channel_from_T2d(
                        T_2d_try, md['mesh3d'], md['X_ref'], mesh2d,
                        md['R'], md['W'], md['H'])
                xi_ch_static = Function(md['V_def'], name="xi_channel_static")
                xi_ch_static.dat.data[:] = xi_ch_try.dat.data_ro
            md_try['xi_channel'] = xi_ch_static

            print(f"  [LS bt={bt}] alpha = {alpha:.3e} → running symmetric MS...")
            try:
                with stop_annotating():
                    r_try, a_try, conv_try, F_norm_try, G_norm_try, \
                        stalled_try = moore_spence_solve_symmetric(
                            r_bif, a_bif, shared_try,
                            tol=ms_tol, max_iter=ms_max_iter,
                            md=md_try,
                            dr_init=float(r_bif - r_ref),
                            da_init=float(a_bif - a_ref),
                            relinearize_basis=True)
            except Exception as e:
                print(f"  [LS bt={bt}] Symmetric MS failed: {e}")
                history[-1]['trials'].append({
                    'alpha': alpha, 'conv': False,
                    'F_norm': float('nan'), 'a_try': float('nan'),
                    'note': f'exception: {e}',
                })
                step_len *= alpha_backtrack
                continue

            # STALL with |F_r| at machine eps == reached FE resolution of the
            # bifurcation (residual is all in J_zz at the J_rz-asymmetry-noise
            # floor); not a sign that alpha was too aggressive. Gate by |F_r|
            # so genuinely unconverged MS (F_r stuck non-zero) still shrinks.
            ms_floor_F = 1e-10
            if (not conv_try) and stalled_try \
                    and float(F_norm_try) < ms_floor_F:
                print(f"  [LS bt={bt}] MS stalled at FE-noise floor "
                      f"(|F_r|={F_norm_try:.2e} < {ms_floor_F:.0e}, "
                      f"|G|={G_norm_try:.2e}); accepting as converged-to-"
                      f"discretisation-limit.")
                conv_try = True

            history[-1]['trials'].append({
                'alpha':  alpha,
                'conv':   bool(conv_try),
                'F_norm': float(F_norm_try),    # |F_r| at MS terminal iterate
                'a_try':  float(a_try),
            })

            if not conv_try:
                print(f"  [LS bt={bt}] Symmetric MS did not converge.")
                if diagnose_basis and not _basis_diag_done:
                    _basis_diag_done = True
                    diagnose_basis_staleness(
                        md_try, shared_try, float(r_try), float(a_try),
                        r_ref, a_ref,
                        ms_tol=ms_tol, ms_max_iter=ms_max_iter,
                        stalled_F=float(F_norm_try), stalled_M=float(G_norm_try))
                step_len *= alpha_backtrack
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
                        print(f"  [LS bt={bt}] Branch-tracking FAILED "
                              f"(ratio {branch_ratio:.3e} > {branch_C:.2g}) "
                              f"— shrinking step_len.")
                        step_len *= alpha_backtrack
                        continue
                else:
                    print(f"  [Branch] DOF-shape mismatch "
                          f"({u_old_arr.shape} vs {u_new_arr.shape}) — check skipped.")
            except Exception as e:
                print(f"  [Branch] check skipped ({e})")

            J_try = (float(a_try) - float(a_target)) ** 2

            # Quadratic-a line-search model (consistent with the d_true step scaling).
            # With gn_scale = −ε/d_true the step is T += α·gn_scale·V_a, so
            #   a(α) ≈ a_old − α·ε + ½·α²·(ε²/d_true²)·Q ,  Q = ⟨V_a,∇²a·V_a⟩
            # and the J=(a−a*)² reduction along the GN step is
            #   pred = α·2ε² − α²·ε²·(1 + ε·Q/d_true²).
            # Q comes from the previous accepted step (lag-by-one); its Q_eff
            # was reconstructed with the SAME d_true² scaling below. Step 1
            # has Q_eff_prev=None → linear fallback.
            actual_red = float(J) - float(J_try)
            if Q_eff_prev is not None and np.isfinite(Q_eff_prev):
                k_dim = eps * Q_eff_prev / (d_true ** 2)
                pred = (alpha * grad_norm_sq_M
                        - alpha * alpha * eps_sq * (1.0 + k_dim))
                pred_label = f"quad(k={k_dim:+.3f})"
            else:
                k_dim = float('nan')
                pred = alpha * grad_norm_sq_M
                pred_label = "lin(init)"
            if pred > 1e-30:
                rho = actual_red / pred
            elif actual_red > 0.0:
                # Fix(3): the (heuristic, lag-by-one) quadratic model predicts
                # no reachable reduction (pred ≤ 0), but J ACTUALLY decreased
                # and the step is already validated (MS converged, mesh-OK,
                # branch-OK — all checked above before this point). Ground
                # truth overrides a stale model: accept. Without this, a
                # genuinely-improving step was rejected purely on a blown-up
                # Q_eff (H¹ step 2: a_try 0.109, J_try < J, rho forced to −1).
                rho = 1.0
            elif pred < -1e-30:
                rho = -1.0
            else:
                rho = -1.0 if J_try > J else 1.0
            # Diagnostic: residual model error in α² units. With the linear
            # model this was ≈ 2·d²J/dα²; with the quadratic model it now
            # captures the cubic+ residual and should be ≪ the leading
            # quadratic term when the lag-by-one Q estimate is good.
            if alpha > 1e-5:
                H_obs = 2.0 * (pred - actual_red) / (alpha * alpha)
            else:
                H_obs = float('nan')
            print(f"  [LS bt={bt}] a_try = {a_try:.6f},  "
                  f"J_try = {J_try:.4e}  (vs J = {J:.4e})  rho = {rho:+.4f}  "
                  f"[pred_{pred_label}={pred:.2e}, H_obs={H_obs:.2e}]")

            if rho >= rho_accept:
                print(f"  [LS] Step ACCEPTED (bt={bt}, alpha={alpha:.4e}, "
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
                # Safeguarded quadratic-interpolation backtracking
                # (Nocedal-Wright §3.5). Instead of blindly halving step_len, fit
                # q(α) = J + J'(0)·α + c·α² through the values we already have —
                # J(0)=J, J'(0)=-grad_norm_sq_M (the descent slope of J=(a-a*)²
                # along the GN step, = -2ε²), and J(alpha)=J_try — and jump to
                # its minimiser α* = -J'(0)/(2c). When the trial overshot a
                # curved a_bif(T), this lands near the target in ONE more trial
                # instead of bisecting toward it. Clamp to [0.1·α, 0.5·α] (the
                # standard safeguard: guarantees ≥2× shrink, never a tiny jump),
                # and fall back to plain halving if the parabola is non-convex
                # (c ≤ 0). No new tuned parameters; uses the same J / slope the
                # rho-test already computed this iteration.
                Jp0 = -float(grad_norm_sq_M)           # dJ/dα|₀ < 0 (descent)
                c = (float(J_try) - float(J) - Jp0 * alpha) / (alpha * alpha)
                if c > 1e-30:
                    alpha_star = -Jp0 / (2.0 * c)
                    step_len = float(np.clip(alpha_star,
                                          0.1 * alpha, alpha_backtrack * alpha))
                    print(f"  [LS bt={bt}] quad-interp backtrack: "
                          f"α* = {alpha_star:.4e} → step_len = {step_len:.4e} "
                          f"(vs halve {alpha_backtrack*alpha:.4e})")
                else:
                    step_len *= alpha_backtrack

        if not accepted:
            print(f"  [ShapeOpt-Sym] line search failed — stopping.")
            break

        prev_step_init = step_init
        # Standard line-search step_init update: grow on good rho, hold otherwise.
        # No curvature-based cap — if the doubled seed is too aggressive,
        # the next step's line search backtracks it; if it's just right, we save
        # MS solves by not artificially shrinking it.
        if accepted_rho > rho_grow:
            step_init = min(2.0 * step_len, float(step_max))
        else:
            step_init = min(step_len, float(step_max))
        if step_init != prev_step_init:
            print(f"  [LS] step_next: {prev_step_init:.3e} → {step_init:.3e}  "
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

        # ── Curvature diagnostic (cheap, no extra MS-solve) ──
        # The GN step is T += α·gn_scale·V_a with gn_scale = −ε/d_true. Under
        # the linear-a + Taylor-quadratic model:
        #     a(T+step) = a_old − α·ε + ½·α²·(ε²/d_true²)·⟨V_a, ∇²a·V_a⟩
        # Hence  ε_new = (1−α)·ε + ½·α²·ε²·Q/d_true²  with  Q := ⟨V_a,∇²a·V_a⟩.
        # Solve for Q from the *observed* (ε_old, ε_new, α, d_true) of this
        # step. MUST use d_true² (the true step denominator), NOT g_a_sq_M²,
        # to stay consistent with the gn_scale fix; Q_eff then feeds next
        # step's k_dim with the same d_true² scaling.
        #
        # - Q stable over outer steps  ⇒  ε_new ∝ ε²  ⇒  quadratic convergence.
        # - Q growing as ε ↘            ⇒  bifurcation parameter has higher
        #     local curvature near the target shape — *physical* slowdown.
        eps_old = float(eps)
        eps_new = float(a_bif - a_target)
        ratio   = (eps_new / eps_old) if abs(eps_old) > 1e-30 else float('nan')
        if abs(alpha) > 1e-12 and eps_sq > 1e-30 and abs(d_true) > 1e-30:
            Q_eff = (2.0 * (d_true ** 2)
                     * (eps_new - (1.0 - alpha) * eps_old)
                     / (alpha * alpha * eps_sq))
        else:
            Q_eff = float('nan')
        k_obs = ((eps * Q_eff / (d_true ** 2))
                 if (np.isfinite(Q_eff) and abs(d_true) > 1e-30)
                 else float('nan'))
        print(f"  [Curv-Diag] ε: {eps_old:+.4e} → {eps_new:+.4e}   "
              f"ratio = {ratio:+.4f}   (quad-conv ⇔ ratio ∝ ε)")
        print(f"  [Curv-Diag] Q_eff = ⟨V_a, ∇²a·V_a⟩ ≈ {Q_eff:.4e}   "
              f"k_obs = ε·Q/d_true² = {k_obs:+.4e}  (this step's quad coeff)")
        if np.isfinite(Q_eff):
            Q_eff_prev = Q_eff
        target_eps = float(tol_J) ** 0.5
        if 0 < ratio < 1 and abs(eps_new) > target_eps:
            steps_remaining = float(np.log(target_eps / abs(eps_new))
                                    / np.log(ratio))
            print(f"  [Curv-Diag] If ratio holds: ~{steps_remaining:.1f} more "
                  f"steps to ε < √tol_J = {target_eps:.2e}")

        # ── Snap-and-reset: absorb (dr, 0, da) into xi_baseline and
        # re-solve basis around the new bif.
        print(f"  [Snap] Re-solving ALE basis around new bif point...")
        reset_ale_basis_for_step(md, drift_dr, 0.0, drift_da)
        r_ref, a_ref = float(r_bif), float(a_bif)

        with stop_annotating():
            mesh2d.coordinates.assign(X_ref_2d + T_2d)

        # ── Mesh-quality monitor on the accepted, deformed cross-section ──
        # The Riesz H¹/elasticity solve already relaxes the interior per step
        # (boundary-only RHS → V_a is the elastic extension of the boundary
        # gradient). This just makes the ACCUMULATED quality visible and flags
        # when the cross-section mesh is starting to distort — the line-search gate
        # (mesh_quality_floor) is what actually bounds it during the line
        # search. q = 4√3·A/Σℓ² ∈ (0,1], 1 = equilateral.
        q_min, q_mean = mesh2d_cell_quality(mesh2d)
        flag = "  ⚠ below warn level" if q_min < mesh_quality_warn else ""
        print(f"  [MeshQ] accepted step: q_min = {q_min:.4f}  "
              f"q_mean = {q_mean:.4f}  "
              f"(floor = {mesh_quality_floor:.2f}, warn = {mesh_quality_warn:.2f})"
              f"{flag}")
        if q_min < mesh_quality_warn:
            print(f"  [MeshQ] cross-section mesh distorting (q_min={q_min:.4f}); "
                  f"line search will reject trials below floor={mesh_quality_floor:.2f}. "
                  f"If progress stalls here, consider the linear-elasticity "
                  f"Riesz metric upgrade (couples ε(V)/div V) before shrinking.")

        # Snapshot V_a for the [Diag] cosine/Rayleigh diagnostic at the
        # next outer step. V_a here was computed at the *start* of this
        # step (before T_2d was modified); pairing it against the V_a
        # recomputed at the start of the next step reveals whether the
        # descent direction is locked or rotating.
        with stop_annotating():
            prev_V_a = V_a.copy(deepcopy=True)
            prev_g_a_sq_M = g_a_sq_M

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
                                W, H, a_bif, J_final, plot_dir)

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


def taylor_test_field_eval_block(md, u_bar_2d, *,
                                 n_eps=5, eps0=1e-3, factor=0.5, seed=0):
    """Isolated Taylor test of ``differentiable_field_eval`` w.r.t. xi.

    Strips away the perturbed-flow downstream: build J(xi) = ∫|lift(xi)|² dx
    directly from differentiable_field_eval and FD-check its AD gradient. This
    pinpoints whether the ~+1.35 spurious lifting-sensitivity (branch1 AD gave
    ∂F_r/∂T = -1.61 vs forward FD -2.97) lives in THIS block's adjoint
    (rate1→1) or downstream in perturbed_flow (rate1→2 here).
    """
    from background_flow_differentiable import (
        differentiable_field_eval, _build_CG1_to_CG2_map)

    mesh3d   = md['mesh3d']
    V_def    = md['V_def']
    X_ref    = md['X_ref']
    R_, W_, H_ = md['R'], md['W'], md['H']
    field_dim = u_bar_2d.ufl_shape[0]

    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    M    = _build_CG1_to_CG2_map(mesh3d)

    print(f"\n{'#'*70}")
    print(f"#  TAYLOR TEST  differentiable_field_eval  (∂/∂xi — isolated block)")
    print(f"#  field_dim={field_dim}, eps0={eps0:.1e}, factor={factor}, "
          f"n_eps={n_eps}")
    print(f"{'#'*70}")

    # Smooth direction h in V_def (CG1 vector on mesh3d), bounded so the
    # perturbed query points stay near the reference cross-section.
    with stop_annotating():
        Xc = np.asarray(X_ref.dat.data_ro)
        h = Function(V_def, name="xi_dir")
        h.dat.data[:, 0] = np.sin(Xc[:, 2])
        h.dat.data[:, 1] = np.cos(Xc[:, 2])
        h.dat.data[:, 2] = np.sin(Xc[:, 0] + Xc[:, 1])

    set_working_tape(Tape())
    continue_annotation()
    xi  = Function(V_def, name="xi_ctrl")
    out = differentiable_field_eval(
        xi, X_ref, u_bar_2d, R_, W_, H_, V_def, V_3d, field_dim, M)
    J   = assemble(inner(out, out) * dx)
    rf  = ReducedFunctional(J, Control(xi))
    g   = rf.derivative()
    stop_annotating()

    with stop_annotating():
        zero = Function(V_def)
    J0   = float(rf(zero))
    dJdm = float(np.sum(np.asarray(g.dat.data_ro)
                        * np.asarray(h.dat.data_ro)))
    print(f"  J(0) = {J0:.8e}   ⟨g, h⟩_l2 = {dJdm:+.6e}")

    rows = []
    eps = float(eps0)
    for _k in range(n_eps):
        with stop_annotating():
            xip = Function(V_def)
            xip.dat.data[:] = eps * np.asarray(h.dat.data_ro)
        Jp = float(rf(xip))
        r0 = abs(Jp - J0)
        r1 = abs(Jp - J0 - eps * dJdm)
        rows.append((eps, r0, r1))
        print(f"  eps={eps:.3e}  J={Jp:.8e}  r0={r0:.3e}  r1={r1:.3e}")
        eps *= factor

    print(f"\n  {'eps':>11} {'r0':>12} {'r1':>12} {'rate0':>7} {'rate1':>7}")
    prev = None
    rates1 = []
    for (e, r0, r1) in rows:
        if prev is None:
            rt0 = rt1 = float('nan')
        else:
            pe, pr0, pr1 = prev
            lr = np.log(pe / e)
            rt0 = np.log(pr0 / r0) / lr if r0 > 0 and pr0 > 0 else float('nan')
            rt1 = np.log(pr1 / r1) / lr if r1 > 0 and pr1 > 0 else float('nan')
            if np.isfinite(rt1):
                rates1.append(rt1)
        print(f"  {e:>11.3e} {r0:>12.3e} {r1:>12.3e} {rt0:>7.3f} {rt1:>7.3f}")
        prev = (e, r0, r1)
    best = max(rates1) if rates1 else float('nan')
    verdict = "PASS (block OK)" if np.isfinite(best) and best > 1.8 \
        else "FAIL (block adjoint BUG)"
    print(f"  → best rate1 = {best:.3f}  ⇒  {verdict}")
    print(f"{'#'*70}\n")
    return best


def run_from_main_symmetric(r0, a_target, max_steps=50, tol_J=1e-8,
                             alpha_step=0.05, riesz_alpha=1.0,
                             riesz_beta=1e-2,
                             riesz_metric="h1",
                             riesz_mu=None, riesz_lambda=None,
                             ms_tol=1e-12, ms_max_iter=15, n_grid_2d=128,
                             a0=0.1375,
                             mesh_quality_floor=0.20, mesh_quality_warn=0.35,
                             taylor_check=False, taylor_only=True,
                             taylor_kw=None, block_test=False,
                             diagnose_basis=False):
    """Top-level driver matching ``shape_optimization.run_from_main`` but
    without z arguments (z=0 fixed by symmetry) and using the symmetric
    Newton + Moore-Spence path.
    """
    with stop_annotating():
        bg = background_flow_differentiable(R, H, W, Re)
        G, U_m, u_bar_2d, p_bar_tilde = bg.solve_2D_background_flow()

    shared_data = (R, H, W, L_c, U_c, Re,
                   G, U_m, u_bar_2d, p_bar_tilde)

    r_eq, md0, dr0 = newton_root_refine_symmetric(
        r0, a0, shared_data, tol=1e-10, max_iter=15)

    # Isolated AD-block check (fast — skips the expensive MS / shape-gradient
    # path). Returns right after so we can iterate on the block adjoint.
    if block_test:
        taylor_test_field_eval_block(md0, u_bar_2d)
        return {'block_test': True}

    r_ref, a_ref = r0, a0

    with stop_annotating():
        r_bif, a_bif, conv_ms, F_norm0, _G0, _stalled0 = \
            moore_spence_solve_symmetric(
                r_eq, a0, shared_data,
                tol=ms_tol, max_iter=ms_max_iter,
                md=md0, dr_init=dr0, da_init=0.0)
        print(f"  Initial bifurcation residual: |F_r| = {F_norm0:.3e}")

    if not conv_ms:
        raise RuntimeError(
            "Symmetric Moore-Spence did not converge for initial domain")

    bif_init = {'r': r_bif, 'a': a_bif, 'converged': True}

    print(f"\n  Initial bifurcation: a = {a_bif:.6f} (target = {a_target:.4f})")

    result = run_shape_optimization_symmetric(
        a_target, shared_data, md0, bif_init,
        r_ref=r_ref, a_ref=a_ref,
        max_steps=max_steps, tol_J=tol_J,
        alpha_step=alpha_step,
        riesz_alpha=riesz_alpha, riesz_beta=riesz_beta,
        riesz_metric=riesz_metric, riesz_mu=riesz_mu, riesz_lambda=riesz_lambda,
        ms_tol=ms_tol, ms_max_iter=ms_max_iter,
        n_grid_2d=n_grid_2d,
        mesh_quality_floor=mesh_quality_floor,
        mesh_quality_warn=mesh_quality_warn,
        taylor_check=taylor_check, taylor_only=taylor_only,
        taylor_kw=taylor_kw,
        diagnose_basis=diagnose_basis)

    # When taylor_only stopped the run after the gradient check, skip the
    # checkpoint/section-export bookkeeping — there is no optimised T to save.
    if result.get('taylor_only'):
        return result

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
            R_phys=R_phys, H_phys=H_phys, W_phys=W_phys,
            Q_phys=Q_phys, rho_phys=rho_phys, mu_phys=mu_phys,
            R=R, H=H, W=W,
            L_c=L_c, U_c=U_c, Re=Re,
            a_target=a_target,
            n_grid_2d=n_grid_2d,
            filepath="output_optimized_section_symmetric.pkl",
        )
    except Exception as e:
        print(f"  Warning: could not save optimized-section pickle: {e}")

    return result


if __name__ == "__main__":

    # a_target imported from problem_setup (= a_target_phys / L_c).

    # Initial guess from the bifurcation diagram, z=0 hardcoded.
    r_off_init = 0.6098
    a_init     = 0.1375

    # ── Gradient check switch ──
    # While the shape gradient ∂a_bif/∂T is being debugged, run a Taylor test
    # at step 1 and stop (taylor_only=True) — no extra CLI argument needed,
    # this IS the normal run. Set TAYLOR_CHECK = False to actually optimise.
    TAYLOR_CHECK = False

    # ── Isolated AD-block test switch ──
    # When True, run ONLY the differentiable_field_eval Taylor test (fast,
    # skips MS / shape-gradient) and stop. Use while fixing the block adjoint.
    BLOCK_TEST = False

    # ── Riesz / mesh-moving metric switch ──
    # "h1"         : α∫∇V:∇W + β∫V·W  (vector-Laplacian; historical default,
    #                each component smoothed independently).
    # "elasticity" : 2μ∫ε(V):ε(W) + λ∫(div V)(div W) + β∫V·W  (linear
    #                elasticity; couples components, penalises local volume
    #                change → better-conditioned mesh moves under large
    #                boundary deformation). μ, λ default to riesz_alpha.
    # The gradient is metric-independent (Taylor-verified either way); only
    # the descent direction / step scaling change. Flip this to compare.
    RIESZ_METRIC = "elasticity"  # ← "h1" (vector-Laplacian) or "elasticity"

    # ── ALE-basis staleness probe ──
    # When True, the FIRST non-converged inner-MS trial (e.g. the α=0.5 stall
    # at step 1) triggers a one-shot, non-destructive re-centring of the ALE
    # basis at that trial's bif point + an MS re-solve. Decides between
    # cause (1) basis staleness vs (2/3) pitchfork-flattening / F_z mesh noise.
    DIAGNOSE_BASIS = True

    result = run_from_main_symmetric(
        a_target=a_target,
        r0=r_off_init,
        a0=a_init,
        block_test=BLOCK_TEST,
        taylor_check=TAYLOR_CHECK,
        taylor_only=True,
        diagnose_basis=DIAGNOSE_BASIS,
        taylor_kw=dict(eps0=4e-3, n_eps=4, factor=0.5,
                       dir_seed=0, dir_smooth=0.05, dir_max=1.0),
        # At ~5e-4 da/step (initial) growing as line search auto-promotes step_len,
        # the trajectory reaches a≈0.1 in roughly 40-70 steps. 100 is a
        # defensive budget. The 3-step smoke test in log.txt confirmed the
        # patch descends correctly (rho ≈ +0.49 across the validated range).
        max_steps=100,
        tol_J=1e-8,
        # α=1 is the *natural* full Newton step under the GN parametrisation
        # (T += α·ε/g_a²·V_a; α=1 lands at a_target in the linear-a model).
        # The line search's ρ-check + mesh-ok + branch-tracking catch over-aggressive
        # trials; max_backtrack=12 halvings reach down to ~1.2e-4 if needed.
        alpha_step=1.0,
        # ── safer fallback if step 1 burns too many backtracks ──
        # alpha_step=1e-2,
        riesz_alpha=0.1,
        riesz_beta=1e-2,
        riesz_metric=RIESZ_METRIC,
        # μ, λ for the elasticity metric; None → default to riesz_alpha each.
        # Ignored when RIESZ_METRIC == "h1".
        riesz_mu=None,
        riesz_lambda=None,
        ms_tol=1e-12,
        ms_max_iter=30,
        # 64 keeps the gradient check fast; the gradient bug is structural
        # (resolution-independent). Raise to 120 for the real optimisation.
        n_grid_2d=64 if TAYLOR_CHECK else 120,
    )

    if result.get('block_test'):
        print("\nBlock test done. Set BLOCK_TEST = False to continue.")
    elif result.get('taylor_only'):
        print("\nTaylor gradient check done (taylor_only). "
              "Set TAYLOR_CHECK = False to run the optimisation.")
    else:
        print("\nFinal result:")
        print(f"  a_bif = {result['a_bif']:.8f}")
        print(f"  J     = {result['J_final']:.6e}")
        print(f"  steps = {len(result['history'])}")
