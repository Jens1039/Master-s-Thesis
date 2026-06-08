import os
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
from contextlib import contextmanager
from datetime import datetime
from firedrake import *
import gc
from firedrake.adjoint import stop_annotating, annotate_tape, continue_annotation, get_working_tape, set_working_tape, Tape, ReducedFunctional, Control

from background_flow_differentiable import background_flow_differentiable, build_3d_background_flow_differentiable, vom_transfer, _compute_inv_perm
from perturbed_flow_differentiable import perturbed_flow_differentiable, _build_xi, check_mesh_quality, evaluate_forces, reset_ale_basis_for_step, DECOUPLED_LIFT
from locate_bifurcation_points import newton_root_refine, moore_spence_solve
from problem_setup import *


# ---------------------------------------------------------------------------
# Defensive patch: firedrake's GenericSolveBlock.prepare_evaluate_hessian
# implicitly returns ``None`` when ``hessian_input`` or ``tlm_output`` is
# None (lines ~403-409), but the matching ``evaluate_hessian_component``
# (line 440) then does ``prepared["adj_sol2"]`` and crashes.  Add a
# short-circuit at the top of ``evaluate_hessian_component`` so any block
# with no SOA contribution simply returns None instead of crashing.
# ---------------------------------------------------------------------------
def _install_solveblock_hessian_guard():
    from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock
    orig = GenericSolveBlock.evaluate_hessian_component

    def wrapper(self, inputs, hessian_inputs, adj_inputs, block_variable,
                idx, relevant_dependencies, prepared=None):
        if prepared is None:
            return None
        return orig(self, inputs, hessian_inputs, adj_inputs, block_variable,
                    idx, relevant_dependencies, prepared)

    GenericSolveBlock.evaluate_hessian_component = wrapper

_install_solveblock_hessian_guard()


# ──────────────────────────────────────────────────────────────────────────
# Ref-swap context manager for ``build_xi_channel_from_T2d``.
#
# The VOM lookup inside build_xi_channel_from_T2d uses query points clipped
# to the reference rectangle [0, W]×[0, H]. As soon as T_2d grows past ~1%
# of the cross-section, mesh2d.coordinates may sit at deformed positions
# whose extent no longer covers the clipped query set, so
# VertexOnlyMesh raises VertexOnlyMeshMissingPointsError.
#
# When ``xi_channel_ref_swap(X_ref_2d)`` is active, build_xi_channel_from_T2d
# performs an off-tape ``mesh2d.coordinates.assign(X_ref_2d)`` BEFORE the
# VOM construction. The element/local-coord lookup is frozen at that
# moment (inside its own ``with stop_annotating()``), so the subsequent
# on-tape interpolate is safe after the coord restoration.
#
# The ``restore`` switch governs what happens after the call returns:
#   - restore=True  (on-tape callers): mesh2d.coords are restored to the
#     pre-swap state, so any on-tape solve that ran on the deformed mesh
#     (and whose adjoint backward pass evaluates forms using LIVE
#     mesh.coords) sees the right state.
#   - restore=False (off-tape callers, e.g. TR-backtrack): mesh2d stays
#     at reference. The downstream off-tape MS solve evaluates forces
#     with mesh2d in its current state, and empirically MS@reference
#     gives the correct rho sign whereas MS@deformed flips it.
# ──────────────────────────────────────────────────────────────────────────
_xi_channel_X_ref_2d = None       # set via xi_channel_ref_swap CM
_xi_channel_restore  = True       # whether the swap is restored on exit


@contextmanager
def xi_channel_ref_swap(X_ref_2d, restore=True):
    """Activate the mesh2d → X_ref_2d coord swap inside
    ``build_xi_channel_from_T2d`` for the duration of the block.
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


def _solve_bg_on_mesh(mesh2d, R, H, W, Re_float):

    bg = background_flow_differentiable(R, H, W, Re_float,
                                        mesh2d=mesh2d)
    return bg.solve_2D_background_flow()


def build_xi_channel_from_T2d(T_2d, mesh3d, X_ref, mesh2d, R, W, H):

    # ── Optional ref-swap (see xi_channel_ref_swap docstring) ──
    # When the CM is active and T_2d is non-trivially non-zero, swap
    # mesh2d.coordinates to X_ref_2d off-tape for the VOM construction.
    # The T_2d-zero early-out matters: at step 1 of the shape-opt loop
    # T_2d is exactly zero, so mesh2d.coords are already at reference;
    # the swap is a no-op semantically but firedrake's internal cache
    # invalidation on the coord Function empirically flipped the resulting
    # shape-gradient direction in earlier attempts. Comparing via
    # T_2d.dat.data is robust to function-space-instance differences
    # between X_ref_2d and mesh2d.coordinates (a naive np.array_equal on
    # mesh2d.coords vs X_ref_2d returned False due to DOF reorderings).
    _ref_swap_active = (_xi_channel_X_ref_2d is not None)
    _saved_coords    = None
    if _ref_swap_active:
        with stop_annotating():
            t_max = float(np.max(np.abs(np.asarray(T_2d.dat.data_ro))))
        if t_max < 1e-14:
            _ref_swap_active = False
            print(f"    [xi-patch] context active, T_2d max={t_max:.2e} < 1e-14 "
                  f"→ no-op passthrough")
        else:
            mode = "swap+restore" if _xi_channel_restore else "swap (no restore)"
            print(f"    [xi-patch] context active, T_2d max={t_max:.2e} → {mode}")
            with stop_annotating():
                if _xi_channel_restore:
                    _saved_coords = mesh2d.coordinates.copy(deepcopy=True)
                mesh2d.coordinates.assign(_xi_channel_X_ref_2d)

    V_def = X_ref.function_space()

    # --- Step 1: VOM query points from REFERENCE 3D positions (off tape) ---
    with stop_annotating():
        X_ref_arr = X_ref.dat.data_ro.copy()          # (N, 3)
        rho_ref   = np.sqrt(X_ref_arr[:, 0]**2 + X_ref_arr[:, 1]**2)

        s_ref = rho_ref - R + 0.5 * W
        t_ref = X_ref_arr[:, 2] + 0.5 * H

        query_pts = np.column_stack([
            np.clip(s_ref, 0.0, W),
            np.clip(t_ref, 0.0, H),
        ])

        vom_ch   = VertexOnlyMesh(mesh2d, query_pts, missing_points_behaviour="error")
        V_vom_ch = VectorFunctionSpace(vom_ch, "DG", 0, dim=2)
        inv_perm_ch = _compute_inv_perm(vom_ch, query_pts)

        # Precompute cos/sin theta for cylindrical → Cartesian lifting.
        # Small safety: guard against nodes exactly on the axis (rho=0).
        safe_rho  = np.where(rho_ref > 0, rho_ref, 1.0)
        cos_arr   = X_ref_arr[:, 0] / safe_rho
        sin_arr   = X_ref_arr[:, 1] / safe_rho

        # Store as CG1 scalar Functions (off tape — they are fixed geometry)
        V_scalar  = FunctionSpace(mesh3d, "CG", 1)
        cos_fn    = Function(V_scalar, name="cos_theta_ch")
        sin_fn    = Function(V_scalar, name="sin_theta_ch")
        cos_fn.dat.data[:] = cos_arr
        sin_fn.dat.data[:] = sin_arr

    # --- Step 2: Interpolate T_2d onto VOM (ON TAPE — InterpolateBlock) ---
    T_vom = Function(V_vom_ch).interpolate(T_2d)

    # --- Step 3: Transfer to 2-component CG1 on mesh3d (ON TAPE — VOMTransferBlock) ---
    V_cg1_2 = VectorFunctionSpace(mesh3d, "CG", 1, dim=2)
    T_at_nodes = vom_transfer(T_vom, V_cg1_2, inv_perm_ch, V_vom_ch)

    # --- Step 4: Lift to 3D deformation (ON TAPE — InterpolateBlock) ---
    # T_at_nodes[0] is the radial deformation (delta_s),
    # T_at_nodes[1] is the axial  deformation (delta_t = delta_z).
    xi_channel = Function(V_def, name="xi_channel")
    xi_channel.interpolate(as_vector([
        T_at_nodes[0] * cos_fn,
        T_at_nodes[0] * sin_fn,
        T_at_nodes[1],
    ]))

    return xi_channel


def compute_DG_at_bif(r_bif, z_bif, a_bif, phi_bif, l_vec, mesh_data, r_ref, z_ref, a_ref):

    dr = float(r_bif - r_ref)
    dz = float(z_bif - z_ref)
    da = float(a_bif - a_ref)

    F_base, J_full, dJphi_dx = evaluate_forces(dr, dz, da, mesh_data, hessian_phi=phi_bif)

    J_sp    = J_full[:, :2]
    dF_da   = J_full[:, 2]

    DG = np.zeros((5, 5))
    DG[0:2, 0:2] = J_sp
    DG[0:2, 2]   = dF_da
    DG[2:4, 0]   = dJphi_dx[:, 0]
    DG[2:4, 1]   = dJphi_dx[:, 1]
    DG[2:4, 2]   = dJphi_dx[:, 2]
    DG[2:4, 3:5] = J_sp
    DG[4, 3:5]   = l_vec

    return DG, F_base, J_sp


def eval_forces_with_bg_on_tape(r_off, z_off, a, u_bar_2d, p_bar_2d, G_val, mesh_data, r_ref, z_ref, a_ref,
                                    T_2d=None, mesh2d=None, rz_on_tape=False,
                                    lift_on_tape=False, return_components=False):

    md     = mesh_data
    mesh3d = md['mesh3d']
    dr_fn_ctrl = dz_fn_ctrl = None

    if rz_on_tape:
        # Particle position ON TAPE (for mixed Hessian d²F/(dT dr) etc.)
        R_space  = FunctionSpace(mesh3d, "R", 0)
        dr_fn_ctrl = Function(R_space, name="delta_r").assign(float(r_off - r_ref))
        dz_fn_ctrl = Function(R_space, name="delta_z").assign(float(z_off - z_ref))
        da_fn      = Function(R_space, name="delta_a").assign(float(a  - a_ref))
        xi_particle = _build_xi(dr_fn_ctrl, dz_fn_ctrl, da_fn, md)
    else:
        # Particle position OFF TAPE (r/z/a are fixed constants)
        with stop_annotating():
            R_space  = FunctionSpace(mesh3d, "R", 0)
            dr_fn    = Function(R_space).assign(float(r_off - r_ref))
            dz_fn    = Function(R_space).assign(float(z_off - z_ref))
            da_fn    = Function(R_space).assign(float(a  - a_ref))
            xi_particle = _build_xi(dr_fn, dz_fn, da_fn, md)

    xi_lift = None
    if T_2d is not None and mesh2d is not None:
        # --- Channel wall deformation from T_2d (ON TAPE) ---
        xi_channel = build_xi_channel_from_T2d(
            T_2d, mesh3d, md['X_ref'], mesh2d,
            md['R'], md['W'], md['H'])

        if rz_on_tape:
            # Both xi_channel and xi_particle are on tape
            xi_sum = xi_channel + xi_particle
            xi_particle_for_lift = xi_particle
        else:
            # xi_particle is constant (off tape), gradient flows only
            # through xi_channel → T_2d.
            with stop_annotating():
                xi_particle_fn = Function(md['V_def'], name="xi_particle_const")
                xi_particle_fn.dat.data[:] = xi_particle.dat.data_ro
            xi_sum = xi_channel + xi_particle_fn   # AddBlock on tape
            xi_particle_for_lift = xi_particle_fn

        # Materialize the full deformation as a Function in V_def.
        # md['xi_baseline'] (snap from past accepted steps) is added
        # off-tape — it does not depend on the current Controls.
        xi_total = Function(md['V_def'], name="xi_total")
        xi_total.assign(md['xi_baseline'] + xi_sum)

        # Channel-decoupled lift geometry (see DECOUPLED_LIFT in
        # perturbed_flow_differentiable): the u_bar sampling positions exclude
        # xi_channel, so the lift is smooth in T while xi_particle stays in
        # (z-sampling for the Hessian-VP kept). Mirrors evaluate_forces' xi_lift
        # exactly → objective/gradient consistent. At T=0 this == xi_total.
        if DECOUPLED_LIFT:
            xi_lift = Function(md['V_def'], name="xi_lift")
            xi_lift.assign(md['xi_baseline'] + xi_particle_for_lift)

        # --- Deform 3D mesh: AssignBlock on tape ---
        mesh3d.coordinates.assign(md['X_ref'] + xi_total)

    else:
        # Fallback: no channel deformation — move mesh off tape
        xi_total = xi_particle
        with stop_annotating():
            mesh3d.coordinates.assign(
                md['X_ref'] + md['xi_baseline'] + xi_particle)

    with stop_annotating():
        check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    # --- 3D background flow with u_bar_2d ON TAPE ---
    # CHANNEL-DECOUPLED lift (DECOUPLED_LIFT, see perturbed_flow_differentiable
    # module header): sample u_bar_2d at xi_lift = X_ref + xi_baseline +
    # xi_particle (channel-undeformed), so the lift is smooth in T (xi_channel
    # excluded) while the (z) particle sampling stays on tape. The caller must
    # leave mesh2d at the REFERENCE cross-section so these channel-undeformed
    # query points locate correctly. This MATCHES evaluate_forces' xi_lift, so
    # the shape gradient is consistent with the bif objective. At T=0 it is a
    # bit-for-bit no-op (xi_lift == xi_total).
    #
    # Legacy fallbacks (DECOUPLED_LIFT False): lift_on_tape selects the fully-
    # coupled branch1 (full xi_total) or the frozen branch2 (X_ref/xi=None).
    if DECOUPLED_LIFT and xi_lift is not None:
        u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
            md['R'], md['H'], md['W'], G_val,
            mesh3d, md['tags'], u_bar_2d, p_bar_2d,
            X_ref=md['X_ref'], xi=xi_lift)
    elif lift_on_tape and (T_2d is not None and mesh2d is not None):
        u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
            md['R'], md['H'], md['W'], G_val,
            mesh3d, md['tags'], u_bar_2d, p_bar_2d,
            X_ref=md['X_ref'], xi=xi_total)
    else:
        u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
            md['R'], md['H'], md['W'], G_val,
            mesh3d, md['tags'], u_bar_2d, p_bar_2d,
            X_ref=None, xi=None)

    # --- Perturbed-flow Stokes solve (CachedStokesSolveBlock on tape) ---
    pf = perturbed_flow_differentiable(
        md['R'], md['H'], md['W'], md['L'],
        float(a), md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi_total, u_cyl_3d)

    if return_components:
        F_p_x, F_p_z, _comps = pf.F_p(return_components=True)
        comps = {k: float(v) for k, v in _comps.items()}
        return float(F_p_x), float(F_p_z), comps

    F_p_x, F_p_z = pf.F_p()
    if rz_on_tape:
        return F_p_x, F_p_z, dr_fn_ctrl, dz_fn_ctrl
    return F_p_x, F_p_z


def compute_shape_gradient(r_bif, z_bif, a_bif, phi_bif, l_vec, a_target, DG_5x5, T_2d, mesh2d, X_ref_2d,
                            R, H, W, Re_float, mesh_data, r_ref, z_ref, a_ref):
    """Shape derivative of the *bifurcation parameter* ``a_bif`` w.r.t. T.

    Returned cofunction is ∇a (not ∇J). The caller assembles the Gauss-
    Newton step (a_bif − a_target) · V_a / ‖V_a‖²_M from the Riesz lift
    V_a — this is the pseudo-inverse of the rank-1 leading-order Hessian
    2·∇a⊗∇a against the ε-scaled gradient 2ε·∇a. Two payoffs: (i) the
    ε-free factor ∇a is resolved well by AD whereas ∇J = 2ε·∇a sinks
    into the noise floor near the optimum; and (ii) the unit step α=1
    lands at a_target in one shot of the linear-a model, giving the
    outer trust region a fixed scale. Convergence: GN is quadratic at a
    zero-residual problem; plain SD on J = (a − a_target)² with fixed
    step is *linear* at the metric-dependent rate (1 − 2s·g_a²), not
    cubic — earlier "ε³ stagnation" / "Hessian ∝ ε" claims were wrong.

    Derivation (implicit function on the bifurcation constraint G = 0):

        a = a(T), defined implicitly by G(y, T) = 0 with
        y = (r, z, a, phi_r, phi_z) ∈ R^5.
        L_a = a + μ^T G ,    ∂L_a/∂y = 0  ⇒  DG_y^T μ = (0, 0, 1, 0, 0)
        da/dT = μ_0 · ∂F_p_x/∂T + μ_1 · ∂F_p_z/∂T
              + μ_2 · ∂(J·phi)_x/∂T + μ_3 · ∂(J·phi)_y/∂T
        (the last two terms are Hessian-VPs in TLM direction phi=(phi_r, phi_z))

    Note: ``a_target`` is kept in the signature only for callers that
    still need it for ε bookkeeping; the gradient itself no longer
    depends on it.
    """
    # ∂a/∂y for y = (r, z, a, phi_r, phi_z): pick out the a-component.
    rhs_adj = np.array([0., 0., 1., 0., 0.])
    cond_DG = np.linalg.cond(DG_5x5)
    print(f"  [Shape] cond(DG) = {cond_DG:.3e}")

    # get lagrange multiplier (now μ for ``a``, not for ``J`` — ε-free)
    lambda_adj = np.linalg.solve(DG_5x5.T, rhs_adj)

    lam_str = ', '.join(f'{v:.4e}' for v in lambda_adj)
    print(f"  [Shape] lambda_adj = [{lam_str}]")

    lam0 = float(lambda_adj[0])
    lam1 = float(lambda_adj[1])
    lam2 = float(lambda_adj[2])
    lam3 = float(lambda_adj[3])
    # lam4 = 0, since G_5 == ||phi|| - 1 doesn't depend on T

    set_working_tape(Tape())
    continue_annotation()

    c_T = Control(T_2d)

    mesh2d.coordinates.assign(X_ref_2d + T_2d)

    G_val, _, u_bar_2d_new, p_bar_2d_new = _solve_bg_on_mesh(mesh2d, R, H, W, Re_float)

    print(f"  [Shape] G = {G_val:.6e}")

    print(f"  [Shape] Evaluating forces at y* = "f"(r={r_bif:.4f}, z={z_bif:.4f}, a={a_bif:.4f})...")
    # The eval_forces_with_bg_on_tape internal call to build_xi_channel_from_T2d
    # does a VOM lookup with query points clipped to [0, W]×[0, H].
    # mesh2d.coordinates are currently at X_ref_2d + T_2d (deformed, from the
    # BG NS solve above); once T_2d grows past ~1% of the cross-section the
    # deformed boundary moves out of the reference rectangle and VOM raises
    # VertexOnlyMeshMissingPointsError. The context manager swaps
    # mesh2d.coords to reference (off-tape) for the build_xi_channel call,
    # then restores so that build_3d_background_flow_differentiable
    # (called later inside eval_forces) still sees the deformed mesh2d
    # for the u_bar_2d → 3D lifting.
    with xi_channel_ref_swap(X_ref_2d):
        F_p_x, F_p_z, dr_fn, dz_fn = eval_forces_with_bg_on_tape(
            r_bif, z_bif, a_bif, u_bar_2d_new, p_bar_2d_new, G_val,
            mesh_data, r_ref, z_ref, a_ref,
            T_2d=T_2d, mesh2d=mesh2d, rz_on_tape=True)

    print(f"  [Shape] F(y*) at current T_2d: " f"F_x = {float(F_p_x):.4e}, F_z = {float(F_p_z):.4e}")

    c_r = Control(dr_fn)
    c_z = Control(dz_fn)
    controls = [c_r, c_z, c_T]

    Lambda12 = lam0 * F_p_x + lam1 * F_p_z

    print("  [Shape] Computing G1/G2 shape gradient (first derivatives)...")
    rf_12      = ReducedFunctional(Lambda12, controls)
    d12        = rf_12.derivative()
    shape_grad = d12[2]

    Lambda34 = lam2 * F_p_x + lam3 * F_p_z

    with stop_annotating():
        R_space  = dr_fn.function_space()
        phi_r_fn = Function(R_space).assign(float(phi_bif[0]))
        phi_z_fn = Function(R_space).assign(float(phi_bif[1]))
    # Pass None (rather than a zero Function) for the T_2d direction.
    # A zero Function would force pyadjoint's TLM to walk every block
    # downstream of T_2d (mesh2d.coords -> BG NS solve -> ...), and
    # ufl.expand_derivatives chokes on CoordinateDerivative(zero) inside
    # the BG NS SolveBlock's evaluate_tlm_component.  None tells pyadjoint
    # "this control has no TLM input" and the path is skipped entirely —
    # mathematically identical (we want d²F/(dT dr) and d²F/(dT dz), not
    # d²F/dT² ; the T_2d direction is irrelevant to this Hessian-VP).
    m_dot = [phi_r_fn, phi_z_fn, None]

    print("  [Shape] Computing G3/G4 shape gradient (Hessian-VP)...")
    rf_34 = ReducedFunctional(Lambda34, controls)
    rf_34.derivative()
    H34   = rf_34.hessian(m_dot)

    # H34[2] is the T_2d component: lam2·dG3/dT + lam3·dG4/dT
    shape_grad.dat.data[:] += np.asarray(H34[2].dat.data_ro)

    stop_annotating()
    get_working_tape().clear_tape()
    gc.collect()

    # Restore 2D mesh to current T_2d (may have been modified by the tape run)
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)

    # Restore 3D mesh to bifurcation position + current channel shape.
    # (moore_spence_solve will reassign via its own tape, but setting
    # a clean state here avoids surprises in the line-search calls.)
    with stop_annotating():
        mesh3d_r = mesh_data['mesh3d']
        R_sp     = FunctionSpace(mesh3d_r, "R", 0)
        dr_r = Function(R_sp).assign(float(r_bif - r_ref))
        dz_r = Function(R_sp).assign(float(z_bif - z_ref))
        da_r = Function(R_sp).assign(float(a_bif - a_ref))
        xi_restore = _build_xi(dr_r, dz_r, da_r, mesh_data)
        mesh3d_r.coordinates.assign(
            mesh_data['X_ref'] + mesh_data['xi_baseline']
            + xi_restore + mesh_data['xi_channel'])

    return shape_grad, lambda_adj


def riesz_metric_form(u, v, *, metric="h1",
                      alpha=1.0, beta=1e-2, mu=None, lam=None):
    """UFL bilinear form for the Riesz / mesh-moving metric M(u, v).

    metric="h1"  (default, historically-validated vector-Laplacian):
        M(u,v) = α ∫∇u:∇v dx + β ∫u·v dx
        Each component is smoothed independently (no coupling).

    metric="elasticity"  (linear elasticity):
        M(u,v) = 2μ ∫ε(u):ε(v) dx + λ ∫(div u)(div v) dx + β ∫u·v dx,
        ε(u) = sym(∇u). Couples the components through the symmetric
        gradient and penalises local volume change via div u — the standard
        elasticity-based mesh-moving regulariser, which keeps the deformed
        mesh better conditioned than the plain Laplacian. μ, λ default to α.
        The β·L² term is retained for coercivity/scale (with corner pinning
        the form is already coercive, but β keeps it well-scaled).

    IMPORTANT — single source of truth. The TR step/pred calibration in the
    optimiser silently relies on the identity

        g_a_sq_M = ‖V_a‖²_M = ⟨∇a, V_a⟩    (V_a = M⁻¹·∇a, any SPD M),

    from which gn_scale = −ε/g_a_sq_M lands a at a_target (α=1) and
    grad_norm_sq_M = 2ε² is the pred slope. This holds for ANY metric ONLY
    IF the explicit ‖V_a‖²_M assembly uses the SAME form as the Riesz solve.
    Route every metric evaluation (solve, g_a_sq_M, inner_M_form) through
    THIS function so they cannot drift apart.
    """
    if metric == "h1":
        return alpha * inner(grad(u), grad(v)) * dx + beta * inner(u, v) * dx
    if metric == "elasticity":
        mu_  = alpha if mu  is None else mu
        lam_ = alpha if lam is None else lam
        eps_u = sym(grad(u))
        eps_v = sym(grad(v))
        return (2.0 * mu_ * inner(eps_u, eps_v) * dx
                + lam_ * div(u) * div(v) * dx
                + beta * inner(u, v) * dx)
    raise ValueError(f"unknown riesz metric '{metric}' "
                     f"(expected 'h1' or 'elasticity')")


def riesz_representative(shape_grad_cofunction, mesh2d,
                          alpha_elast=1.0, beta_l2=1e-2,
                          mask_interior=True,
                          fix_corners=True, W=None, H=None,
                          metric="h1", mu=None, lam=None):
    """Compute the H¹ Riesz representative of the shape gradient.

    Inner product:

        (V, W) = α ∫ ∇V:∇W dx + β ∫ V·W dx

    With ``mask_interior=True`` (default), the interior DOFs of the
    shape-gradient cofunction are zeroed *before* the Riesz solve
    (Hadamard projection). Zeroing happens on the RHS, NOT on V_rep
    after the solve — otherwise the boundary values of V_rep would be
    contaminated through the elliptic coupling.

    With ``fix_corners=True`` (default), the four corners of the
    rectangular cross-section are pinned via homogeneous Dirichlet BC.
    This eliminates the H¹ semi-norm kernel (the two rigid translations
    of Ω₀); rotations and uniform scaling have ∇V ≠ 0 and are already
    suppressed by the α∫∇V:∇W term itself. The L² term β∫V·W is kept
    for additional coercivity / scale-setting of the bilinear form.
    """
    if fix_corners and (W is None or H is None):
        raise ValueError("fix_corners=True requires W and H.")
    V_2d = shape_grad_cofunction.function_space().dual()
    v    = TrialFunction(V_2d)
    w    = TestFunction(V_2d)

    # ── Hadamard diagnostic ──
    # A "true" shape derivative dJ/dT lives only on the boundary (Hadamard
    # structure theorem). The volumetric AD-computed gradient *should* have
    # zero interior values up to FE noise.
    with stop_annotating():
        boundary_dofs = DirichletBC(V_2d, Constant((0.0, 0.0)),
                                    "on_boundary").nodes
        is_bd = np.zeros(V_2d.dim() // V_2d.value_size, dtype=bool)
        is_bd[boundary_dofs] = True
        g_arr = np.asarray(shape_grad_cofunction.dat.data_ro)
        # Per-node L2 norm: sqrt(g_x^2 + g_y^2)
        node_norm = np.sqrt(np.sum(g_arr ** 2, axis=1))
        bd_l2  = float(np.linalg.norm(node_norm[is_bd]))
        int_l2 = float(np.linalg.norm(node_norm[~is_bd]))
        bd_max  = float(node_norm[is_bd].max()) if is_bd.any() else 0.0
        int_max = float(node_norm[~is_bd].max()) if (~is_bd).any() else 0.0
        n_bd  = int(is_bd.sum())
        n_int = int((~is_bd).sum())
        print(f"  [Hadamard] boundary DOFs: {n_bd}  interior DOFs: {n_int}")
        print(f"  [Hadamard] |g_∂Ω|_L2  = {bd_l2:.4e}  "
              f"max-per-node = {bd_max:.4e}")
        print(f"  [Hadamard] |g_int|_L2 = {int_l2:.4e}  "
              f"max-per-node = {int_max:.4e}")
        if bd_l2 > 0:
            print(f"  [Hadamard] ratio |g_int|/|g_∂Ω|        = {int_l2/bd_l2:.3e}")
            density_int = int_l2 / np.sqrt(max(n_int, 1))
            density_bd  = bd_l2 / np.sqrt(max(n_bd, 1))
            if density_bd > 0:
                print(f"  [Hadamard] per-DOF density int/bd ratio = "
                      f"{density_int/density_bd:.3e}  "
                      f"(≪1 means interior is just FE noise)")

    # ── RHS construction ──
    # mask_interior=True: zero the interior contributions in the cofunction
    # and use the masked version as the linear form on the RHS. Note that
    # we *copy* before mutating so that the original shape_grad cofunction
    # (held by the caller and on the tape) is not modified.
    if mask_interior:
        with stop_annotating():
            shape_grad_bd = shape_grad_cofunction.copy(deepcopy=True)
            shape_grad_bd.dat.data[~is_bd, :] = 0.0
        L_form = shape_grad_bd
        rhs_label = "boundary-only"
    else:
        L_form = shape_grad_cofunction
        rhs_label = "full (boundary + interior)"
    print(f"  [Hadamard] Riesz RHS: {rhs_label}")

    a_form = riesz_metric_form(v, w, metric=metric,
                               alpha=alpha_elast, beta=beta_l2,
                               mu=mu, lam=lam)
    if metric == "elasticity":
        _mu  = alpha_elast if mu  is None else mu
        _lam = alpha_elast if lam is None else lam
        print(f"  [Riesz] ELASTICITY metric: μ={_mu:.3g}  λ={_lam:.3g}  "
              f"β={beta_l2:.3g}")
    else:
        print(f"  [Riesz] H¹ metric: α={alpha_elast:.3g}  β={beta_l2:.3g}")

    V_rep = Function(V_2d, name="V_rep")

    with stop_annotating():
        if fix_corners:
            # ── Identify the 4 corner mesh nodes ──
            X_arr = np.asarray(mesh2d.coordinates.dat.data_ro)
            atol = 1e-6
            is_corner = (
                (np.isclose(X_arr[:, 0], 0.0, atol=atol)
                 | np.isclose(X_arr[:, 0], W, atol=atol))
                & (np.isclose(X_arr[:, 1], 0.0, atol=atol)
                   | np.isclose(X_arr[:, 1], H, atol=atol))
            )
            corner_nodes = np.where(is_corner)[0]
            print(f"  [Riesz] Pinning {len(corner_nodes)} corner nodes "
                  f"at {X_arr[corner_nodes].tolist()}")

            # Assemble system, then zero corner-DOF rows of A (set diag=1)
            # and the corresponding entries in the RHS vector. This is the
            # standard PETSc idiom for imposing homogeneous Dirichlet BC at
            # a sparse set of DOFs without a SubDomain marker.
            A = assemble(a_form)
            b = assemble(L_form)

            # VectorFunctionSpace dim=2 → 2 DOFs per node (component-interlaced)
            n_comp = V_2d.value_size  # = 2
            corner_dofs = np.concatenate([
                n_comp * corner_nodes + c for c in range(n_comp)
            ]).astype(np.int32)

            A.M.handle.zeroRows(corner_dofs, diag=1.0)
            b.dat.data[corner_nodes, :] = 0.0

            solver = LinearSolver(A, solver_parameters={
                "ksp_type": "cg",
                "pc_type":  "hypre",
                "ksp_rtol": 1e-10,
                "ksp_atol": 1e-14,
            })
            solver.solve(V_rep, b)
        else:
            solve(a_form == L_form, V_rep,
                  solver_parameters={
                      "ksp_type": "cg",
                      "pc_type":  "hypre",
                      "ksp_rtol": 1e-10,
                      "ksp_atol": 1e-14,
                  })

        # Diagnostic: V_rep boundary vs interior magnitude. With corner
        # pinning the rigid-body amplification (1/β) is gone, so V_rep
        # is much smaller. Boundary should now be visibly larger than
        # interior (wall-bending pattern instead of near-uniform field).
        v_arr = np.asarray(V_rep.dat.data_ro)
        v_node_norm = np.sqrt(np.sum(v_arr ** 2, axis=1))
        v_bd_max  = float(v_node_norm[is_bd].max())  if is_bd.any() else 0.0
        v_int_max = float(v_node_norm[~is_bd].max()) if (~is_bd).any() else 0.0
        v_mean    = float(np.linalg.norm(v_arr.mean(axis=0)))
        print(f"  [Hadamard] |V_rep| max-per-node: boundary = {v_bd_max:.4e},  "
              f"interior = {v_int_max:.4e}")
        print(f"  [Hadamard] |mean V_rep|_2 = {v_mean:.4e}  "
              f"(≪ V_rep_max means no net translation)")

    return V_rep


def inner_M_form(u, v, riesz_alpha, riesz_beta):
    """M-bilinear pairing between two V_2d Functions (H¹ Riesz metric).

    Mirrors the metric used by ``riesz_representative`` — H¹ Dirichlet
    energy + L² stabilisation — so that diagnostic cosines and norms
    live in the same Hilbert space as V_a itself.
    """
    return float(assemble(
        riesz_alpha  * inner(grad(u), grad(v)) * dx
      + riesz_beta   * inner(u, v) * dx))


def mesh2d_cell_quality(mesh2d):
    """Per-cell shape quality of a 2D simplex mesh, in its CURRENT coordinate
    state.

    Uses the normalised "radius-ratio-like" measure for triangles

        q = 4·√3 · Area / (ℓ₁² + ℓ₂² + ℓ₃²)  ∈ (0, 1],

    with q = 1 for an equilateral triangle and q → 0 as the cell degenerates
    (collapses to a sliver). This is a true *shape* quality — unlike
    min|J|/max|J|, which conflates cell-size variation with distortion (after
    a boundary deformation near-wall cells legitimately shrink, dragging the
    |J| ratio down even when every cell is still well-shaped).

    Returns (q_min, q_mean) over all cells. Cheap, fully vectorised, off-tape.
    """
    with stop_annotating():
        coords = np.asarray(mesh2d.coordinates.dat.data_ro)              # (N, 2)
        cmap   = mesh2d.coordinates.function_space() \
                       .cell_node_map().values                          # (ncells, 3)
        tri = coords[cmap]                                              # (ncells, 3, 2)
        v0, v1, v2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
        a = v1 - v0
        b = v2 - v0
        l2 = (np.sum((v1 - v0) ** 2, axis=1)
              + np.sum((v2 - v1) ** 2, axis=1)
              + np.sum((v0 - v2) ** 2, axis=1))
        area = 0.5 * np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
        l2_safe = np.where(l2 > 0, l2, 1.0)
        q = (4.0 * np.sqrt(3.0)) * area / l2_safe
    return float(q.min()), float(q.mean())


def check_2d_mesh_quality(mesh2d, tol_min_jacobian=0.05, ref_signs=None,
                          tol_quality=None):

    try:
        with stop_annotating():
            V_scalar = FunctionSpace(mesh2d, "DG", 0)
            J_fn     = Function(V_scalar)
            J_fn.interpolate(JacobianDeterminant(mesh2d))
            j_arr = np.asarray(J_fn.dat.data_ro)
        abs_min = float(np.abs(j_arr).min())
        abs_max = float(np.abs(j_arr).max())

        if ref_signs is not None:
            inverted = np.sign(j_arr) != ref_signs
            n_inv = int(np.sum(inverted))
            if n_inv > 0:
                print(f"  [mesh2d check] FAIL: {n_inv} cell(s) flipped sign "
                      f"(min |J| = {abs_min:.3e}, max |J| = {abs_max:.3e})")
                return False

        # Hard quality gate: reject a (non-inverted) mesh whose worst cell
        # shape-quality has dropped below tol_quality. Used by the TR
        # backtrack to keep deformations gentle enough that the Riesz
        # elastic mesh-move stays well-conditioned — the lever that fits a
        # smoother which is already maxed out per step.
        if tol_quality is not None:
            q_min, q_mean = mesh2d_cell_quality(mesh2d)
            if q_min < tol_quality:
                print(f"  [mesh2d check] FAIL: min cell quality "
                      f"q_min = {q_min:.3e} < tol_quality = {tol_quality:.3e} "
                      f"(q_mean = {q_mean:.3e})")
                return False

        if abs_max > 0 and abs_min / abs_max < tol_min_jacobian:
            print(f"  [mesh2d check] WARNING: near-degenerate cell "
                  f"(min |J| / max |J| = {abs_min / abs_max:.3e})")
        return True
    except Exception as e:
        print(f"  [mesh2d check] quality check skipped ({e})")
        return True


def run_shape_optimization(a_target, shared_data, mesh_data_init,
                            bif_result_init,
                            *,
                            r_ref, z_ref, a_ref,
                            max_steps=50, tol_J=1e-8,
                            alpha_step=1.0, alpha_min=1e-8,
                            alpha_backtrack=0.5, max_backtrack=12,
                            Delta_max=1.0, branch_C=1.0,
                            riesz_alpha=1.0, riesz_beta=1e-2,
                            ms_tol=1e-12, ms_max_iter=30,
                            n_grid_2d=128,
                            plot_dir=None):

    R, H, W, L_c, U_c, Re, _, _, _, _ = shared_data

    # Default plot directory: timestamped, so concurrent runs (e.g. one in
    # tmux + one fresh) write to separate folders and don't overwrite each
    # other's images. Pass plot_dir explicitly to override this.
    if plot_dir is None:
        plot_dir = "images/shape_opt_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  [ShapeOpt] Cross-section snapshots → {plot_dir}/")

    mesh2d = RectangleMesh(n_grid_2d, n_grid_2d, W, H, quadrilateral=False, diagonal="crossed", comm=COMM_WORLD)

    V_2d   = VectorFunctionSpace(mesh2d, "CG", 1)
    T_2d   = Function(V_2d, name="T_2d")
    with stop_annotating():
        X_ref_2d = Function(V_2d, name="X_ref_2d")
        X_ref_2d.interpolate(SpatialCoordinate(mesh2d))

    # Here we build a snapshot of all the signs of the jacobi-determinant on the undeformed mesh. RectangleMesh triangles
    # alternate orientation, so half the cells have det(J) < 0; mesh inversion is detected by sign FLIPS, not by det(J) <= 0.
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d)
        V_dg0_ref = FunctionSpace(mesh2d, "DG", 0)
        J_ref     = Function(V_dg0_ref)
        J_ref.interpolate(JacobianDeterminant(mesh2d))
        ref_signs_2d = np.sign(np.asarray(J_ref.dat.data_ro)).copy()

    # Here we get the results from the newton and moore-spence-refinement of the inital guess
    r_bif    = float(bif_result_init['r'])
    z_bif    = float(bif_result_init['z'])
    a_bif    = float(bif_result_init['a'])
    phi_bif  = np.asarray(bif_result_init['phi'], dtype=float)
    l_vec    = phi_bif.copy()

    md          = mesh_data_init
    shared_cur  = shared_data

    print(f"\n{'#'*70}")
    print(f"#  SHAPE OPTIMISATION  (Algorithm 4.1)")
    print(f"#  Target:  a* = {a_target:.4f}")
    print(f"#  Initial: a* = {a_bif:.6f}")
    print(f"{'#'*70}")

    history     = []
    converged   = False

    # Trust-region backtracking, mirroring the MS inner-loop TR
    # (locate_bifurcation_points._globalize_tr).
    #   pred  = α · 2ε²                    (linear-a J-model: -dJ/dα|_{α=0})
    #   actual = J − J_try
    #   ρ      = actual / pred
    #   ρ ≥ eta_accept (0.1): accept; ρ > eta_good (0.4): grow Delta ×2
    #   else: shrink Delta ×0.5, retry
    # alpha_seed plays the role of `Delta` carried across outer steps.
    # alpha_step is the initial trust radius Delta_0; Delta_max is a soft
    # upper cap on growth. A trial is also rejected if the new background-
    # flow state deviates more than branch_C·‖u_new‖ from the previous
    # one (Boullé–Farrell–Paganini eq. 4.3) — guards against MS converging
    # onto a different bifurcation branch when alpha is too aggressive.
    eta_accept = 0.1
    # eta_good lowered from 0.75: under the GN parametrisation the first
    # accepted steps typically have ρ ≈ 0.5, sitting in the "no growth"
    # band (between eta_accept and eta_good) and clamping Delta forever
    # at alpha_step. With eta_good=0.4 the TR auto-promotes Delta on
    # these steady-good steps, and the eta_accept safety net still
    # catches over-aggressive trials.
    eta_good   = 0.4
    alpha_seed = min(float(alpha_step), float(Delta_max))

    # No persistent curvature state. The TR uses pred = α·2ε² as the
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

    # Lag-by-one curvature estimate for the TR quadratic model. After each
    # accepted step the Curv-Diag block reconstructs
    #   Q := ⟨V_a, ∇²a·V_a⟩_L²
    # from (ε_old, ε_new, α). This Q is then used as the curvature in next
    # iteration's pred. Cheap (no extra Hessian-VP), and the snap-and-reset
    # keeps the local a-curvature roughly stable between outer steps so a
    # one-step lag is a reasonable estimator. Step 1 (no history) falls
    # back to the linear-a model.
    Q_eff_prev = None

    # Tracks the (r,z,a) bif at the START of the previous iteration, so we
    # can print the per-step trajectory delta — a leading indicator for
    # regime changes (e.g. dz jumping signals approach of the bif to z=0).
    prev_r_bif = prev_z_bif = prev_a_bif = None

    for step in range(max_steps):

        J = (a_bif - a_target) ** 2

        print(f"\n{'='*65}")
        print(f"  STEP {step+1:3d}  |  a_bif = {a_bif:.8f}  |  J = {J:.6e}")
        if prev_a_bif is not None:
            print(f"  Δ since prev step:  "
                  f"da = {a_bif - prev_a_bif:+.3e}  "
                  f"dz = {z_bif - prev_z_bif:+.3e}  "
                  f"dr = {r_bif - prev_r_bif:+.3e}")
        print(f"{'='*65}")
        prev_r_bif, prev_z_bif, prev_a_bif = float(r_bif), float(z_bif), float(a_bif)

        history.append({
            'step':  step,
            'a_bif': a_bif,
            'J':     J,
            'alpha': None,
            'trials': [],   # per-LS-trial diagnostics: list of dicts
                            # {alpha, conv, F_norm, a_try}. Used to track
                            # MS noise-floor evolution as ALE drift grows.
        })

        # Snapshot the current cross-section. step=0 is the initial
        # rectangular state; step=k>0 is the state after step k-1 was
        # accepted (i.e. each saved image_fix shows the result of one
        # outer-loop shape change).
        if plot_dir is not None:
            save_cross_section_plot(step + 1, mesh2d, X_ref_2d, T_2d,
                                    W, H, a_bif, J, plot_dir)

        if J < tol_J:
            print(f"\n  CONVERGED: J = {J:.4e} < tol = {tol_J:.4e}")
            converged = True
            break

        DG_5x5, F_at_bif, J_sp = compute_DG_at_bif(r_bif, z_bif, a_bif, phi_bif, l_vec, md, r_ref, z_ref, a_ref)

        res_check = np.linalg.norm(F_at_bif)
        print(f"  |F(y*)| = {res_check:.4e}  (should be ~0 at bifurcation)")
        if res_check > 1e-4:
            print("  WARNING: large residual — bifurcation point may have drifted")

        # ── Drift diagnostics: 3D mesh deformation under composed xi ──
        # xi_baseline grows monotonically as each snap absorbs (dr,dz,da).
        # If the 3D mesh starts shearing/inverting, MS noise floor will rise
        # before any algorithmic alarm fires. det(I + ∇xi) < 0 in any cell =
        # mesh inversion. Note: xi_particle is *not* included here — it is
        # rebuilt every MS-iter from the trial (dr,dz,da) and not stored.
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
        print(f"  [Drift] 3D det(I+∇xi) range = [{detF_min:+.3e}, {detF_max:+.3e}]  "
              f"inverted cells = {n_inv}")

        grad_a, lambda_adj = compute_shape_gradient(
            r_bif, z_bif, a_bif, phi_bif, l_vec, a_target, DG_5x5,
            T_2d, mesh2d, X_ref_2d, R, H, W, Re, md, r_ref, z_ref, a_ref)

        # ── Riesz lift of ∇a (H¹ metric) ──
        # Same solver as the legacy V_rep = riesz(2ε·∇a) path, but the RHS
        # is now the ε-free ∇a, so V_a has fixed magnitude across outer
        # steps and the Gauss-Newton step picks up ε only via the explicit
        # (a_bif − a_target) factor in the step-direction construction
        # below.
        V_a = riesz_representative(
            grad_a, mesh2d,
            alpha_elast=riesz_alpha, beta_l2=riesz_beta,
            fix_corners=True, W=W, H=H)

        with stop_annotating():
            g_a_sq_L2 = float(assemble(inner(V_a, V_a) * dx))
            # ‖V_a‖²_M for the H¹ Riesz metric M used by riesz_representative:
            #   M(V,W) = α·∫∇V:∇W + β·∫V·W
            # With V_a = M⁻¹·∇a we have ⟨∇a, V_a⟩ = ‖V_a‖²_M = the
            # directional-derivative coefficient of a along V_a.
            g_a_sq_M = float(assemble(
                riesz_alpha  * inner(grad(V_a), grad(V_a)) * dx
              + riesz_beta   * inner(V_a, V_a) * dx))
        print(f"  ||V_a||_L2 = {g_a_sq_L2 ** 0.5:.4e}   "
              f"||V_a||_M = {g_a_sq_M ** 0.5:.4e}  (H¹ metric)")

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
                                        riesz_alpha, riesz_beta)
                eta = Function(V_2d, name="V_a_step_diff")
                eta.dat.data[:] = (np.asarray(prev_V_a.dat.data_ro)
                                   - np.asarray(V_a.dat.data_ro))
                sn = inner_M_form(prev_V_a, eta,
                                   riesz_alpha, riesz_beta)
                nn = inner_M_form(eta, eta,
                                   riesz_alpha, riesz_beta)
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

        # ── Gauss-Newton step direction ──
        # Linear-a model:  a(T + s·V_a) ≈ a + s·⟨∇a, V_a⟩_L² = a + s·g_a_sq_M.
        # Choosing s = (a_target − a)/g_a_sq_M ≡ −ε/g_a_sq_M lands a at
        # a_target in one shot (α = 1). The TR-BT then scales by α ∈ (0, 1].
        #
        # Step-direction sign note. The Riesz solve here uses the SAME
        # convention as the legacy V_rep = riesz(2ε·∇a) path: with that
        # RHS, T += α·V_rep was empirically descending (J decreased). So
        # V_a = V_rep/(2ε) is descending for objective a too. With ε > 0
        # (a_bif > a_target) we therefore step T += +(ε/g_a_sq_M)·V_a to
        # *reduce* a — matching the legacy descent direction up to scaling.
        eps    = float(a_bif - a_target)
        eps_sq = eps * eps
        gn_scale = eps / g_a_sq_M
        V_rep = Function(V_2d, name="step_GN")
        V_rep.dat.data[:] = gn_scale * np.asarray(V_a.dat.data_ro)

        step_norm_M  = abs(gn_scale) * (g_a_sq_M  ** 0.5)
        step_norm_L2 = abs(gn_scale) * (g_a_sq_L2 ** 0.5)
        print(f"  [GN] α=1 step: ||step||_L2 = {step_norm_L2:.4e}  "
              f"||step||_M = {step_norm_M:.4e}  "
              f"(ε = {eps:+.4e}, ε² = {eps_sq:.4e})")

        # Under the GN parametrisation the linear J-slope is
        #     -dJ/dα |_{α=0} = +2·ε² ,
        # i.e. pred_lin(α) = α·2ε² is the linear-a model's predicted J
        # reduction. We rely on the actual ρ-check after MS to catch
        # any over-aggressive α (no model-based pre-rejection): MS
        # converges cleanly, mesh-quality is monitored, and rho =
        # (J − J_try)/pred_lin self-corrects via TR backtracking.
        grad_norm_sq_M = 2.0 * eps_sq

        # --- Sanity check: is mesh2d in a clean state BEFORE line search? ---
        # If the tape replay corrupted X_ref_2d or mesh2d.coordinates, the
        # line search will see a pre-inverted mesh no matter how small alpha
        # is.  Reset mesh2d to X_ref_2d + T_2d explicitly and verify.
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

            # Step size = trust radius. The descent direction V_rep has no
            # natural "Newton length", so we are always at the trust boundary.
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
                            mesh2d, R, H, W, Re)
            except Exception as e:
                print(f"  [TR-BT bt={bt}] Background flow failed: {e}")
                Delta *= alpha_backtrack
                continue

            shared_try = (R, H, W, L_c, U_c, Re,
                          G_try, U_m_try, u_bar_try, p_bar_try)

            # --- Update mesh_data with new background flow ---
            md_try = dict(md)
            md_try['u_bar_2d']      = u_bar_try
            md_try['p_bar_tilde_2d'] = p_bar_try
            md_try['G']             = G_try
            md_try['U_m']           = U_m_try

            # --- Lift T_2d_try to 3D and store as static deformation ---
            # evaluate_forces (called from moore_spence_solve) reads md['xi_channel']
            # to keep the 3D geometry consistent with u_bar_2d on the deformed
            # cross-section. Build off-tape (constant during the MS iterations).
            #
            # TR-backtrack is off-tape — no AD adjoint to keep consistent.
            # The subsequent MS evaluates forces using LIVE mesh2d.coords;
            # MS@reference empirically gives the correct descent rho whereas
            # MS@deformed (after restore) flips the rho sign. So:
            # restore=False — mesh2d stays at reference after the call,
            # matching the agg.txt manual-swap pattern. The next outer
            # iteration's pre-LS check re-assigns mesh2d.coords regardless.
            with stop_annotating():
                with xi_channel_ref_swap(X_ref_2d, restore=False):
                    xi_ch_try = build_xi_channel_from_T2d(
                        T_2d_try, md['mesh3d'], md['X_ref'], mesh2d,
                        md['R'], md['W'], md['H'])
                xi_ch_static = Function(md['V_def'], name="xi_channel_static")
                xi_ch_static.dat.data[:] = xi_ch_try.dat.data_ro
            md_try['xi_channel'] = xi_ch_static

            # --- Re-solve Moore-Spence on deformed geometry ---
            print(f"  [TR-BT bt={bt}] alpha = {alpha:.3e} → running Moore-Spence...")
            try:
                with stop_annotating():
                    r_try, z_try, a_try, phi_try, conv_try, F_norm_try, \
                        G_norm_try, stalled_try = moore_spence_solve(
                            r_bif, z_bif, a_bif, shared_try,
                            tol=ms_tol, max_iter=ms_max_iter,
                            md=md_try,
                            dr_init=float(r_bif - r_ref),
                            dz_init=float(z_bif - z_ref),
                            da_init=float(a_bif - a_ref))
            except Exception as e:
                print(f"  [TR-BT bt={bt}] Moore-Spence failed: {e}")
                history[-1]['trials'].append({
                    'alpha': alpha, 'conv': False,
                    'F_norm': float('nan'), 'a_try': float('nan'),
                    'note': f'exception: {e}',
                })
                Delta *= alpha_backtrack
                continue

            # STALL with |F| at machine eps == reached FE resolution of the
            # bifurcation (residual is all in J·phi at the FE-noise floor);
            # not a sign that alpha was too aggressive. Gate by |F| so a
            # genuinely unconverged MS (F stuck non-zero) still shrinks.
            ms_floor_F = 1e-10
            if (not conv_try) and stalled_try \
                    and float(F_norm_try) < ms_floor_F:
                print(f"  [TR-BT bt={bt}] MS stalled at FE-noise floor "
                      f"(|F|={F_norm_try:.2e} < {ms_floor_F:.0e}, "
                      f"|G|={G_norm_try:.2e}); accepting as converged-to-"
                      f"discretisation-limit.")
                conv_try = True

            # Diagnostic: MS-end |F| per trial. For converged trials it's
            # ~machine precision; for stalled trials it's the FE noise-floor
            # plateau. Watching this across steps tells us whether the
            # plateau is rising with cumulative ALE drift (snap-and-reset
            # would help) or stays roughly constant (drift not the cause).
            history[-1]['trials'].append({
                'alpha':  alpha,
                'conv':   bool(conv_try),
                'F_norm': float(F_norm_try),
                'a_try':  float(a_try),
            })

            if not conv_try:
                print(f"  [TR-BT bt={bt}] Moore-Spence did not converge.")
                Delta *= alpha_backtrack
                continue

            # --- Branch-tracking check (Boullé–Farrell–Paganini eq. 4.3) ---
            # Reject if the background-flow state moved more than branch_C·‖u_new‖
            # under this shape update. The moving-mesh / fixed-connectivity setup
            # makes the diffeomorphism pullback trivial in DOF space, so a
            # straight node-DOF subtraction is the correct ‖u^k∘(I+δT)^{-1} − u^{k+1}‖.
            # Without this guard, large α can drag MS onto a different branch
            # silently — the next iteration then optimizes the wrong target.
            try:
                u_old_arr = np.asarray(shared_cur[8].dat.data_ro, dtype=float)
                u_new_arr = np.asarray(u_bar_try.dat.data_ro,     dtype=float)
                if u_old_arr.shape == u_new_arr.shape:
                    delta_u = float(np.linalg.norm(u_new_arr - u_old_arr))
                    base_u  = float(np.linalg.norm(u_new_arr))
                    branch_ratio = delta_u / max(base_u, 1e-30)
                    print(f"  [Branch] ‖u_new − u_old‖/‖u_new‖ = {branch_ratio:.3e}  "
                          f"(C = {branch_C:.2g})")
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

            # Quadratic-a TR model. Under
            #   a(α) ≈ a_old − α·ε + ½·α²·ε²·Q/g⁴_M ,  Q = ⟨V_a,∇²a·V_a⟩_L²
            # the J=(a−a*)² reduction along the GN step is
            #   pred = α·2ε² − α²·ε²·(1 + ε·Q/g⁴_M).
            # Q comes from the previous accepted step (lag-by-one). Step 1
            # has Q_eff_prev=None → linear fallback. If the quadratic term
            # makes pred ≤ 0 the model says no reduction is reachable along
            # V_rep — force backtrack regardless of J_try.
            actual_red = float(J) - float(J_try)
            if Q_eff_prev is not None and np.isfinite(Q_eff_prev):
                k_dim = eps * Q_eff_prev / (g_a_sq_M ** 2)
                pred = (alpha * grad_norm_sq_M
                        - alpha * alpha * eps_sq * (1.0 + k_dim))
                pred_label = f"quad(k={k_dim:+.3f})"
            else:
                k_dim = float('nan')
                pred = alpha * grad_norm_sq_M
                pred_label = "lin(init)"
            if pred > 1e-30:
                rho = actual_red / pred
            elif pred < -1e-30:
                # Quadratic model predicts ascent → reject without trusting
                # an "accidental" J_try < J that could come from FE noise.
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
            print(f"  [TR-BT bt={bt}] a_try = {a_try:.6f},  J_try = {J_try:.4e}  "
                  f"(vs J = {J:.4e})  rho = {rho:+.4f}  "
                  f"[pred_{pred_label}={pred:.2e}, H_obs={H_obs:.2e}]")

            if rho >= eta_accept:
                # Accept step
                print(f"  [TR-BT] Step ACCEPTED (bt={bt}, alpha={alpha:.4e}, "
                      f"rho={rho:+.4f})")
                T_2d.assign(T_2d_try)
                r_bif   = float(r_try)
                z_bif   = float(z_try)
                a_bif   = float(a_try)
                phi_bif = np.asarray(phi_try, dtype=float)
                l_vec   = phi_bif.copy()
                md          = md_try
                shared_cur  = shared_try
                history[-1]['alpha'] = alpha
                history[-1]['rho']   = rho
                accepted = True
                accepted_rho = rho
                break
            else:
                Delta *= alpha_backtrack

        if not accepted:
            print(f"  [ShapeOpt] TR backtracking failed — stopping.")
            break

        # Standard TR seed update: grow on good rho, hold otherwise.
        # No curvature-based cap — if the doubled seed is too aggressive,
        # the next step's TR backtracks it; if it's just right, we save
        # MS solves by not artificially shrinking it.
        prev_seed = alpha_seed
        if accepted_rho > eta_good:
            alpha_seed = min(2.0 * Delta, float(Delta_max))
        else:
            alpha_seed = min(Delta, float(Delta_max))  # accept but don't grow
        if alpha_seed != prev_seed:
            print(f"  [TR-BT] Delta_next: {prev_seed:.3e} → {alpha_seed:.3e}  "
                  f"(rho={accepted_rho:+.4f})")

        # ── ALE-drift diagnostic: per-step plateau |F| of failed trials ──
        # If this rises monotonically with the cumulative drift |r_bif - r_ref|
        # etc., the snap-and-reset of the ALE basis is justified. If it stays
        # roughly constant, ALE drift is not the dominant noise source.
        trials = history[-1]['trials']
        failed_F = [t['F_norm'] for t in trials if not t['conv']]
        accepted_F = [t['F_norm'] for t in trials if t['conv']]
        drift_dz  = float(z_bif - z_ref)
        drift_dr  = float(r_bif - r_ref)
        drift_da  = float(a_bif - a_ref)
        if failed_F:
            failed_str = ", ".join(f"{x:.2e}" for x in failed_F)
        else:
            failed_str = "—"
        accepted_str = f"{accepted_F[-1]:.2e}" if accepted_F else "—"
        print(f"\n  [DIAG step {step+1}] drift (dr,dz,da) from r_ref to new bif: "
              f"({drift_dr:+.2e}, {drift_dz:+.2e}, {drift_da:+.2e})")
        print(f"  [DIAG step {step+1}] plateau |F| of failed trials: [{failed_str}]"
              f"  | accepted |F|: {accepted_str}")

        # ── Curvature diagnostic (cheap, no extra MS-solve) ──
        # The GN step is T += α·(ε/g_a²)·V_a. Under the linear-a + Taylor-
        # quadratic model:
        #     a(T+step) = a_old − α·ε + ½·α²·(ε/g_a²)²·⟨V_a, ∇²a·V_a⟩_L²
        # Hence  ε_new = (1−α)·ε + ½·α²·ε²·Q/g_a⁴  with  Q := ⟨V_a, ∇²a·V_a⟩.
        # Solve for Q from the *observed* (ε_old, ε_new, α, g_a²) of this step
        # to get the curvature that actually governed the residual.
        #
        # - Q stable over outer steps  ⇒  ε_new ∝ ε²  ⇒  quadratic convergence.
        # - Q growing as ε ↘            ⇒  bifurcation parameter has higher
        #     local curvature near the target shape — *physical* slowdown.
        # - Q/g_a² is the dominant M-metric eigenvalue along V_a; per-step
        #     Δa-ceiling in ANY descent direction is bounded by g_a²/(2·λ_dom).
        eps_old = float(eps)
        eps_new = float(a_bif - a_target)
        ratio   = (eps_new / eps_old) if abs(eps_old) > 1e-30 else float('nan')
        if abs(alpha) > 1e-12 and eps_sq > 1e-30:
            Q_eff = (2.0 * (g_a_sq_M ** 2)
                     * (eps_new - (1.0 - alpha) * eps_old)
                     / (alpha * alpha * eps_sq))
        else:
            Q_eff = float('nan')
        lam_dom_est = (Q_eff / g_a_sq_M) if g_a_sq_M > 1e-30 else float('nan')
        print(f"  [Curv-Diag] ε: {eps_old:+.4e} → {eps_new:+.4e}   "
              f"ratio = {ratio:+.4f}   (quad-conv ⇔ ratio ∝ ε)")
        print(f"  [Curv-Diag] Q_eff = ⟨V_a, ∇²a·V_a⟩_L² ≈ {Q_eff:.4e}   "
              f"λ_dom_est = Q_eff/g_a² = {lam_dom_est:.4e}")
        if np.isfinite(Q_eff):
            Q_eff_prev = Q_eff
        target_eps = float(tol_J) ** 0.5
        if 0 < ratio < 1 and abs(eps_new) > target_eps:
            steps_remaining = float(np.log(target_eps / abs(eps_new))
                                    / np.log(ratio))
            print(f"  [Curv-Diag] If ratio holds: ~{steps_remaining:.1f} more "
                  f"steps to ε < √tol_J = {target_eps:.2e}")

        # ── Snap-and-reset: re-linearise the ALE basis around the new bif ──
        # Absorbs (dr, dz, da) into md['xi_baseline'], updates the spherical-
        # hole centre and reference radius, and re-solves basis_r/z/a at the
        # new rest state. The next compute_DG_at_bif and MS calls then start
        # from (dr,dz,da)=0 with a freshly linearised basis, keeping the FE
        # noise floor low even after many shape-opt steps.
        print(f"  [Snap] Re-solving ALE basis around new bif point...")
        reset_ale_basis_for_step(md, drift_dr, drift_dz, drift_da)
        r_ref, z_ref, a_ref = float(r_bif), float(z_bif), float(a_bif)

        # Restore 2D mesh to accepted T_2d
        with stop_annotating():
            mesh2d.coordinates.assign(X_ref_2d + T_2d)

        # Snapshot V_a for the [Diag] cosine/Rayleigh diagnostic at the
        # next outer step. V_a here was computed at the *start* of this
        # step (before T_2d was modified); pairing it against the V_a
        # recomputed at the start of the next step reveals whether the
        # descent direction is locked or rotating.
        with stop_annotating():
            prev_V_a = V_a.copy(deepcopy=True)
            prev_g_a_sq_M = g_a_sq_M

    # Final objective
    J_final = (float(a_bif) - float(a_target)) ** 2

    print(f"\n{'#'*70}")
    print(f"#  SHAPE OPTIMISATION FINISHED")
    print(f"#  Steps run:    {step + 1}")
    print(f"#  a_bif_final:  {a_bif:.8f}")
    print(f"#  J_final:      {J_final:.6e}")
    print(f"#  Converged:    {converged}")
    print(f"{'#'*70}")

    # Restore 2D mesh to final T_2d (just in case)
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)

    # Final snapshot: captures the post-last-accepted state, which the
    # in-loop snapshot only takes at the *start* of an iteration.
    if plot_dir is not None:
        save_cross_section_plot(step + 2, mesh2d, X_ref_2d, T_2d,
                                W, H, a_bif, J_final, plot_dir)

    return {
        'T_2d':      T_2d,
        'mesh2d':    mesh2d,
        'X_ref_2d':  X_ref_2d,
        'a_bif':     a_bif,
        'r_bif':     r_bif,
        'z_bif':     z_bif,
        'phi_bif':   phi_bif,
        'J_final':   J_final,
        'history':   history,
        'converged': converged,
        'shared_data_final': shared_cur,
        'mesh_data_final':   md,
    }


def save_cross_section_plot(step, mesh2d, X_ref_2d, T_2d, W, H,
                             a_bif, J, output_dir, exaggeration=10.0):
    """Save a two-panel PNG visualisation of the current deformed cross-section.

    Left panel: TRUE-SCALE. Reference rectangle dashed grey, deformed
    boundary solid blue, mesh nodes as grey scatter. Lets you check
    whether the mesh has tangled or whether T_2d has run away into
    unphysical magnitudes. For sane runs ‖T_2d‖ ≪ 1, so the deformed
    boundary lies essentially on top of the reference.

    Right panel: EXAGGERATED by a factor (default 10×). Same content
    but the deformation amplitude is multiplied. Makes the *pattern*
    of the wall deformation visible (wall bulges, asymmetries, etc.).
    The title labels this panel as exaggerated; numbers along the axes
    no longer represent physical positions.
    """
    import os
    import matplotlib
    matplotlib.use("Agg")              # headless-safe (server-side runs)
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    with stop_annotating():
        X_ref_arr = np.asarray(X_ref_2d.dat.data_ro)
        T_arr     = np.asarray(T_2d.dat.data_ro)

        # ── Diagnostic: prove boundary vs interior are actually moving ──
        # If max boundary displacement ≪ max interior displacement, then
        # we're really only re-parameterising the interior (= visualisation
        # is right, the boundary genuinely is barely moving).
        # If they're comparable, the boundary IS moving but the polyline
        # rendering hides sub-pixel motion.
        atol_bd = 1e-9
        on_bd = (
            np.isclose(X_ref_arr[:, 0], 0.0,   atol=atol_bd)
            | np.isclose(X_ref_arr[:, 0], W, atol=atol_bd)
            | np.isclose(X_ref_arr[:, 1], 0.0,   atol=atol_bd)
            | np.isclose(X_ref_arr[:, 1], H, atol=atol_bd)
        )
        per_node = np.sqrt(np.sum(T_arr ** 2, axis=1))
        bd_max  = float(per_node[on_bd].max())  if on_bd.any() else 0.0
        int_max = float(per_node[~on_bd].max()) if (~on_bd).any() else 0.0
        bd_mean = float(per_node[on_bd].mean()) if on_bd.any() else 0.0
        int_mean = float(per_node[~on_bd].mean()) if (~on_bd).any() else 0.0
        print(f"  [Plot] T_2d displacement: "
              f"boundary max = {bd_max:.3e} (mean {bd_mean:.3e}),  "
              f"interior max = {int_max:.3e} (mean {int_mean:.3e})")

    # Boundaries: true and exaggerated.
    boundary_true = extract_deformed_boundary(mesh2d, X_ref_2d, T_2d,
                                              W, H)

    # Fixed exaggeration factor across all steps so the images are
    # directly comparable: same multiplier on every plot lets the eye
    # read the actual growth of T_2d between steps.
    effective_exag = exaggeration
    T_arr_exag = effective_exag * T_arr
    with stop_annotating():
        T_2d_exag = Function(T_2d.function_space())
        T_2d_exag.dat.data[:] = T_arr_exag
    boundary_exag = extract_deformed_boundary(mesh2d, X_ref_2d, T_2d_exag,
                                              W, H)

    boundary_true_closed = np.vstack([boundary_true, boundary_true[:1]])
    boundary_exag_closed = np.vstack([boundary_exag, boundary_exag[:1]])

    X_def_true = X_ref_arr + T_arr
    X_def_exag = X_ref_arr + T_arr_exag

    # Shift display coords so the axes are centered on the channel
    # mid-plane: x, z ∈ [-W/2, W/2] × [-H/2, H/2] rather than the
    # mesh-frame [0, W] × [0, H]. Pure visualization — underlying
    # mesh data is untouched.
    shift = np.array([0.5 * W, 0.5 * H])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ref_box = np.array([[0, 0], [W, 0], [W, H],
                        [0, H], [0, 0]]) - shift

    # ── Left: true-scale ──
    ax1.plot(ref_box[:, 0], ref_box[:, 1],
             "--", color="0.5", linewidth=1.2, label="Reference rectangle")
    ax1.scatter(X_def_true[:, 0] - shift[0], X_def_true[:, 1] - shift[1],
                s=2, color="0.7", alpha=0.5, label="Mesh nodes")
    ax1.plot(boundary_true_closed[:, 0] - shift[0],
             boundary_true_closed[:, 1] - shift[1],
             "-", color="C0", linewidth=2.0, label="Deformed boundary")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$z$")
    ax1.set_title("True scale")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    margin1 = max(0.05 * W, 0.05 * H)
    ax1.set_xlim(-0.5 * W - margin1, 0.5 * W + margin1)
    ax1.set_ylim(-0.5 * H - margin1, 0.5 * H + margin1)

    # ── Right: exaggerated ──
    ax2.plot(ref_box[:, 0], ref_box[:, 1],
             "--", color="0.5", linewidth=1.2, label="Reference rectangle")
    ax2.scatter(X_def_exag[:, 0] - shift[0], X_def_exag[:, 1] - shift[1],
                s=2, color="0.7", alpha=0.5, label="Mesh nodes")
    ax2.plot(boundary_exag_closed[:, 0] - shift[0],
             boundary_exag_closed[:, 1] - shift[1],
             "-", color="C1", linewidth=2.0, label="Deformed boundary")
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$z$")
    ax2.set_title(f"Exaggerated × {effective_exag:.0f}")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    # Auto-extend the limits if the exaggerated boundary blows past the reference
    span_x = max(abs(boundary_exag[:, 0].min()),
                 abs(boundary_exag[:, 0].max() - W))
    span_y = max(abs(boundary_exag[:, 1].min()),
                 abs(boundary_exag[:, 1].max() - H))
    margin2 = max(0.05 * W, 0.05 * H, 1.1 * span_x, 1.1 * span_y)
    ax2.set_xlim(-0.5 * W - margin2, 0.5 * W + margin2)
    ax2.set_ylim(-0.5 * H - margin2, 0.5 * H + margin2)

    # Per-step header
    T_norm = float(np.linalg.norm(T_arr))
    fig.suptitle(f"Step {step}:  $a_{{\\mathrm{{bif}}}}$ = {a_bif:.6f},  "
                 f"$J$ = {J:.3e},  $\\|T_{{2d}}\\|_2$ = {T_norm:.3e}",
                 fontsize=12)

    out_path = os.path.join(output_dir, f"step_{step:03d}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved {out_path}")


def extract_deformed_boundary(mesh2d, X_ref_2d, T_2d, W, H, atol=1e-9):

    X_ref = np.asarray(X_ref_2d.dat.data_ro)                 # (N, 2)
    X_def = X_ref + np.asarray(T_2d.dat.data_ro)             # (N, 2)

    on_bottom = np.isclose(X_ref[:, 1], 0.0,   atol=atol)
    on_top    = np.isclose(X_ref[:, 1], H, atol=atol)
    on_left   = np.isclose(X_ref[:, 0], 0.0,   atol=atol)
    on_right  = np.isclose(X_ref[:, 0], W, atol=atol)

    # Walk CCW starting from (0, 0): bottom (L→R), right (B→T),
    # top (R→L), left (T→B).  Drop corners already included in previous edge.
    bottom_idx = np.where(on_bottom)[0]
    bottom_idx = bottom_idx[np.argsort(X_ref[bottom_idx, 0])]

    right_idx  = np.where(on_right)[0]
    right_idx  = right_idx[np.argsort(X_ref[right_idx, 1])]

    top_idx    = np.where(on_top)[0]
    top_idx    = top_idx[np.argsort(-X_ref[top_idx, 0])]

    left_idx   = np.where(on_left)[0]
    left_idx   = left_idx[np.argsort(-X_ref[left_idx, 1])]

    # Concatenate, dropping the shared corner node at each junction.
    ordered_idx = np.concatenate([
        bottom_idx,
        right_idx[1:],
        top_idx[1:],
        left_idx[1:-1],
    ])

    return X_def[ordered_idx]


def save_optimized_section(result, *, R_phys, H_phys, W_phys, Q_phys, rho_phys, mu_phys,
                           R, H, W, L_c, U_c, Re,
                           a_target, n_grid_2d, filepath):
    """Pickle the optimized cross-section so another script can rebuild the
    deformed 2D mesh, recompute the background flow on it, and hand the
    deformed cross-section to the 3D geometry builder.

    The pickle contains everything needed for verification:

      - ``boundary_pts_2d`` : (N, 2) ordered CCW polyline in (s, t) hat coords.
      - ``T_2d_data``       : (Nnodes, 2) deformation values on the reference
                              CG1 VectorFunctionSpace (matches the DOF order
                              of a freshly-constructed RectangleMesh of the
                              same resolution on COMM_SELF).
      - ``n_grid_2d``       : mesh resolution used during optimization.
      - Dimensional + non-dimensional geometry / flow parameters.
      - Bifurcation-point info (r_bif, z_bif, a_bif, phi_bif).

    Parameters
    ----------
    result : dict
        Output of ``run_shape_optimization``.
    filepath : str or Path
        Where to write the pickle.
    """
    mesh2d   = result['mesh2d']
    X_ref_2d = result['X_ref_2d']
    T_2d     = result['T_2d']

    boundary_pts_2d = extract_deformed_boundary(
        mesh2d, X_ref_2d, T_2d, W, H)

    with stop_annotating():
        T_data     = np.asarray(T_2d.dat.data_ro).copy()
        X_ref_data = np.asarray(X_ref_2d.dat.data_ro).copy()

    data = {
        # --- Deformed cross-section ---
        'boundary_pts_2d': boundary_pts_2d,
        'T_2d_data':       T_data,
        'X_ref_2d_data':   X_ref_data,
        'n_grid_2d':       int(n_grid_2d),
        # --- Geometry + flow parameters (dim) ---
        'R_phys': R_phys, 'H_phys': H_phys, 'W_phys': W_phys,
        'Q_phys': Q_phys, 'rho_phys': rho_phys, 'mu_phys': mu_phys,
        # --- Non-dimensional ---
        'R': R, 'H': H, 'W': W,
        'L_c':   L_c,   'U_c':   U_c,   'Re':    Re,
        # --- Optimization result ---
        'a_target':  a_target,
        'a_bif':     float(result['a_bif']),
        'r_bif':     float(result['r_bif']),
        'z_bif':     float(result['z_bif']),
        'phi_bif':   np.asarray(result['phi_bif'], dtype=float),
        'J_final':   float(result['J_final']),
        'converged': bool(result['converged']),
    }

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    print(f"  Saved optimized section to {filepath}")

    return data


def run_from_main(r0, z0, a0, a_target, max_steps=50, tol_J=1e-8,
                  alpha_step=1.0, riesz_alpha=1.0, riesz_beta=1e-2,
                  ms_tol=1e-12, ms_max_iter=15, n_grid_2d=128):

    with stop_annotating():
        bg = background_flow_differentiable(R, H, W, Re)
        G, U_m, u_bar_2d, p_bar_tilde = bg.solve_2D_background_flow()

    shared_data = (R, H, W, L_c, U_c, Re, G, U_m, u_bar_2d, p_bar_tilde)

    r_eq, z_eq, md0, dr0, dz0 = newton_root_refine(r0, z0, a0, shared_data, tol=1e-10, max_iter=15)

    r_ref, z_ref, a_ref = r0, z0, a0

    with stop_annotating():
        r_bif, z_bif, a_bif, phi_bif, conv_ms, F_norm0, _G0, _stalled0 = \
            moore_spence_solve(
                r_eq, z_eq, a0, shared_data,
                tol=ms_tol, max_iter=ms_max_iter,
                md=md0, dr_init=dr0, dz_init=dz0)
        print(f"  Initial bifurcation residual: |F| = {F_norm0:.3e}")

    if not conv_ms:
        raise RuntimeError("Moore-Spence did not converge for initial domain")

    bif_init = {'r': r_bif, 'z': z_bif, 'a': a_bif,
                'phi': phi_bif, 'converged': True}

    print(f"\n  Initial bifurcation: a = {a_bif:.6f} (target = {a_target:.4f})")

    result = run_shape_optimization(
        a_target, shared_data, md0, bif_init,
        r_ref=r_ref, z_ref=z_ref, a_ref=a_ref,
        max_steps=max_steps, tol_J=tol_J,
        alpha_step=alpha_step,
        riesz_alpha=riesz_alpha, riesz_beta=riesz_beta,
        ms_tol=ms_tol, ms_max_iter=ms_max_iter,
        n_grid_2d=n_grid_2d)

    T_final  = result['T_2d']
    out_path = "output_shape_T2d.h5"
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
            filepath="output_optimized_section.pkl",
        )
    except Exception as e:
        print(f"  Warning: could not save optimized-section pickle: {e}")

    return result


if __name__ == "__main__":

    # a_target imported from problem_setup (= a_target_phys / L_c).

    # Initial guess from the bifurcation diagramm
    r_off_init = 0.6098
    z_off_init = 0.0
    a_init = 0.1375

    print("\nparticle_maxh_rel = ", particle_maxh_rel)
    print("global_maxh_rel = ", global_maxh_rel)

    result = run_from_main(
        a_target=a_target,
        r0=r_off_init,
        z0=z_off_init,
        a0=a_init,
        max_steps=100,
        tol_J=1e-8,
        # α=1 is the *natural* full Newton step under the GN parametrisation
        # (T += α·ε/g_a²·V_a; α=1 lands at a_target in the linear-a model).
        # The TR's ρ-check + mesh-ok + branch-tracking catch over-aggressive
        # trials; max_backtrack=12 halvings reach down to ~1.2e-4 if needed.
        alpha_step=1.0,
        # ── safer fallback if step 1 burns too many backtracks ──
        # alpha_step=1e-2,
        riesz_alpha=0.1,
        riesz_beta=1e-2,
        ms_tol=1e-12,
        ms_max_iter=30,
        n_grid_2d=120
    )
    
    print("\nFinal result:")
    print(f"  a_bif = {result['a_bif']:.8f}")
    print(f"  J     = {result['J_final']:.6e}")
    print(f"  steps = {len(result['history'])}")
