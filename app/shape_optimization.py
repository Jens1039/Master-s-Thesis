import os
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
from datetime import datetime
from firedrake import *
import gc
from firedrake.adjoint import stop_annotating, annotate_tape, continue_annotation, get_working_tape, set_working_tape, Tape, ReducedFunctional, Control

from background_flow_differentiable import background_flow_differentiable, build_3d_background_flow_differentiable, vom_transfer, _compute_inv_perm
from perturbed_flow_differentiable import perturbed_flow_differentiable, _build_xi_hat, check_mesh_quality, evaluate_forces, reset_ale_basis_for_step
from locate_bifurcation_points import newton_root_refine, moore_spence_solve
from config_paper_parameters import *
from nondimensionalization import nondimensionalisation


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


def _solve_bg_on_mesh(mesh2d, R_hat, H_hat, W_hat, Re_float):

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re_float,
                                        mesh2d=mesh2d)
    return bg.solve_2D_background_flow()


def build_xi_channel_from_T2d(T_2d, mesh3d, X_ref, mesh2d, R_hat, W_hat, H_hat):

    V_def = X_ref.function_space()

    # --- Step 1: VOM query points from REFERENCE 3D positions (off tape) ---
    with stop_annotating():
        X_ref_arr = X_ref.dat.data_ro.copy()          # (N, 3)
        rho_ref   = np.sqrt(X_ref_arr[:, 0]**2 + X_ref_arr[:, 1]**2)

        s_ref = rho_ref - R_hat + 0.5 * W_hat
        t_ref = X_ref_arr[:, 2] + 0.5 * H_hat

        query_pts = np.column_stack([
            np.clip(s_ref, 0.0, W_hat),
            np.clip(t_ref, 0.0, H_hat),
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


def eval_forces_with_bg_on_tape(r_off, z_off, a_hat, u_bar_2d, p_bar_2d, G_hat_val, mesh_data, r_ref, z_ref, a_ref,
                                    T_2d=None, mesh2d=None, rz_on_tape=False):

    md     = mesh_data
    mesh3d = md['mesh3d']
    dr_fn_ctrl = dz_fn_ctrl = None

    if rz_on_tape:
        # Particle position ON TAPE (for mixed Hessian d²F/(dT dr) etc.)
        R_space  = FunctionSpace(mesh3d, "R", 0)
        dr_fn_ctrl = Function(R_space, name="delta_r").assign(float(r_off - r_ref))
        dz_fn_ctrl = Function(R_space, name="delta_z").assign(float(z_off - z_ref))
        da_fn      = Function(R_space, name="delta_a").assign(float(a_hat  - a_ref))
        xi_particle = _build_xi_hat(dr_fn_ctrl, dz_fn_ctrl, da_fn, md)
    else:
        # Particle position OFF TAPE (r/z/a are fixed constants)
        with stop_annotating():
            R_space  = FunctionSpace(mesh3d, "R", 0)
            dr_fn    = Function(R_space).assign(float(r_off - r_ref))
            dz_fn    = Function(R_space).assign(float(z_off - z_ref))
            da_fn    = Function(R_space).assign(float(a_hat  - a_ref))
            xi_particle = _build_xi_hat(dr_fn, dz_fn, da_fn, md)

    if T_2d is not None and mesh2d is not None:
        # --- Channel wall deformation from T_2d (ON TAPE) ---
        xi_channel = build_xi_channel_from_T2d(
            T_2d, mesh3d, md['X_ref'], mesh2d,
            md['R_hat'], md['W_hat'], md['H_hat'])

        if rz_on_tape:
            # Both xi_channel and xi_particle are on tape
            xi_sum = xi_channel + xi_particle
        else:
            # xi_particle is constant (off tape), gradient flows only
            # through xi_channel → T_2d.
            with stop_annotating():
                xi_particle_fn = Function(md['V_def'], name="xi_particle_const")
                xi_particle_fn.dat.data[:] = xi_particle.dat.data_ro
            xi_sum = xi_channel + xi_particle_fn   # AddBlock on tape

        # Materialize the full deformation as a Function in V_def.
        # md['xi_baseline'] (snap from past accepted steps) is added
        # off-tape — it does not depend on the current Controls.
        xi_total = Function(md['V_def'], name="xi_total")
        xi_total.assign(md['xi_baseline'] + xi_sum)

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

    # --- 3D background flow with u_bar_2d ON TAPE (VOM / xi=None path) ---
    # The VOM query points are computed from the CURRENT (deformed) mesh
    # coordinates, which is physically correct: we evaluate u_bar_2d at
    # the positions of the deformed 3D nodes.
    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], G_hat_val,
        mesh3d, md['tags'], u_bar_2d, p_bar_2d,
        X_ref=None, xi=None)

    # --- Perturbed-flow Stokes solve (CachedStokesSolveBlock on tape) ---
    pf = perturbed_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['L_hat'],
        float(a_hat), md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi_total, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    if rz_on_tape:
        return F_p_x, F_p_z, dr_fn_ctrl, dz_fn_ctrl
    return F_p_x, F_p_z


def compute_shape_gradient(r_bif, z_bif, a_bif, phi_bif, l_vec, a_target, DG_5x5, T_2d, mesh2d, X_ref_2d,
                            R_hat, H_hat, W_hat, Re_float, mesh_data, r_ref, z_ref, a_ref):

    # del J/del y
    rhs_adj = np.array([0., 0., 2. * float(a_bif - a_target), 0., 0.])
    cond_DG = np.linalg.cond(DG_5x5)
    print(f"  [Shape] cond(DG) = {cond_DG:.3e}")

    # get lagrange multiplier
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

    G_hat_val, _, u_bar_2d_new, p_bar_2d_new = _solve_bg_on_mesh(mesh2d, R_hat, H_hat, W_hat, Re_float)

    print(f"  [Shape] G_hat = {G_hat_val:.6e}")

    print(f"  [Shape] Evaluating forces at y* = "f"(r={r_bif:.4f}, z={z_bif:.4f}, a={a_bif:.4f})...")
    F_p_x, F_p_z, dr_fn, dz_fn = eval_forces_with_bg_on_tape(r_bif, z_bif, a_bif, u_bar_2d_new, p_bar_2d_new, G_hat_val,
                                                mesh_data, r_ref, z_ref, a_ref, T_2d=T_2d, mesh2d=mesh2d, rz_on_tape=True)

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
        xi_restore = _build_xi_hat(dr_r, dz_r, da_r, mesh_data)
        mesh3d_r.coordinates.assign(
            mesh_data['X_ref'] + mesh_data['xi_baseline']
            + xi_restore + mesh_data['xi_channel'])

    return shape_grad, lambda_adj


def project_z_symmetric(V_rep, mesh2d, H_hat):
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
        mirror_coords[:, 1] = H_hat - coords[:, 1]
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


def riesz_representative(shape_grad_cofunction, mesh2d,
                          alpha_elast=1.0, beta_l2=1e-2,
                          mu_cr=100.0,
                          mask_interior=True,
                          fix_corners=True, W_hat=None, H_hat=None):
    """Compute the Cauchy-Riemann Riesz representative of the shape gradient.

    Inner product (Paganini/Wechsung/Farrell 2018, Eq. 44, citing
    Iglesias/Sturm/Wechsung 2017):

        (V, W) = α ∫ ∇V:∇W dx + μ_cr ∫ (BV)·(BW) dx + β ∫ V·W dx

    where B is the 2D Cauchy-Riemann operator
        BV = (∂_x V_x - ∂_z V_z, ∂_z V_x + ∂_x V_z).

    BV = 0 iff V is conformal (holomorphic) — i.e. an angle-preserving
    map. Non-conformal modes (shear, anisotropic scaling) get an
    additional μ_cr-penalty on top of the H¹ Dirichlet energy, so the
    descent direction is biased toward near-conformal wall-bending
    rather than shear/scaling modes that the plain elasticity metric
    treats equally.

    With ``mask_interior=True`` (default), the interior DOFs of the
    shape-gradient cofunction are zeroed *before* the Riesz solve
    (Hadamard projection). Zeroing happens on the RHS, NOT on V_rep
    after the solve — otherwise the boundary values of V_rep would be
    contaminated through the elliptic coupling.

    With ``fix_corners=True`` (default), the four corners of the
    rectangular cross-section are pinned via homogeneous Dirichlet BC.
    This eliminates the rigid-body null space (2 translations + 1
    rotation) AND the uniform-scaling mode. Combined with the CR
    penalty on shear, only genuine wall-bending modes remain
    energetically cheap.
    """
    if fix_corners and (W_hat is None or H_hat is None):
        raise ValueError("fix_corners=True requires W_hat and H_hat.")
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

    # ── Cauchy-Riemann inner product ──
    # BV = (∂_x V_x - ∂_z V_z, ∂_z V_x + ∂_x V_z); BV = 0 iff V is conformal.
    # μ_cr ∫ |BV|² penalises shear / non-conformal modes; α ∫ |∇V|² is the
    # standard H¹ smoothing; β ∫ |V|² stabilises any residual near-null modes
    # (mostly redundant once corners are pinned).
    def B_cr(p):
        return as_vector([
            p[0].dx(0) - p[1].dx(1),
            p[0].dx(1) + p[1].dx(0),
        ])

    a_form = (alpha_elast * inner(grad(v), grad(w)) * dx
              + mu_cr      * inner(B_cr(v), B_cr(w)) * dx
              + beta_l2    * inner(v, w) * dx)
    print(f"  [Riesz] CR metric: α={alpha_elast:.3g}  "
          f"μ_cr={mu_cr:.3g}  β={beta_l2:.3g}")

    V_rep = Function(V_2d, name="V_rep")

    with stop_annotating():
        if fix_corners:
            # ── Identify the 4 corner mesh nodes ──
            X_arr = np.asarray(mesh2d.coordinates.dat.data_ro)
            atol = 1e-6
            is_corner = (
                (np.isclose(X_arr[:, 0], 0.0, atol=atol)
                 | np.isclose(X_arr[:, 0], W_hat, atol=atol))
                & (np.isclose(X_arr[:, 1], 0.0, atol=atol)
                   | np.isclose(X_arr[:, 1], H_hat, atol=atol))
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

        # ── Conformality diagnostic ──
        # ‖BV‖²/‖∇V‖²: 0 = fully conformal (V is an angle-preserving map,
        # i.e. holomorphic), >0 = has non-conformal shear/anisotropic
        # components. With μ_cr large enough, this ratio should be small;
        # if it's still O(1), the shear modes still dominate V_rep and we
        # need to raise μ_cr (or rethink).
        grad_sq = float(assemble(inner(grad(V_rep), grad(V_rep)) * dx))
        cr_sq   = float(assemble(inner(B_cr(V_rep), B_cr(V_rep)) * dx))
        if grad_sq > 0:
            conformality = cr_sq / grad_sq
            print(f"  [CR] ‖BV_rep‖² / ‖∇V_rep‖² = {conformality:.4e}  "
                  f"(0 = conformal, O(1) = shear-dominated)")

    return V_rep


def check_2d_mesh_quality(mesh2d, tol_min_jacobian=0.05, ref_signs=None):

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
                            alpha_step=0.1, alpha_min=1e-8,
                            alpha_backtrack=0.5, max_backtrack=12,
                            Delta_max=1e-1, branch_C=1.0,
                            riesz_alpha=1.0, riesz_beta=1e-2,
                            riesz_mu_cr=100.0,
                            enforce_z_symmetry=False,
                            ms_tol=1e-12, ms_max_iter=30,
                            n_grid_2d=128,
                            plot_dir=None):

    R_hat, H_hat, W_hat, L_c, U_c, Re, _, _, _, _ = shared_data

    # Default plot directory: timestamped, so concurrent runs (e.g. one in
    # tmux + one fresh) write to separate folders and don't overwrite each
    # other's images. Pass plot_dir explicitly to override this.
    if plot_dir is None:
        plot_dir = "images/shape_opt_run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"  [ShapeOpt] Cross-section snapshots → {plot_dir}/")

    mesh2d = RectangleMesh(n_grid_2d, n_grid_2d, W_hat, H_hat, quadrilateral=False, diagonal="crossed", comm=COMM_WORLD)

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
    print(f"#  Target:  a_hat* = {a_target:.4f}")
    print(f"#  Initial: a_hat* = {a_bif:.6f}")
    print(f"{'#'*70}")

    history     = []
    converged   = False

    # Trust-region backtracking, mirroring the MS inner-loop TR
    # (locate_bifurcation_points._globalize_tr).
    #   pred  = α · ‖V_rep‖²_L²            (linear J-model)
    #   actual = J − J_try
    #   ρ      = actual / pred
    #   ρ ≥ eta_accept (0.1): accept; ρ > eta_good (0.75): grow Delta ×2
    #   else: shrink Delta ×0.5, retry
    # alpha_seed plays the role of `Delta` carried across outer steps.
    # alpha_step is the initial trust radius Delta_0; Delta_max is a soft
    # upper cap on growth (rho > 0.75 doubles but is then clamped). A trial
    # is also rejected if the new background-flow state deviates more than
    # branch_C·‖u_new‖ from the previous one (Boullé–Farrell–Paganini eq.
    # 4.3) — guards against MS converging onto a different bifurcation
    # branch when alpha is too aggressive.
    eta_accept = 0.1
    eta_good   = 0.75
    alpha_seed = min(float(alpha_step), float(Delta_max))

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
                                    W_hat, H_hat, a_bif, J, plot_dir)

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

        shape_grad, lambda_adj = compute_shape_gradient(r_bif, z_bif, a_bif, phi_bif, l_vec, a_target, DG_5x5,
                                            T_2d, mesh2d, X_ref_2d, R_hat, H_hat, W_hat, Re, md, r_ref, z_ref, a_ref)

        V_rep = riesz_representative(shape_grad, mesh2d,
                                      alpha_elast=riesz_alpha, beta_l2=riesz_beta,
                                      mu_cr=riesz_mu_cr,
                                      fix_corners=True, W_hat=W_hat, H_hat=H_hat)

        # Optional: project V_rep onto the z-symmetric subspace. Kills any
        # antisymmetric component that would drag the +z Bif toward z=0.
        if enforce_z_symmetry:
            V_rep, max_anti, mirror_dist = project_z_symmetric(
                V_rep, mesh2d, H_hat)
            print(f"  [Symm] z-symmetric projection: "
                  f"max antisym killed = {max_anti:.3e}  "
                  f"max mirror-lookup dist = {mirror_dist:.2e}")

        with stop_annotating():
            grad_norm_sq_L2 = float(assemble(inner(V_rep, V_rep) * dx))
            # ‖V_rep‖²_M for the CR-Riesz metric M used by riesz_representative:
            #   M(V,W) = α·∫∇V:∇W + μ_cr·∫BV·BW + β·∫V·W
            # With V_rep = M⁻¹·shape_grad we have ⟨shape_grad, V_rep⟩ = ‖V_rep‖²_M,
            # i.e. this is the exact first-order predicted J-decrease per unit α.
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
                            mesh2d, R_hat, H_hat, W_hat, Re)
            except Exception as e:
                print(f"  [TR-BT bt={bt}] Background flow failed: {e}")
                Delta *= alpha_backtrack
                continue

            shared_try = (R_hat, H_hat, W_hat, L_c, U_c, Re,
                          G_try, U_m_try, u_bar_try, p_bar_try)

            # --- Update mesh_data with new background flow ---
            md_try = dict(md)
            md_try['u_bar_2d_hat']      = u_bar_try
            md_try['p_bar_tilde_2d_hat'] = p_bar_try
            md_try['G_hat']             = G_try
            md_try['U_m_hat']           = U_m_try

            # --- Lift T_2d_try to 3D and store as static deformation ---
            # evaluate_forces (called from moore_spence_solve) reads md['xi_channel']
            # to keep the 3D geometry consistent with u_bar_2d_hat on the deformed
            # cross-section. Build off-tape (constant during the MS iterations).
            with stop_annotating():
                xi_ch_try = build_xi_channel_from_T2d(
                    T_2d_try, md['mesh3d'], md['X_ref'], mesh2d,
                    md['R_hat'], md['W_hat'], md['H_hat'])
                xi_ch_static = Function(md['V_def'], name="xi_channel_static")
                xi_ch_static.dat.data[:] = xi_ch_try.dat.data_ro
            md_try['xi_channel'] = xi_ch_static

            # --- Re-solve Moore-Spence on deformed geometry ---
            print(f"  [TR-BT bt={bt}] alpha = {alpha:.3e} → running Moore-Spence...")
            try:
                with stop_annotating():
                    r_try, z_try, a_try, phi_try, conv_try, F_norm_try = \
                        moore_spence_solve(
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

            # Trust-region acceptance via ρ = actual / predicted.
            #   predicted reduction:  α · ‖V_rep‖²_M   (linear J-model in the
            #                         CR-Riesz metric M used to compute V_rep —
            #                         exact directional-derivative coefficient)
            #   actual    reduction:  J − J_try
            pred = alpha * grad_norm_sq_M
            if pred > 1e-30:
                rho = (J - J_try) / pred
            else:
                rho = -1.0 if J_try > J else 1.0
            print(f"  [TR-BT bt={bt}] a_try = {a_try:.6f},  J_try = {J_try:.4e}  "
                  f"(vs J = {J:.4e})  rho = {rho:+.4f}")

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

        # Update trust radius for the next step (carried via alpha_seed).
        # Soft cap at Delta_max — protects against runaway growth on a string
        # of good ρ values; the branch-tracking check is the primary safety net.
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
                                W_hat, H_hat, a_bif, J_final, plot_dir)

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


def save_cross_section_plot(step, mesh2d, X_ref_2d, T_2d, W_hat, H_hat,
                             a_bif, J, output_dir, exaggeration=1000.0):
    """Save a two-panel PNG visualisation of the current deformed cross-section.

    Left panel: TRUE-SCALE. Reference rectangle dashed grey, deformed
    boundary solid blue, mesh nodes as grey scatter. Lets you check
    whether the mesh has tangled or whether T_2d has run away into
    unphysical magnitudes. For sane runs ‖T_2d‖ ≪ 1, so the deformed
    boundary lies essentially on top of the reference.

    Right panel: EXAGGERATED by a factor (default 1000×). Same content
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
            | np.isclose(X_ref_arr[:, 0], W_hat, atol=atol_bd)
            | np.isclose(X_ref_arr[:, 1], 0.0,   atol=atol_bd)
            | np.isclose(X_ref_arr[:, 1], H_hat, atol=atol_bd)
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
                                              W_hat, H_hat)

    # Fixed exaggeration factor across all steps so the images are
    # directly comparable: same multiplier on every plot lets the eye
    # read the actual growth of T_2d between steps.
    effective_exag = exaggeration
    T_arr_exag = effective_exag * T_arr
    with stop_annotating():
        T_2d_exag = Function(T_2d.function_space())
        T_2d_exag.dat.data[:] = T_arr_exag
    boundary_exag = extract_deformed_boundary(mesh2d, X_ref_2d, T_2d_exag,
                                              W_hat, H_hat)

    boundary_true_closed = np.vstack([boundary_true, boundary_true[:1]])
    boundary_exag_closed = np.vstack([boundary_exag, boundary_exag[:1]])

    X_def_true = X_ref_arr + T_arr
    X_def_exag = X_ref_arr + T_arr_exag

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ref_box = np.array([[0, 0], [W_hat, 0], [W_hat, H_hat],
                        [0, H_hat], [0, 0]])

    # ── Left: true-scale ──
    ax1.plot(ref_box[:, 0], ref_box[:, 1],
             "--", color="0.5", linewidth=1.2, label="Reference rectangle")
    ax1.scatter(X_def_true[:, 0], X_def_true[:, 1],
                s=2, color="0.7", alpha=0.5, label="Mesh nodes")
    ax1.plot(boundary_true_closed[:, 0], boundary_true_closed[:, 1],
             "-", color="C0", linewidth=2.0, label="Deformed boundary")
    ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x$")
    ax1.set_ylabel(r"$z$")
    ax1.set_title("True scale")
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    margin1 = max(0.05 * W_hat, 0.05 * H_hat)
    ax1.set_xlim(-margin1, W_hat + margin1)
    ax1.set_ylim(-margin1, H_hat + margin1)

    # ── Right: exaggerated ──
    ax2.plot(ref_box[:, 0], ref_box[:, 1],
             "--", color="0.5", linewidth=1.2, label="Reference rectangle")
    ax2.scatter(X_def_exag[:, 0], X_def_exag[:, 1],
                s=2, color="0.7", alpha=0.5, label="Mesh nodes")
    ax2.plot(boundary_exag_closed[:, 0], boundary_exag_closed[:, 1],
             "-", color="C1", linewidth=2.0, label="Deformed boundary")
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$z$")
    ax2.set_title(f"Exaggerated × {effective_exag:.0f}")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    # Auto-extend the limits if the exaggerated boundary blows past the reference
    span_x = max(abs(boundary_exag[:, 0].min()),
                 abs(boundary_exag[:, 0].max() - W_hat))
    span_y = max(abs(boundary_exag[:, 1].min()),
                 abs(boundary_exag[:, 1].max() - H_hat))
    margin2 = max(0.05 * W_hat, 0.05 * H_hat, 1.1 * span_x, 1.1 * span_y)
    ax2.set_xlim(-margin2, W_hat + margin2)
    ax2.set_ylim(-margin2, H_hat + margin2)

    # Per-step header
    T_norm = float(np.linalg.norm(T_arr))
    fig.suptitle(f"Step {step}:  $a_{{\\mathrm{{bif}}}}$ = {a_bif:.6f},  "
                 f"$J$ = {J:.3e},  $\\|T_{{2d}}\\|_2$ = {T_norm:.3e}",
                 fontsize=12)

    out_path = os.path.join(output_dir, f"step_{step:03d}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved {out_path}")


def extract_deformed_boundary(mesh2d, X_ref_2d, T_2d, W_hat, H_hat, atol=1e-9):

    X_ref = np.asarray(X_ref_2d.dat.data_ro)                 # (N, 2)
    X_def = X_ref + np.asarray(T_2d.dat.data_ro)             # (N, 2)

    on_bottom = np.isclose(X_ref[:, 1], 0.0,   atol=atol)
    on_top    = np.isclose(X_ref[:, 1], H_hat, atol=atol)
    on_left   = np.isclose(X_ref[:, 0], 0.0,   atol=atol)
    on_right  = np.isclose(X_ref[:, 0], W_hat, atol=atol)

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


def save_optimized_section(result, *, R, H, W, Q, rho, mu,
                           R_hat, H_hat, W_hat, L_c, U_c, Re,
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
        mesh2d, X_ref_2d, T_2d, W_hat, H_hat)

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
        'R': R, 'H': H, 'W': W,
        'Q': Q, 'rho': rho, 'mu': mu,
        # --- Non-dimensional ---
        'R_hat': R_hat, 'H_hat': H_hat, 'W_hat': W_hat,
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
                  alpha_step=0.05, riesz_alpha=1.0, riesz_beta=1e-2,
                  riesz_mu_cr=100.0,
                  enforce_z_symmetry=False,
                  ms_tol=1e-12, ms_max_iter=15, n_grid_2d=128):

    R_hat, H_hat, W_hat, a_hat, L_c, U_c, Re = nondimensionalisation(R, H, W, a, Q, rho, mu, print_values=True)

    with stop_annotating():
        bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
        G_hat, U_m_hat, u_bar_2d, p_bar_tilde = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d, p_bar_tilde)

    r_eq, z_eq, md0, dr0, dz0 = newton_root_refine(r0, z0, a0, shared_data, tol=1e-10, max_iter=15)

    r_ref, z_ref, a_ref = r0, z0, a0

    with stop_annotating():
        r_bif, z_bif, a_bif, phi_bif, conv_ms, F_norm0 = moore_spence_solve(
            r_eq, z_eq, a0, shared_data,
            tol=ms_tol, max_iter=ms_max_iter,
            md=md0, dr_init=dr0, dz_init=dz0)
        print(f"  Initial bifurcation residual: |F| = {F_norm0:.3e}")

    if not conv_ms:
        raise RuntimeError("Moore-Spence did not converge for initial domain")

    bif_init = {'r': r_bif, 'z': z_bif, 'a': a_bif,
                'phi': phi_bif, 'converged': True}

    print(f"\n  Initial bifurcation: a_hat = {a_bif:.6f} (target = {a_target:.4f})")

    result = run_shape_optimization(
        a_target, shared_data, md0, bif_init,
        r_ref=r_ref, z_ref=z_ref, a_ref=a_ref,
        max_steps=max_steps, tol_J=tol_J,
        alpha_step=alpha_step,
        riesz_alpha=riesz_alpha, riesz_beta=riesz_beta,
        riesz_mu_cr=riesz_mu_cr,
        enforce_z_symmetry=enforce_z_symmetry,
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
            R=R, H=H, W=W, Q=Q, rho=rho, mu=mu,
            R_hat=R_hat, H_hat=H_hat, W_hat=W_hat,
            L_c=L_c, U_c=U_c, Re=Re,
            a_target=a_target,
            n_grid_2d=n_grid_2d,
            filepath="output_optimized_section.pkl",
        )
    except Exception as e:
        print(f"  Warning: could not save optimized-section pickle: {e}")

    return result


if __name__ == "__main__":

    a_target = 0.1

    # Initial guess from the bifurcation diagramm
    r_off_hat_init = 0.6098
    z_off_hat_init = 0.0
    a_hat_init = 0.1375

    print("\nparticle_maxh_rel = ", particle_maxh_rel)
    print("global_maxh_rel = ", global_maxh_rel)

    result = run_from_main(
        a_target=a_target,
        r0=r_off_hat_init,
        z0=z_off_hat_init,
        a0=a_hat_init,
        max_steps=50,
        tol_J=1e-8,
        # alpha_step: previously held at 1e-3 because α=1e-2 had MS plateau
        # at the FE noise floor — but that was before CR + corner-fix. With
        # rigid-body amplification gone (corner BCs) and V_rep ~98% conformal
        # (CR penalty), the starting |G| in each LS trial should now be clean
        # enough for Newton to drive to machine precision even at α=1e-2.
        # If MS plateaus or stalls reappear, the stall detector + Armijo
        # backtracking will catch it. Per-step ‖T_2d‖_max ≈ 1.5e-4 ≪ 1
        # so linearization should still hold.
        alpha_step=1e-2,
        # H¹ smoothing strength. With the CR penalty doing most of the
        # mode-selection now, α=10 is overkill — keep it moderate.
        riesz_alpha=10.0,
        riesz_beta=1e-2,
        # Cauchy-Riemann penalty (Paganini/Wechsung/Farrell 2018). Penalises
        # non-conformal modes (shear, anisotropic scaling) preferentially,
        # biases descent toward conformal wall-bending. Effective shear
        # penalty ≈ (1 + μ_cr/α) × baseline. 100 / 10 = 10× extra cost for
        # shear modes.
        riesz_mu_cr=100.0,
        # Constrain V_rep to z-symmetric deformations about z=H/2.
        # Antisymmetric components would drag the +z Bif toward z=0 and
        # eventually trigger coalescence with the -z mirror Bif (numerical
        # degeneracy). With this on, dz_Bif/d_step is 0 by construction and
        # the Bif moves only in (r,a). Flip to False for an asymmetric run.
        enforce_z_symmetry=False,
        ms_tol=1e-12,
        ms_max_iter=30,
        n_grid_2d=120
    )
    
    print("\nFinal result:")
    print(f"  a_bif = {result['a_bif']:.8f}")
    print(f"  J     = {result['J_final']:.6e}")
    print(f"  steps = {len(result['history'])}")
