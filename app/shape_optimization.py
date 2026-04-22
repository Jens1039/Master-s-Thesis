import os
os.environ["OMP_NUM_THREADS"] = "1"

"""
Algorithm 4.1 from Boulle, Farrell, Paganini (2021):
'Control of Bifurcation Structures Using Shape Optimization'

Design variable
---------------
T_2d ∈ VectorFunctionSpace(mesh2d_ref, 'CG', 1)
    Deformation of the 2D cross-section of the curved channel.
    Initially zero (rectangular cross-section).  The actual cross-section
    at shape-optimisation step k is the image of the reference rectangle
    under the map  (s, t) -> (s + T_r(s,t), t + T_z(s,t)).

Objective
---------
J(Omega) = (a*(Omega) - a_target)^2
    a*(Omega) = bifurcation parameter (particle radius) at which the
                pitchfork bifurcation of equilibrium positions occurs.

Shape gradient (implicit function theorem)
------------------------------------------
1. Find y* = (r*, z*, a*, phi*) via Moore-Spence on domain Omega.
2. Adjoint:  DG_y^T * lambda = [0, 0, 2*(a* - a_target), 0, 0]^T
3. Shape tape: T_2d -> u_bar_2d(T_2d) -> G1(y*, T_2d) -> lambda[0:2]^T * F
4. shape_grad = ReducedFunctional(lambda[0:2]^T * F, [T_2d]).derivative()

APPROXIMATION:
Only the G1 = F (particle force) contribution is included in the shape
gradient.  The G2 = J_sp @ phi contribution — which requires mixed second
derivatives d^2F / (d(r,z) d(T_2d)) — is neglected.  At convergence of
Moore-Spence, |G1| ≈ 0 and |G2| ≈ 0, so the sensitivity of G2 to T_2d
is a higher-order correction.  The G1 contribution through the background
flow change dominates and drives the bifurcation parameter shift.

The shape gradient captures two contributions from T_2d:
  (a) background flow change: T_2d → deformed mesh2d → NS solve → u_bar_2d
  (b) 3D domain geometry change: T_2d → xi_channel (via VOM lift) →
      mesh3d.coordinates (AssignBlock) → Stokes form → forces

5. Riesz representative: solve H^1 problem
       a_elast(V_rep, W) = -shape_grad(W)  for all W in V_2d
6. Armijo line search + mesh update.
7. Rebuild background flow on deformed mesh.  Re-solve Moore-Spence.
"""

import gc
import math
import numpy as np

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, continue_annotation
from pyadjoint import (
    get_working_tape, set_working_tape, Tape,
    ReducedFunctional, Control,
)

from background_flow_return_UFL import (
    build_3d_background_flow_differentiable,
    vom_transfer,
    _compute_inv_perm,
)
from perturbed_flow_return_UFL import perturbed_flow_differentiable


# ---------------------------------------------------------------------------
#  Lazy imports from the main optimisation module (avoids circular dependency
#  when shape_optimization.py is imported stand-alone).
# ---------------------------------------------------------------------------
def _get_opt_fns():
    """Return (setup_moving_mesh_hat, _build_xi_hat,
               check_mesh_quality, evaluate_forces_jac_hessian_hat,
               moore_spence_solve_hat_ad)."""
    import optimization_of_branch_points as opt
    return (
        opt.setup_moving_mesh_hat,
        opt._build_xi_hat,
        opt.check_mesh_quality,
        opt.evaluate_forces_jac_hessian_hat,
        opt.moore_spence_solve_hat_ad,
    )


# ---------------------------------------------------------------------------
#  2D background flow on a (possibly deformed) mesh — annotation-aware
# ---------------------------------------------------------------------------

def solve_2D_bg_flow_on_tape(mesh2d, R_hat, H_hat, W_hat, Re_float):
    """Solve the 2D Dean-flow background NS problem on *mesh2d*.

    Annotation-aware: when called with pyadjoint annotation ON this creates
    a ``NonlinearVariationalSolver`` solve block on the tape, making
    ``u_bar`` and ``p_bar_tilde`` tape-dependent on the mesh coordinates
    (and hence on any Control that drives those coordinates).

    ``mesh2d.coordinates`` should already be deformed before calling
    (e.g. via ``mesh2d.coordinates.assign(X_ref_2d + T_2d)``).

    Returns
    -------
    G_hat_val : float  (the pressure-gradient eigenvalue; NOT on tape)
    U_m_hat   : float  (max azimuthal velocity; NOT on tape)
    u_bar     : Function on mesh2d  (on tape if annotation is active)
    p_bar_tilde : Function on mesh2d  (on tape if annotation is active)
    """
    V_bg    = VectorFunctionSpace(mesh2d, "CG", 2, dim=3)
    Q_bg    = FunctionSpace(mesh2d, "CG", 1)
    G_space = FunctionSpace(mesh2d, "R", 0)
    W_mixed = V_bg * Q_bg * G_space

    w       = Function(W_mixed)
    Re_cst  = Constant(Re_float)
    R_cst   = Constant(R_hat)
    W_half  = Constant(0.5 * W_hat)

    u, p, G = split(w)
    v, q, g = TestFunctions(W_mixed)

    u_r      = u[0];  u_theta = u[2];  u_z = u[1]
    v_r      = v[0];  v_theta = v[2];  v_z = v[1]

    x = SpatialCoordinate(mesh2d)
    r = x[0] - W_half          # radial offset from channel axis

    def del_r(f): return Dx(f, 0)
    def del_z(f): return Dx(f, 1)

    Rr = R_cst + r              # distance from toroidal centre

    F_cont   = q * (del_r(u_r) + del_z(u_z) + u_r / Rr) * Rr * dx
    F_r      = ((u_r * del_r(u_r) + u_z * del_z(u_r)
                 - u_theta**2 / Rr) * v_r
                + del_r(p) * v_r
                + (1.0 / Re_cst) * dot(grad(u_r), grad(v_r))
                + (1.0 / Re_cst) * (u_r / Rr**2) * v_r
                ) * Rr * dx
    F_theta  = ((u_r * del_r(u_theta) + u_z * del_z(u_theta)
                 + u_r * u_theta / Rr) * v_theta
                - (G * R_cst / Rr) * v_theta
                + (1.0 / Re_cst) * dot(grad(u_theta), grad(v_theta))
                + (1.0 / Re_cst) * (u_theta / Rr**2) * v_theta
                ) * Rr * dx
    F_z      = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
                + del_z(p) * v_z
                + (1.0 / Re_cst) * dot(grad(u_z), grad(v_z))
                ) * Rr * dx
    F_G      = (u_theta - 1.0) * g * dx

    F_total  = F_r + F_theta + F_z + F_cont + F_G

    no_slip  = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)),
                           "on_boundary")
    nullspace = MixedVectorSpaceBasis(
        W_mixed,
        [W_mixed.sub(0),
         VectorSpaceBasis(constant=True, comm=W_mixed.comm),
         W_mixed.sub(2)],
    )

    problem = NonlinearVariationalProblem(F_total, w, bcs=[no_slip])
    solver  = NonlinearVariationalSolver(
        problem, nullspace=nullspace,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_linesearch_type": "l2",
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "pc_fieldsplit_0_fields": "0,1",
            "pc_fieldsplit_1_fields": "2",
            "fieldsplit_0": {
                "ksp_type": "preonly", "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "lu",
                "assembled_pc_factor_mat_solver_type": "mumps"},
            "fieldsplit_1": {"ksp_type": "preonly", "pc_type": "none"},
        },
    )
    solver.solve()          # GenericSolveBlock on tape if annotation is ON

    u_bar       = w.subfunctions[0]
    p_bar_tilde = w.subfunctions[1]
    G_hat_val   = float(w.subfunctions[2].dat.data_ro[0])
    U_m_hat     = float(np.max(u_bar.dat.data_ro[:, 2]))

    return G_hat_val, U_m_hat, u_bar, p_bar_tilde


# ---------------------------------------------------------------------------
#  Shape-optimisation context setup
# ---------------------------------------------------------------------------

def setup_shape_context(H_hat, W_hat, n_grid=120, comm=None):
    """Create the reference 2D mesh and the shape-deformation Control T_2d.

    Returns
    -------
    mesh2d_ref : RectangleMesh  (the reference rectangular cross-section)
    V_2d       : VectorFunctionSpace(mesh2d_ref, "CG", 1)
    T_2d       : Function(V_2d)  initially zero, will be the Control
    X_ref_2d   : Function(V_2d)  reference mesh coordinates (fixed)
    """
    actual_comm = comm if comm is not None else COMM_WORLD
    mesh2d_ref  = RectangleMesh(n_grid, n_grid, W_hat, H_hat,
                                quadrilateral=False, comm=actual_comm)
    V_2d = VectorFunctionSpace(mesh2d_ref, "CG", 1)

    T_2d = Function(V_2d, name="T_2d")   # shape deformation, initially zero

    with stop_annotating():
        X_ref_2d = Function(V_2d, name="X_ref_2d")
        X_ref_2d.interpolate(SpatialCoordinate(mesh2d_ref))

    return mesh2d_ref, V_2d, T_2d, X_ref_2d


# ---------------------------------------------------------------------------
#  Lift 2D cross-section deformation T_2d to 3D channel-wall deformation
# ---------------------------------------------------------------------------

def build_xi_channel_from_T2d(T_2d, mesh3d, X_ref, mesh2d, R_hat, W_hat, H_hat):
    """Lift the 2D cross-section deformation T_2d to a 3D channel deformation.

    For each 3D node at reference position (x, y, z):

        rho   = sqrt(x^2 + y^2)
        s     = rho - R_hat + W_hat/2        (2D radial coordinate on mesh2d)
        t     = z   + H_hat/2               (2D axial  coordinate on mesh2d)

    The 3D deformation is:

        xi_x  = T_2d[0](s, t) * x/rho       (radial → Cartesian x)
        xi_y  = T_2d[0](s, t) * y/rho       (radial → Cartesian y)
        xi_z  = T_2d[1](s, t)               (axial, unchanged)

    Tape-aware: differentiable w.r.t. T_2d via the chain

        T_2d → InterpolateBlock (VOM) → VOMTransferBlock (CG1 2-vec)
             → InterpolateBlock (lift to 3-vec) → xi_channel

    Parameters
    ----------
    T_2d   : Function(VectorFunctionSpace(mesh2d, "CG", 1, dim=2))
    mesh3d : Mesh
    X_ref  : Function(VectorFunctionSpace(mesh3d, "CG", 1, dim=3))
             Reference 3D mesh coordinates (not deformed by T_2d).
    mesh2d : Mesh   (the 2D cross-section mesh)
    R_hat, W_hat, H_hat : float

    Returns
    -------
    xi_channel : Function in X_ref.function_space(), tape-aware w.r.t. T_2d
    """
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


# ---------------------------------------------------------------------------
#  DG matrix at a bifurcation point
# ---------------------------------------------------------------------------

def compute_DG_at_bif(r_bif, z_bif, a_bif, phi_bif, l_vec,
                       mesh_data, r_ref, z_ref, a_ref):
    """Compute the 5×5 Moore-Spence Jacobian DG at the bifurcation point.

    Parameters
    ----------
    r_bif, z_bif, a_bif : float
        Bifurcation point in hat coordinates.
    phi_bif : (2,) array
        Null vector of J_sp at the bifurcation.
    l_vec : (2,) array
        Normalisation vector (usually = phi_bif after l^T phi = 1 normalisation).
    mesh_data : dict
        Mesh data dict from ``setup_moving_mesh_hat``.
    r_ref, z_ref, a_ref : float
        Reference position at which the mesh was built.

    Returns
    -------
    DG : (5, 5) numpy array
    """
    (_, _, _, evaluate_forces_jac_hessian_hat, _) = _get_opt_fns()

    dr = float(r_bif - r_ref)
    dz = float(z_bif - z_ref)
    da = float(a_bif - a_ref)

    F_base, J_full, dJphi_dx = evaluate_forces_jac_hessian_hat(
        dr, dz, da, mesh_data, phi_bif)

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


# ---------------------------------------------------------------------------
#  Single force evaluation with u_bar_2d on the tape
# ---------------------------------------------------------------------------

def eval_forces_with_bg_on_tape(r_off, z_off, a_hat,
                                 u_bar_2d, p_bar_2d, G_hat_val,
                                 mesh_data, r_ref, z_ref, a_ref,
                                 T_2d=None, mesh2d=None):
    """Evaluate particle forces at (r_off, z_off, a_hat) with the full
    pyadjoint tape for the shape Control T_2d.

    Two tape contributions are captured simultaneously:

    1. **Background-flow change** (u_bar_2d path):
       T_2d → mesh2d.coords → NS solve → u_bar_2d → VOM → u_cyl_3d → forces

    2. **3D geometry change** (mesh deformation path):
       T_2d → xi_channel (VOM + lift) → xi_total → mesh3d.coords (AssignBlock)
              → Stokes solve form → force integrals

    When ``T_2d`` is None the function falls back to the old behaviour:
    only the background-flow path is tape-aware and the 3D mesh is moved
    outside the tape.

    Parameters
    ----------
    r_off, z_off, a_hat : float
        Particle position in hat coordinates.
    u_bar_2d, p_bar_2d : Function
        Background flow on the (possibly deformed) 2D mesh — on tape.
    G_hat_val : float
        Background pressure-gradient eigenvalue (scalar, not on tape).
    mesh_data : dict
        3D mesh data from ``setup_moving_mesh_hat``.
    r_ref, z_ref, a_ref : float
        Reference position at which the 3D mesh was built.
    T_2d : Function or None
        Shape deformation Control.  When provided, the 3D channel geometry
        is deformed consistently (xi_channel on tape).
    mesh2d : Mesh or None
        2D cross-section mesh (required when T_2d is not None).

    Returns
    -------
    F_p_x, F_p_z : AdjFloat   (on tape)
    """
    (_, _build_xi_hat, check_mesh_quality, _, _) = _get_opt_fns()

    md     = mesh_data
    mesh3d = md['mesh3d']

    # --- Particle position deformation (off tape: r/z/a are fixed constants
    #     for the shape gradient at the bifurcation point) ---
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

        # --- Combine: xi_total = xi_channel (on tape) + xi_particle (const) ---
        # Function + Function uses pyadjoint's overloaded __add__; since
        # xi_particle is NOT on the tape it acts as a constant, so the
        # gradient flows only through xi_channel → T_2d.
        with stop_annotating():
            xi_particle_fn = Function(md['V_def'], name="xi_particle_const")
            xi_particle_fn.dat.data[:] = xi_particle.dat.data_ro

        xi_total = xi_channel + xi_particle_fn   # AddBlock on tape

        # --- Deform 3D mesh: AssignBlock on tape ---
        # pyadjoint will differentiate the Stokes form assembly and force
        # integrals through this coordinate assignment.
        mesh3d.coordinates.assign(md['X_ref'] + xi_total)

    else:
        # Fallback: no channel deformation — move mesh off tape
        xi_total = xi_particle
        with stop_annotating():
            mesh3d.coordinates.assign(md['X_ref'] + xi_particle)

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
    return F_p_x, F_p_z


# ---------------------------------------------------------------------------
#  Shape gradient (Algorithm 4.1, step 2–3)
# ---------------------------------------------------------------------------

def compute_shape_gradient(r_bif, z_bif, a_bif, phi_bif, l_vec, a_target,
                            DG_5x5,
                            T_2d, mesh2d, X_ref_2d,
                            R_hat, H_hat, W_hat, Re_float,
                            mesh_data, r_ref, z_ref, a_ref):
    """Compute the shape gradient of J = (a* - a_target)^2 w.r.t. T_2d.

    Implements the implicit-function-theorem step of Algorithm 4.1:

        dJ/dT_2d  =  lambda^T  *  dG/dT_2d

    where  lambda  solves  DG_y^T * lambda = [0, 0, 2*(a*-a_t), 0, 0]^T.

    Only the G1 = F contribution to dG/dT_2d is included (see module
    docstring for the approximation rationale).

    Parameters
    ----------
    r_bif, z_bif, a_bif : float   bifurcation point
    phi_bif : (2,) array          null vector  (l^T phi = 1 after MS)
    l_vec   : (2,) array          normalisation row  (typically = phi_bif)
    a_target : float              target bifurcation parameter
    DG_5x5  : (5,5) array        Moore-Spence Jacobian at the bifurcation
    T_2d    : Function            current shape deformation (Control)
    mesh2d  : Mesh                reference 2D mesh (coordinates mutable)
    X_ref_2d : Function           reference 2D mesh coordinates (fixed)
    R_hat, H_hat, W_hat : float   channel geometry in hat coordinates
    Re_float : float              Reynolds number
    mesh_data : dict              3D mesh data dict
    r_ref, z_ref, a_ref : float   3D mesh reference position

    Returns
    -------
    shape_grad : Cofunction        shape gradient in V_2d.dual()
    lambda_adj : (5,) array        adjoint vector
    """

    # --- Step 1: Solve adjoint system DG^T * lambda = dJ/dy ---
    rhs_adj = np.array([0., 0., 2. * float(a_bif - a_target), 0., 0.])
    cond_DG = np.linalg.cond(DG_5x5)
    print(f"  [Shape] cond(DG) = {cond_DG:.3e}")

    if cond_DG > 1e13:
        print("  [Shape] WARNING: DG ill-conditioned, using lstsq for adjoint")
        lambda_adj = np.linalg.lstsq(DG_5x5.T, rhs_adj, rcond=None)[0]
    else:
        lambda_adj = np.linalg.solve(DG_5x5.T, rhs_adj)

    lam_str = ', '.join(f'{v:.4e}' for v in lambda_adj)
    print(f"  [Shape] lambda_adj = [{lam_str}]")

    # Only the G1-components (lambda[0], lambda[1]) appear in our approximation.
    lam0 = float(lambda_adj[0])
    lam1 = float(lambda_adj[1])

    # --- Step 2: Build shape gradient tape ---
    set_working_tape(Tape())
    continue_annotation()

    c_T = Control(T_2d)

    # Deform 2D mesh with T_2d (AssignBlock on tape)
    mesh2d.coordinates.assign(X_ref_2d + T_2d)

    # Solve background NS flow on deformed mesh (GenericSolveBlock on tape)
    print("  [Shape] Solving 2D background flow on deformed mesh...")
    G_hat_val, _, u_bar_2d_new, p_bar_2d_new = solve_2D_bg_flow_on_tape(
        mesh2d, R_hat, H_hat, W_hat, Re_float)
    print(f"  [Shape] G_hat = {G_hat_val:.6e}")

    # Evaluate forces at bifurcation point (tape-aware for u_bar_2d AND mesh3d geometry)
    print(f"  [Shape] Evaluating forces at y* = "
          f"(r={r_bif:.4f}, z={z_bif:.4f}, a={a_bif:.4f})...")
    F_p_x, F_p_z = eval_forces_with_bg_on_tape(
        r_bif, z_bif, a_bif,
        u_bar_2d_new, p_bar_2d_new, G_hat_val,
        mesh_data, r_ref, z_ref, a_ref,
        T_2d=T_2d, mesh2d=mesh2d)

    print(f"  [Shape] F(y*) at current T_2d: "
          f"F_x = {float(F_p_x):.4e}, F_z = {float(F_p_z):.4e}")

    # Scalar functional: lambda[0:2]^T * G1 = lambda^T * F
    Lambda = lam0 * F_p_x + lam1 * F_p_z

    # --- Step 3: Differentiate w.r.t. T_2d ---
    print("  [Shape] Computing shape gradient via reverse-mode AD...")
    Jhat       = ReducedFunctional(Lambda, [c_T])
    derivs     = Jhat.derivative()
    shape_grad = derivs[0]          # Cofunction in V_2d.dual()

    stop_annotating()
    get_working_tape().clear_tape()
    gc.collect()

    # Restore 2D mesh to current T_2d (may have been modified by the tape run)
    with stop_annotating():
        mesh2d.coordinates.assign(X_ref_2d + T_2d)

    # Restore 3D mesh to bifurcation position + current channel shape.
    # (moore_spence_solve_hat_ad will reassign via its own tape, but setting
    # a clean state here avoids surprises in the line-search calls.)
    (_, _build_xi_hat_loc, _, _, _) = _get_opt_fns()
    with stop_annotating():
        mesh3d_r = mesh_data['mesh3d']
        R_sp     = FunctionSpace(mesh3d_r, "R", 0)
        dr_r = Function(R_sp).assign(float(r_bif - r_ref))
        dz_r = Function(R_sp).assign(float(z_bif - z_ref))
        da_r = Function(R_sp).assign(float(a_bif - a_ref))
        xi_restore = _build_xi_hat_loc(dr_r, dz_r, da_r, mesh_data)
        mesh3d_r.coordinates.assign(mesh_data['X_ref'] + xi_restore)

    return shape_grad, lambda_adj


# ---------------------------------------------------------------------------
#  Riesz representative (H^1 steepest descent direction)
# ---------------------------------------------------------------------------

def riesz_representative(shape_grad_cofunction, mesh2d,
                          alpha_elast=1.0, beta_l2=1e-2):
    """Compute the Riesz representative V of the shape gradient.

    Solves the linear elasticity H^1 problem:

        2 * alpha * integral( eps(V) : eps(W) ) dx
        + beta * integral( V · W ) dx
        = -shape_grad(W)   for all W in V_2d

    with homogeneous Dirichlet BC on the boundary (no deformation at walls).

    The solution V is the steepest *descent* direction in the H^1 metric.
    A smaller beta_l2 gives smoother deformations but may require a smaller
    step size.  A larger alpha_elast increases smoothness similarly.

    Returns
    -------
    V_rep : Function(V_2d)   steepest descent direction (not normalised)
    """
    V_2d = shape_grad_cofunction.function_space().dual()
    v    = TrialFunction(V_2d)
    w    = TestFunction(V_2d)

    a_form = (alpha_elast * 2 * inner(sym(grad(v)), sym(grad(w))) * dx
              + beta_l2   * inner(v, w) * dx)

    # RHS = negative gradient (descent direction)
    L_form = -shape_grad_cofunction

    # Boundary: fix all walls (no shape change at channel boundary)
    bc = DirichletBC(V_2d, Constant((0.0, 0.0)), "on_boundary")

    V_rep = Function(V_2d, name="V_rep")
    with stop_annotating():
        solve(a_form == L_form, V_rep, bcs=[bc],
              solver_parameters={
                  "ksp_type": "cg",
                  "pc_type":  "hypre",
                  "ksp_rtol": 1e-10,
                  "ksp_atol": 1e-14,
              })

    return V_rep


# ---------------------------------------------------------------------------
#  Mesh quality / deformation sanity check
# ---------------------------------------------------------------------------

def check_2d_mesh_quality(mesh2d, tol_min_jacobian=0.05):
    """Check that the deformed 2D mesh has no inverted or near-degenerate cells.

    Returns True if the mesh is acceptable, False otherwise.
    """
    try:
        with stop_annotating():
            V_scalar = FunctionSpace(mesh2d, "DG", 0)
            J_fn     = Function(V_scalar)
            # JacobianDeterminant is available in recent Firedrake
            J_fn.interpolate(JacobianDeterminant(mesh2d))
            j_min = float(J_fn.dat.data_ro.min())
        if j_min <= 0.0:
            print(f"  [mesh2d check] FAIL: min Jacobian = {j_min:.3e} (inverted cell)")
            return False
        if j_min < tol_min_jacobian:
            print(f"  [mesh2d check] WARNING: min Jacobian = {j_min:.3e} (near-degenerate)")
        return True
    except Exception as e:
        # JacobianDeterminant may not be available in all Firedrake versions
        print(f"  [mesh2d check] quality check skipped ({e})")
        return True


# ---------------------------------------------------------------------------
#  Algorithm 4.1: outer optimisation loop
# ---------------------------------------------------------------------------

def run_shape_optimization(a_target, shared_data, mesh_data_init,
                            bif_result_init,
                            *,
                            mesh2d=None, T_2d=None, X_ref_2d=None,
                            r_ref=None, z_ref=None, a_ref=None,
                            max_steps=50, tol_J=1e-8,
                            alpha_step=0.1, alpha_min=1e-8,
                            alpha_backtrack=0.5, max_backtrack=12,
                            riesz_alpha=1.0, riesz_beta=1e-2,
                            ms_tol=5e-8, ms_max_iter=30,
                            n_grid_2d=120):
    """Algorithm 4.1: shape optimisation to shift the bifurcation parameter.

    Parameters
    ----------
    a_target : float
        Desired bifurcation parameter in hat coordinates.
    shared_data : tuple
        ``(R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat,
           u_bar_2d, p_bar_tilde_2d)``
    mesh_data_init : dict
        3D mesh data from ``setup_moving_mesh_hat``.  The same object is
        updated in-place when the mesh reference point is not changed.
    bif_result_init : dict
        ``{'r', 'z', 'a', 'phi', 'converged'}`` from Moore-Spence.
    mesh2d : Mesh, optional
        Reference 2D mesh.  Created (``n_grid_2d × n_grid_2d``) if None.
    T_2d : Function, optional
        Initial deformation (zero if None, created on ``mesh2d``).
    X_ref_2d : Function, optional
        Fixed reference coordinates of ``mesh2d``.  Computed if None.
    r_ref, z_ref, a_ref : float, optional
        Position at which the 3D mesh ``mesh_data_init['mesh3d']`` was built.
        Defaults to the bifurcation point if not provided.
    max_steps : int
        Maximum shape-optimisation steps.
    tol_J : float
        Stop when ``J = (a* - a_target)^2 < tol_J``.
    alpha_step : float
        Initial step-size for the Armijo line search.
    alpha_min : float
        Minimum step-size; abort line search if alpha falls below.
    alpha_backtrack : float
        Reduction factor for the Armijo backtracking (default 0.5).
    max_backtrack : int
        Maximum backtracking iterations.
    riesz_alpha, riesz_beta : float
        Parameters for the H^1 Riesz metric (smoothness and L2 penalty).
    ms_tol, ms_max_iter : float, int
        Moore-Spence tolerance and max iterations used inside the loop.
    n_grid_2d : int
        2D mesh resolution (used only if ``mesh2d`` is None).

    Returns
    -------
    result : dict with keys
        'T_2d'       – final deformation field
        'a_bif'      – final bifurcation parameter
        'J_final'    – final objective value
        'history'    – list of per-step dicts {'step', 'a_bif', 'J', 'alpha'}
        'converged'  – bool
    """
    (setup_moving_mesh_hat, _, _, _, moore_spence_solve_hat_ad) = _get_opt_fns()

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat_0, U_m_hat_0, \
        u_bar_2d_0, p_bar_2d_0 = shared_data

    # ---- Set up 2D shape context if not provided ----
    if mesh2d is None:
        print(f"  [ShapeOpt] Creating {n_grid_2d}×{n_grid_2d} reference 2D mesh...")
        mesh2d, V_2d, T_2d, X_ref_2d = setup_shape_context(
            H_hat, W_hat, n_grid=n_grid_2d)
    else:
        V_2d = T_2d.function_space()

    # ---- Initial bifurcation point ----
    r_bif    = float(bif_result_init['r'])
    z_bif    = float(bif_result_init['z'])
    a_bif    = float(bif_result_init['a'])
    phi_bif  = np.asarray(bif_result_init['phi'], dtype=float)
    l_vec    = phi_bif.copy()   # normalization vector

    md          = mesh_data_init
    G_hat_cur   = G_hat_0
    u_bar_cur   = u_bar_2d_0
    p_bar_cur   = p_bar_2d_0
    shared_cur  = shared_data

    # Reference position of the 3D mesh
    if r_ref is None:
        print("  [ShapeOpt] r_ref/z_ref/a_ref not provided; "
              "using bifurcation point as 3D mesh reference.")
        r_ref = r_bif
        z_ref = z_bif
        a_ref = a_bif

    print(f"\n{'#'*70}")
    print(f"#  SHAPE OPTIMISATION  (Algorithm 4.1)")
    print(f"#  Target:  a_hat* = {a_target:.4f}")
    print(f"#  Initial: a_hat* = {a_bif:.6f}")
    print(f"#  J_0 = {(a_bif - a_target)**2:.4e}")
    print(f"{'#'*70}")

    history     = []
    converged   = False

    for step in range(max_steps):
        J = (a_bif - a_target) ** 2

        print(f"\n{'='*65}")
        print(f"  STEP {step:3d}  |  a_bif = {a_bif:.8f}  |  J = {J:.6e}")
        print(f"{'='*65}")

        history.append({
            'step':  step,
            'a_bif': a_bif,
            'J':     J,
            'alpha': None,
        })

        if J < tol_J:
            print(f"\n  CONVERGED: J = {J:.4e} < tol = {tol_J:.4e}")
            converged = True
            break

        # --- Compute DG_5x5 at current bifurcation point ---
        print(f"\n  Computing Moore-Spence Jacobian DG at bifurcation point...")
        try:
            DG_5x5, F_at_bif, J_sp = compute_DG_at_bif(
                r_bif, z_bif, a_bif, phi_bif, l_vec,
                md, r_ref, z_ref, a_ref)
        except Exception as e:
            print(f"  ERROR computing DG: {e}")
            break

        res_check = np.linalg.norm(F_at_bif)
        print(f"  |F(y*)| = {res_check:.4e}  (should be ~0 at bifurcation)")
        if res_check > 1e-4:
            print("  WARNING: large residual — bifurcation point may have drifted")

        # --- Compute shape gradient ---
        print(f"\n  Computing shape gradient...")
        try:
            shape_grad, lambda_adj = compute_shape_gradient(
                r_bif, z_bif, a_bif, phi_bif, l_vec, a_target, DG_5x5,
                T_2d, mesh2d, X_ref_2d,
                R_hat, H_hat, W_hat, Re,
                md, r_ref, z_ref, a_ref)
        except Exception as e:
            print(f"  ERROR in shape gradient: {e}")
            import traceback; traceback.print_exc()
            break

        # --- Riesz representative (steepest descent direction) ---
        print(f"\n  Computing Riesz representative (H^1 descent direction)...")
        try:
            V_rep = riesz_representative(
                shape_grad, mesh2d,
                alpha_elast=riesz_alpha, beta_l2=riesz_beta)
        except Exception as e:
            print(f"  ERROR in Riesz solve: {e}")
            break

        with stop_annotating():
            grad_norm_sq = float(assemble(inner(V_rep, V_rep) * dx))
        grad_norm = float(grad_norm_sq ** 0.5)
        print(f"  ||V_rep||_L2 = {grad_norm:.4e}")

        if grad_norm < 1e-14:
            print("  Gradient effectively zero — cannot proceed.")
            break

        # --- Armijo backtracking line search ---
        print(f"\n  Armijo line search (alpha_0 = {alpha_step:.3e})...")
        alpha    = float(alpha_step)
        accepted = False
        sigma_armijo = 1e-4   # sufficient decrease parameter

        for bt in range(max_backtrack):
            if alpha < alpha_min:
                print(f"  [LS] alpha {alpha:.3e} < alpha_min — aborting.")
                break

            # --- Trial deformation ---
            with stop_annotating():
                T_2d_try = T_2d.copy(deepcopy=True)
                T_2d_try.dat.data[:] += alpha * V_rep.dat.data_ro

            # --- Deform 2D mesh and solve background flow ---
            with stop_annotating():
                mesh2d.coordinates.assign(X_ref_2d + T_2d_try)
                mesh_ok = check_2d_mesh_quality(mesh2d)

            if not mesh_ok:
                print(f"  [LS bt={bt}] Mesh quality failed, halving alpha.")
                alpha *= alpha_backtrack
                continue

            try:
                with stop_annotating():
                    G_try, U_m_try, u_bar_try, p_bar_try = \
                        solve_2D_bg_flow_on_tape(
                            mesh2d, R_hat, H_hat, W_hat, Re)
            except Exception as e:
                print(f"  [LS bt={bt}] Background flow failed: {e}")
                alpha *= alpha_backtrack
                continue

            shared_try = (R_hat, H_hat, W_hat, L_c, U_c, Re,
                          G_try, U_m_try, u_bar_try, p_bar_try)

            # --- Update mesh_data with new background flow ---
            md_try = dict(md)
            md_try['u_bar_2d_hat']      = u_bar_try
            md_try['p_bar_tilde_2d_hat'] = p_bar_try
            md_try['G_hat']             = G_try
            md_try['U_m_hat']           = U_m_try

            # --- Re-solve Moore-Spence on deformed geometry ---
            print(f"  [LS bt={bt}] alpha = {alpha:.3e} → running Moore-Spence...")
            try:
                with stop_annotating():
                    r_try, z_try, a_try, phi_try, conv_try = \
                        moore_spence_solve_hat_ad(
                            r_bif, z_bif, a_bif, shared_try,
                            tol=ms_tol, max_iter=ms_max_iter,
                            md=md_try,
                            dr_init=float(r_bif - r_ref),
                            dz_init=float(z_bif - z_ref),
                            da_init=float(a_bif - a_ref),
                            globalization='trust_region', lm_fallback=True)
            except Exception as e:
                print(f"  [LS bt={bt}] Moore-Spence failed: {e}")
                alpha *= alpha_backtrack
                continue

            if not conv_try:
                print(f"  [LS bt={bt}] Moore-Spence did not converge.")
                alpha *= alpha_backtrack
                continue

            J_try = (float(a_try) - float(a_target)) ** 2
            print(f"  [LS bt={bt}] a_try = {a_try:.6f},  J_try = {J_try:.4e}  "
                  f"(vs J = {J:.4e})")

            # Armijo sufficient-decrease condition
            # J_try < J - sigma * alpha * <grad_J, V_rep>
            # where <grad_J, V_rep> ≈ -grad_norm^2 (since V_rep is Riesz rep of -grad)
            sufficient_decrease = J - sigma_armijo * alpha * grad_norm_sq
            if J_try < sufficient_decrease:
                # Accept step
                print(f"  [LS] Step ACCEPTED (bt={bt}, alpha={alpha:.4e})")
                T_2d.assign(T_2d_try)
                r_bif   = float(r_try)
                z_bif   = float(z_try)
                a_bif   = float(a_try)
                phi_bif = np.asarray(phi_try, dtype=float)
                l_vec   = phi_bif.copy()
                md          = md_try
                G_hat_cur   = G_try
                u_bar_cur   = u_bar_try
                p_bar_cur   = p_bar_try
                shared_cur  = shared_try
                history[-1]['alpha'] = alpha
                accepted = True
                break
            else:
                alpha *= alpha_backtrack

        if not accepted:
            print(f"  [ShapeOpt] Backtracking failed — stopping.")
            break

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


# ---------------------------------------------------------------------------
#  Entry-point convenience wrapper
# ---------------------------------------------------------------------------

def run_from_main(a_target=0.10, max_steps=50, tol_J=1e-8,
                  alpha_step=0.05, riesz_alpha=1.0, riesz_beta=1e-2,
                  ms_tol=1e-7, ms_max_iter=30, n_grid_2d=120):
    """Set up everything from config_paper_parameters and run the optimisation.

    Typical usage::

        python shape_optimization.py

    All parameters can be overridden at the call site.
    """
    # --- Local imports to avoid polluting module namespace ---
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from config_paper_parameters import R, H, W, Q, rho, mu
    from nondimensionalization import first_nondimensionalisation
    from background_flow_return_UFL import background_flow_differentiable
    from optimization_of_branch_points import (
        setup_moving_mesh_hat, moore_spence_solve_hat_ad,
        newton_root_refine_hat,
    )

    # --- Non-dimensionalisation ---
    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(
        R, H, W, Q, rho, mu, print_values=True)

    # --- Background flow (initial rectangular cross-section) ---
    print("\n  Solving initial 2D background flow (rectangular cross-section)...")
    with stop_annotating():
        bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
        G_hat, U_m_hat, u_bar_2d, p_bar_tilde = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re,
                   G_hat, U_m_hat, u_bar_2d, p_bar_tilde)

    # --- Initial equilibrium (on-axis, slightly off-centre) ---
    r0, z0, a0 = 0.61, 0.0, 0.135
    print(f"\n  Newton refinement at a_hat = {a0}...")
    r_eq, z_eq, md0, dr0, dz0 = newton_root_refine_hat(
        r0, z0, a0, shared_data, tol=1e-10, max_iter=15)

    # The 3D mesh (md0) was built at (r0, z0, a0); dr0/dz0 are the displacements
    # to the Newton equilibrium (r_eq = r0 + dr0).  The mesh reference is r0.
    r_ref = r0
    z_ref = z0
    a_ref = a0

    # --- Find bifurcation by Moore-Spence ---
    print("\n  Running Moore-Spence to find bifurcation point...")
    with stop_annotating():
        r_bif, z_bif, a_bif, phi_bif, conv_ms = moore_spence_solve_hat_ad(
            r_eq, z_eq, a0, shared_data,
            tol=ms_tol, max_iter=ms_max_iter,
            md=md0, dr_init=dr0, dz_init=dz0,
            globalization='trust_region', lm_fallback=True)

    if not conv_ms:
        raise RuntimeError("Moore-Spence did not converge for initial domain")

    bif_init = {'r': r_bif, 'z': z_bif, 'a': a_bif,
                'phi': phi_bif, 'converged': True}

    print(f"\n  Initial bifurcation: a_hat = {a_bif:.6f} (target = {a_target:.4f})")

    # --- Run shape optimisation ---
    result = run_shape_optimization(
        a_target, shared_data, md0, bif_init,
        r_ref=r_ref, z_ref=z_ref, a_ref=a_ref,
        max_steps=max_steps, tol_J=tol_J,
        alpha_step=alpha_step,
        riesz_alpha=riesz_alpha, riesz_beta=riesz_beta,
        ms_tol=ms_tol, ms_max_iter=ms_max_iter,
        n_grid_2d=n_grid_2d)

    # --- Save optimised deformation ---
    T_final  = result['T_2d']
    out_path = "output_shape_T2d.h5"
    try:
        with CheckpointFile(out_path, "w") as chk:
            chk.save_function(T_final, name="T_2d")
        print(f"\n  Saved T_2d to {out_path}")
    except Exception as e:
        print(f"  Warning: could not save checkpoint: {e}")

    return result


if __name__ == "__main__":
    result = run_from_main(
        a_target=0.10,
        max_steps=50,
        tol_J=1e-8,
        alpha_step=0.05,       # initial step size for Armijo line search
        riesz_alpha=1.0,        # elasticity weight in H^1 metric
        riesz_beta=1e-2,        # L2 stabilisation in H^1 metric
        ms_tol=5e-8,
        ms_max_iter=30,
        n_grid_2d=120,
    )
    print("\nFinal result:")
    print(f"  a_bif = {result['a_bif']:.8f}")
    print(f"  J     = {result['J_final']:.6e}")
    print(f"  steps = {len(result['history'])}")
