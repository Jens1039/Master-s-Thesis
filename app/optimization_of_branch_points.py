import os
os.environ["OMP_NUM_THREADS"] = "1"

import gc
import numpy as np
import math

from firedrake import *
from firedrake.adjoint import stop_annotating, continue_annotation, annotate_tape, ReducedFunctional, Control
from pyadjoint import set_working_tape, get_working_tape, Tape, Block
from nondimensionalization import first_nondimensionalisation
from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable
from perturbed_flow_return_UFL import perturbed_flow_differentiable
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole
from config_paper_parameters import R, H, W, Q, rho, mu, particle_maxh_rel, global_maxh_rel


class MeshFlippedError(Exception):
    pass


def check_mesh_quality(mesh3d, ref_signs=None):
    """
    Check for flipped (inverted) elements after mesh deformation.
    This works in the following way:
    mesh3d represents a geometric mapping x(ξ) that transforms a reference element (standard tetrahedron) into a "physical" element in 3D space
    x(ξ) = Σ X_i * ψ_i(ξ). where ξ (xi) are the coordinates in the reference space (unit tetrahedron)
    X_i are the global nodal coordinates (the 'coefficients' of the mesh)
    ψ_i (psi) are the shape functions (basis functions) on the reference element
    The jacobian matrix J = ∂x / ∂ξ is the derivative of this mapping with respect to the reference coordinates
        It describes how the reference element is stretched, rotated, and
        sheared to form the physical mesh.
    The jacobian determinant (det(J)) measures the local volume scaling
        - det(J) > 0: Valid element (mapping preserves orientation).
        - det(J) = 0: Degenerate element (element is flattened/zero volume).
        - det(J) < 0: Flipped element (the node ordering was inverted relative
                                        to the reference, or the element was 'turned inside out')
    """

    # We reserve memory space for exactly one number per finite element of our mesh (later filled up with the determinant)
    with stop_annotating():
        DG0 = FunctionSpace(mesh3d, "DG", 0)
        jac_det = Function(DG0)
        jac_det.interpolate(JacobianDeterminant(mesh3d))
    det_vals = jac_det.dat.data_ro
    signs = np.sign(det_vals)

    # If we have elements, that changed the sign, we know it flipped during the deformation
    if ref_signs is not None and not np.array_equal(signs, ref_signs):
        n_flipped = int(np.sum(signs != ref_signs))
        raise MeshFlippedError(f"Mesh has {n_flipped} flipped element(s)")

    # check for degenerate elements
    if float(np.abs(det_vals).min()) == 0.0:
        raise MeshFlippedError("Mesh has degenerate element(s)")

    return signs


def setup_moving_mesh_hat(r_off_hat_init, z_off_hat_init, a_hat_init, R_hat, H_hat, W_hat, Re, G_hat, U_m_hat,
                                                        u_bar_2d_hat, p_bar_tilde_2d_hat, a_mesh_size_res_ref=None):

    L_hat = 4 * max(H_hat, W_hat)

    # Option to ensure, that the resolution of our particle is always the same, even if particle size changes to reduce numerical noise
    if a_mesh_size_res_ref is None:
        a_mesh_size_res_ref = a_hat_init

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R_hat, H_hat, W_hat, L_hat, a_hat_init, particle_maxh_rel * a_mesh_size_res_ref,
                                                                    global_maxh_rel * min(H_hat, W_hat), r_off_hat_init, z_off_hat_init)

    # Mathematically this is the cartesian product of 3 identical spaces of piecewise (on each tetrahedron) linear functions
    V_def = VectorFunctionSpace(mesh3d, "CG", 1)

    with stop_annotating():
        # This function stores the cartesian coordinates of every vortex of our mesh
        # It is simply the continous version of SpatialCoordinate(mesh3d)
        X_ref = Function(V_def)
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    dist = sqrt((X_ref[0] - cx)**2 + (X_ref[1] - cy)**2 + (X_ref[2] - cz)**2)

    # The bump function is 1 on sphere surface and then linearly decays to 0 at r_cut
    r_cut = Constant(0.5 * min(H_hat, W_hat))
    a_c = Constant(a_hat_init)
    bump = max_value(Constant(0.0), 1.0 - max_value(Constant(0.0), dist - a_c) / (r_cut - a_c))

    # Unit direction from sphere center (for radial sphere scaling).
    # Safe because dist >= a_hat_init > 0 on all mesh nodes.
    d_hat_x = (X_ref[0] - cx) / dist
    d_hat_y = (X_ref[1] - cy) / dist
    d_hat_z = (X_ref[2] - cz) / dist

    theta_half = tags["theta"] / 2.0

    ref_signs = check_mesh_quality(mesh3d)

    # Pre-evaluate bump and d_hat to numpy arrays (off-tape) for XiHatBlock.
    with stop_annotating():
        V_scalar = FunctionSpace(mesh3d, "CG", 1)
        bump_fn = Function(V_scalar, name="bump_eval")
        bump_fn.interpolate(bump)
        bump_data = bump_fn.dat.data_ro.copy()
        d_hat_fn = Function(V_def, name="d_hat_eval")
        d_hat_fn.interpolate(as_vector([d_hat_x, d_hat_y, d_hat_z]))
        d_hat_data = d_hat_fn.dat.data_ro.copy()

    print(f"Mesh in hat coords: R={R_hat:.2f}, H={H_hat:.2f}, W={W_hat:.2f}, a={a_hat_init:.4f}, L={L_hat:.2f}")

    return {
        'mesh3d': mesh3d, 'tags': tags,
        'X_ref': X_ref, 'V_def': V_def,
        'bump': bump,
        'bump_data': bump_data, 'd_hat_data': d_hat_data,
        'd_hat_x': d_hat_x, 'd_hat_y': d_hat_y, 'd_hat_z': d_hat_z,
        'cos_th': math.cos(theta_half),
        'sin_th': math.sin(theta_half),
        'R_hat': R_hat, 'H_hat': H_hat, 'W_hat': W_hat,
        'L_hat': L_hat, 'a_hat_init': a_hat_init,
        'G_hat': G_hat, 'Re': Re, 'U_m_hat': U_m_hat,
        'u_bar_2d_hat': u_bar_2d_hat,
        'p_bar_tilde_2d_hat': p_bar_tilde_2d_hat,
        'ref_signs': ref_signs,
    }

def estimate_eigenvectors(J):

    tr = J[0, 0] + J[1, 1]
    det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    disc = tr**2 - 4 * det

    if disc < 0:
        raise ValueError(f"Complex eigenvalues (disc={disc:.4e})")

    sqrt_disc = np.sqrt(disc)
    mu1 = 0.5 * (tr - sqrt_disc)
    mu2 = 0.5 * (tr + sqrt_disc)

    def eigvec(mu):
        M = J - mu * np.eye(2)
        if abs(M[0, 0]) + abs(M[0, 1]) > abs(M[1, 0]) + abs(M[1, 1]):
            v = np.array([-M[0, 1], M[0, 0]])
        else:
            v = np.array([-M[1, 1], M[1, 0]])
        n = np.linalg.norm(v)
        return v / n if n > 1e-15 else np.array([1.0, 0.0])

    phi1 = eigvec(mu1)
    phi2 = eigvec(mu2)

    pairs = [(mu1, phi1), (mu2, phi2)]
    pairs.sort(key=lambda p: abs(p[0]))

    print(f"  Eigenvalues of J_sp: mu_1 = {pairs[0][0]:+.6e}, mu_2 = {pairs[1][0]:+.6e}")
    print(f"  |mu_min|/|mu_max| = {abs(pairs[0][0]) / (abs(pairs[1][0]) + 1e-30):.6e}")
    print(f"  phi_1 = ({pairs[0][1][0]:+.6f}, {pairs[0][1][1]:+.6f})")
    print(f"  phi_2 = ({pairs[1][1][0]:+.6f}, {pairs[1][1][1]:+.6f})")

    return pairs


class XiHatBlock(Block):
    """Tape-aware mesh deformation  xi = f(delta_r, delta_z, delta_a).

    The map is LINEAR in the three R-space controls:
        xi[n, 0] = (delta_r · cos_th + delta_a · d_hat[n, 0]) · bump[n]
        xi[n, 1] = (delta_r · sin_th + delta_a · d_hat[n, 1]) · bump[n]
        xi[n, 2] = (delta_z           + delta_a · d_hat[n, 2]) · bump[n]

    Because the map is linear, the second derivative vanishes: the Hessian
    component is simply the adjoint applied to the second-order seed.

    This replaces ``xi.interpolate(as_vector([...]))`` which puts an
    InterpolateBlock on the tape.  That block's Hessian path crashes in
    pyadjoint/UFL when the expression mixes R-space Functions with CG1
    expressions (BaseFormOperatorDerivative Sum bug).

    Dependencies: idx 0 = delta_r, idx 1 = delta_z, idx 2 = delta_a.
    """

    def __init__(self, delta_r, delta_z, delta_a, xi_out,
                 cos_th, sin_th, bump_data, d_hat_data, V_def):
        super().__init__()
        self.add_dependency(delta_r)
        self.add_dependency(delta_z)
        self.add_dependency(delta_a)
        self.add_output(xi_out.create_block_variable())
        self.cos_th = cos_th
        self.sin_th = sin_th
        self.bump = bump_data       # (n_nodes,)
        self.d_hat = d_hat_data     # (n_nodes, 3)
        self.V_def = V_def

    def recompute_component(self, inputs, block_variable, idx, prepared):
        dr = float(inputs[0].dat.data_ro[0])
        dz = float(inputs[1].dat.data_ro[0])
        da = float(inputs[2].dat.data_ro[0])
        out = block_variable.output
        b, d = self.bump, self.d_hat
        with stop_annotating():
            out.dat.data[:, 0] = (dr * self.cos_th + da * d[:, 0]) * b
            out.dat.data[:, 1] = (dr * self.sin_th + da * d[:, 1]) * b
            out.dat.data[:, 2] = (dz               + da * d[:, 2]) * b
        return out

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx,
                               prepared=None):
        if adj_inputs[0] is None:
            return None
        a = np.asarray(adj_inputs[0].dat.data_ro)   # (n_nodes, 3)
        b, d = self.bump, self.d_hat
        with stop_annotating():
            R_space = inputs[idx].function_space()
            out = Cofunction(R_space.dual())
            if idx == 0:    # delta_r
                out.dat.data[0] = float(np.sum(
                    self.cos_th * b * a[:, 0] + self.sin_th * b * a[:, 1]))
            elif idx == 1:  # delta_z
                out.dat.data[0] = float(np.sum(b * a[:, 2]))
            elif idx == 2:  # delta_a
                out.dat.data[0] = float(np.sum(
                    d[:, 0] * b * a[:, 0] + d[:, 1] * b * a[:, 1]
                    + d[:, 2] * b * a[:, 2]))
        return out

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx,
                               prepared=None):
        b, d = self.bump, self.d_hat
        with stop_annotating():
            out = Function(self.V_def)
            data = np.zeros_like(out.dat.data)
            if tlm_inputs[0] is not None:
                ddr = float(tlm_inputs[0].dat.data_ro[0])
                data[:, 0] += ddr * self.cos_th * b
                data[:, 1] += ddr * self.sin_th * b
            if tlm_inputs[1] is not None:
                ddz = float(tlm_inputs[1].dat.data_ro[0])
                data[:, 2] += ddz * b
            if tlm_inputs[2] is not None:
                dda = float(tlm_inputs[2].dat.data_ro[0])
                data[:, 0] += dda * d[:, 0] * b
                data[:, 1] += dda * d[:, 1] * b
                data[:, 2] += dda * d[:, 2] * b
            out.dat.data[:] = data
        return out

    def evaluate_hessian_component(self, inputs, hessian_inputs, adj_inputs,
                                   block_variable, idx, relevant_dependencies,
                                   prepared=None):
        # Linear block ⇒ second derivative is zero ⇒ only pass through
        # the adjoint of the second-order seed.
        if hessian_inputs[0] is None:
            return None
        return self.evaluate_adj_component(
            inputs, hessian_inputs, block_variable, idx, prepared)


def _build_xi_hat(delta_r, delta_z, delta_a, md):
    """Build mesh deformation via custom tape block (Hessian-safe).
    Replaces the original ``xi.interpolate(as_vector([...]))`` which crashes
    pyadjoint's Hessian on InterpolateBlocks mixing R-space and CG1 coefficients.
    """
    xi = Function(md['V_def'], name="xi")
    b, d = md['bump_data'], md['d_hat_data']
    dr = float(delta_r.dat.data_ro[0])
    dz = float(delta_z.dat.data_ro[0])
    da = float(delta_a.dat.data_ro[0])
    with stop_annotating():
        xi.dat.data[:, 0] = (dr * md['cos_th'] + da * d[:, 0]) * b
        xi.dat.data[:, 1] = (dr * md['sin_th'] + da * d[:, 1]) * b
        xi.dat.data[:, 2] = (dz                + da * d[:, 2]) * b
    if annotate_tape():
        block = XiHatBlock(delta_r, delta_z, delta_a, xi,
                           md['cos_th'], md['sin_th'], b, d, md['V_def'])
        get_working_tape().add_block(block)
    return xi


def evaluate_forces_hat(delta_r_hat, delta_z_hat, delta_a_hat, mesh_data, compute_jacobian=True):
    """Evaluate particle forces in hat coordinates with AD Jacobian.

    Returns F (2-vector) if compute_jacobian=False,
    or (F, J) where J is (2x3) if compute_jacobian=True.
    J columns: [dF/d(delta_r), dF/d(delta_z), dF/d(delta_a)].
    """

    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']
    R_space = FunctionSpace(mesh3d, "R", 0)

    delta_r = Function(R_space, name="delta_r").assign(delta_r_hat)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hat)
    delta_a = Function(R_space, name="delta_a").assign(delta_a_hat)

    xi = _build_xi_hat(delta_r, delta_z, delta_a, md)
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['G_hat'],
        mesh3d, md['tags'], md['u_bar_2d_hat'], md['p_bar_tilde_2d_hat'],
        X_ref=md['X_ref'], xi=xi)

    # Tape-connected total particle radius a_total = a_hat_init + delta_a.
    # Passing this (instead of the frozen scalar md['a_hat_init']) lets the
    # centrifugal a^3 prefactor in perturbed_flow_differentiable update with
    # delta_a, so the moving-mesh evaluation matches a fresh-mesh evaluation
    # of the same physical sphere radius.  Re_p stays as md['Re'] — that is
    # the hat-system convention and must NOT be touched.
    pf = perturbed_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['L_hat'],
        (md['a_hat_init'], delta_a), md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    if not compute_jacobian:
        stop_annotating()
        get_working_tape().clear_tape()
        gc.collect()
        return F

    c_r = Control(delta_r)
    c_z = Control(delta_z)
    c_a = Control(delta_a)

    Jhat_x = ReducedFunctional(F_p_x, [c_r, c_z, c_a])
    Jhat_z = ReducedFunctional(F_p_z, [c_r, c_z, c_a])

    dFx = Jhat_x.derivative()
    dFz = Jhat_z.derivative()

    J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = float(dFx[j].dat.data_ro[0])
        J[1, j] = float(dFz[j].dat.data_ro[0])

    stop_annotating()
    get_working_tape().clear_tape()
    gc.collect()
    return F, J


def evaluate_forces_jac_hessian_hat(delta_r_hat, delta_z_hat, delta_a_hat, mesh_data, phi):
    """Evaluate F, J, and d(J·phi)/dx via reverse + Hessian-vector AD.

    For each force component F_i (i = x, z), pyadjoint's Hessian-vector
    product H_i · m_dot with m_dot = (phi[0], phi[1], 0) evaluates to
        d/dx [ sum_j dF_i/dx_j · phi_j ] = d/dx [ (J·phi)_i ]
    which is exactly the column-block dG2/dx that Moore-Spence needs,
    without any finite differences.

    Returns
    -------
    F        : (2,)  forces (F_x, F_z)
    J        : (2,3) Jacobian [dF/d(delta_r), dF/d(delta_z), dF/d(delta_a)]
    dJphi_dx : (2,3) entries d/dx_k [(J·phi)_i] from H_i · phi_extended.
               Rows = force components, cols = [d/dr, d/dz, d/da].
    """

    set_working_tape(Tape())
    continue_annotation()

    md = mesh_data
    mesh3d = md['mesh3d']
    R_space = FunctionSpace(mesh3d, "R", 0)

    delta_r = Function(R_space, name="delta_r").assign(delta_r_hat)
    delta_z = Function(R_space, name="delta_z").assign(delta_z_hat)
    delta_a = Function(R_space, name="delta_a").assign(delta_a_hat)

    xi = _build_xi_hat(delta_r, delta_z, delta_a, md)
    mesh3d.coordinates.assign(md['X_ref'] + xi)
    check_mesh_quality(mesh3d, ref_signs=md['ref_signs'])

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['G_hat'],
        mesh3d, md['tags'], md['u_bar_2d_hat'], md['p_bar_tilde_2d_hat'],
        X_ref=md['X_ref'], xi=xi)

    # Tape-connected total radius (see evaluate_forces_hat for rationale).
    pf = perturbed_flow_differentiable(
        md['R_hat'], md['H_hat'], md['W_hat'], md['L_hat'],
        (md['a_hat_init'], delta_a), md['Re'],
        mesh3d, md['tags'], u_bar_3d, p_bar_3d,
        md['X_ref'], xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()
    F = np.array([float(F_p_x), float(F_p_z)])

    c_r = Control(delta_r)
    c_z = Control(delta_z)
    c_a = Control(delta_a)
    controls = [c_r, c_z, c_a]

    rf_x = ReducedFunctional(F_p_x, controls)
    rf_z = ReducedFunctional(F_p_z, controls)

    # First-order: full Jacobian via reverse AD
    dFx = rf_x.derivative()
    dFz = rf_z.derivative()
    J = np.zeros((2, 3))
    for j in range(3):
        J[0, j] = float(dFx[j].dat.data_ro[0])
        J[1, j] = float(dFz[j].dat.data_ro[0])

    # Hessian-vector seed: phi extended with 0 for the a-direction
    phi_r_fn = Function(R_space).assign(float(phi[0]))
    phi_z_fn = Function(R_space).assign(float(phi[1]))
    phi_a_fn = Function(R_space).assign(0.0)
    m_dot = [phi_r_fn, phi_z_fn, phi_a_fn]

    # IMPORTANT: derivative() must be called immediately before hessian()
    # for the SAME functional.  Each derivative() call overwrites adj_sol
    # on every SolveBlock on the tape.  If we call rf_z.derivative() then
    # rf_x.hessian(), the hessian uses F_z's adjoint solution — wrong!
    rf_x.derivative()
    Hphi_x = rf_x.hessian(m_dot)
    rf_z.derivative()
    Hphi_z = rf_z.hessian(m_dot)

    dJphi_dx = np.zeros((2, 3))
    for j in range(3):
        dJphi_dx[0, j] = float(Hphi_x[j].dat.data_ro[0])
        dJphi_dx[1, j] = float(Hphi_z[j].dat.data_ro[0])

    stop_annotating()
    get_working_tape().clear_tape()
    gc.collect()
    return F, J, dJphi_dx



def _ms_trial_ad(dr_init, dz_init, da_init, phi_start, l_vec, md, r_ref, z_ref, a_ref, L_c, tol, max_iter):
    """Single Moore-Spence trial with a given starting phi (AD Hessian, TR+LM).
    Returns (r, z, a, phi, converged, final_residual)."""

    dr, dz, da = float(dr_init), float(dz_init), float(da_init)
    phi = phi_start.copy()
    Delta, Delta_max = 1e-3, 1.0
    eta_accept, eta_good = 0.1, 0.75
    res = float('inf')

    for k in range(max_iter):

        # All entries of DG via AD: 1 forward + 2 reverse + 2 H·v
        F_base, J_full, dJphi_dx = evaluate_forces_jac_hessian_hat(
            dr, dz, da, md, phi)
        J_sp = J_full[:, :2]
        dF_da = J_full[:, 2]

        G1 = F_base
        G2 = J_sp @ phi
        G3_val = np.dot(l_vec, phi) - 1.0
        G = np.concatenate([G1, G2, [G3_val]])
        res = np.linalg.norm(G)

        eigs = np.linalg.eigvals(J_sp)
        eig_min = eigs[np.argmin(np.abs(eigs))]
        r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
        print(f"\n  Iter {k:2d} | r = {r_cur:+.8f}  z = {z_cur:+.8f}  a = {a_cur:.8f}")
        print(f"         | |G| = {res:.4e}  |F| = {np.linalg.norm(G1):.4e}"
              f"  |J*phi| = {np.linalg.norm(G2):.4e}"
              f"  eig_min = {eig_min:+.4e}")

        if res < tol:
            a_phys = a_cur * L_c
            print(f"\n  -> Bifurcation point found after {k} iterations!")
            print(f"     r_off_hat = {r_cur:.10f}")
            print(f"     z_off_hat = {z_cur:.10f}")
            print(f"     a_hat     = {a_cur:.10f}  (a = {a_phys * 1e6:.4f} um)")
            print(f"     phi       = ({phi[0]:.8f}, {phi[1]:.8f})")
            print(f"     sigma_min = {sv.min():.6e}")
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

        D = np.ones(5)
        DG_s = DG.copy()

        cond_DG = np.linalg.cond(DG)
        cond_DG_s = np.linalg.cond(DG_s)
        print(f"         | cond(DG) = {cond_DG:.2e}")

        if cond_DG_s > 1e14:
            print("  !! DG ill-conditioned — aborting.")
            return r_cur, z_cur, a_cur, phi, False, res

        try:
            dy_s = np.linalg.solve(DG_s, -G)
        except np.linalg.LinAlgError:
            print("  !! DG singular — aborting.")
            return r_cur, z_cur, a_cur, phi, False, res

        dy = dy_s / D

        print(f"         | step: dr={dy[0]:+.4e}  dz={dy[1]:+.4e}  da={dy[2]:+.4e}")

        # ── Trust-region globalization with LM fallback ──
        # dy_s is the equilibrated step (dimensionless after column scaling).
        # The TR radius Delta is measured in this scaled space.

        def _eval_step(dy_candidate):
            dr_t = dr + dy_candidate[0]
            dz_t = dz + dy_candidate[1]
            da_t = da + dy_candidate[2]
            phi_t = phi + dy_candidate[3:5]
            if a_ref + da_t <= 0:
                return float('nan'), False
            try:
                F_t, J_t = evaluate_forces_hat(dr_t, dz_t, da_t, md)
                G_t = np.concatenate([F_t, J_t[:, :2] @ phi_t,
                                      [np.dot(l_vec, phi_t) - 1.0]])
                return float(np.linalg.norm(G_t)), True
            except MeshFlippedError:
                return float('nan'), False

        '''
        step_norm_s = np.linalg.norm(dy_s)
        if step_norm_s > Delta:
            dy_s_try = dy_s * (Delta / step_norm_s)
            dy_try = dy_s_try / D
            print(f"         | TR: step clipped {step_norm_s:.4e} -> {Delta:.4e}"
                  f"  (dr={dy_try[0]:+.4e}  dz={dy_try[1]:+.4e}  da={dy_try[2]:+.4e})")
        else:
            dy_try = dy_s / D
        '''

        step_norm_phys = np.linalg.norm(dy[:3])  # nur r, z, a
        if step_norm_phys > Delta:
            dy_try = dy * (Delta / step_norm_phys)
            print(f"         | TR: physical step clipped {step_norm_phys:.4e} -> {Delta:.4e}")
        else:
            dy_try = dy.copy()

        # Predicted reduction from the linear model
        G_pred = G + DG @ dy_try
        pred = res**2 - np.linalg.norm(G_pred)**2

        # Evaluate actual reduction of the Newton step
        res_try, step_ok = _eval_step(dy_try)
        if step_ok:
            ared = res**2 - res_try**2
            rho = ared / pred if abs(pred) > 1e-30 else (1.0 if ared >= 0 else -1.0)
        else:
            rho = -1.0

        print(f"         | TR: |G_try|={res_try if step_ok else float('nan'):.4e}"
              f"  rho={rho:+.4f}  Delta={Delta:.4e}")

        accepted = (rho >= eta_accept and step_ok)

        # Optional LM fallback when the Newton step is rejected
        if not accepted:
            # Normal equations (DG_s^T DG_s + lam*I) dy_s = -DG_s^T G
            # always yield a descent direction for |G|^2.
            lam = 1e-6 * np.linalg.norm(DG_s)**2
            for _lm_it in range(8):
                A = DG_s.T @ DG_s + lam * np.eye(5)
                b = -DG_s.T @ G
                try:
                    dy_s_lm = np.linalg.solve(A, b)
                except np.linalg.LinAlgError:
                    lam *= 10.0
                    continue
                dy_lm = dy_s_lm / D
                res_lm, ok_lm = _eval_step(dy_lm)
                if ok_lm and res_lm < res:
                    print(f"         | LM(lam={lam:.2e}): |G_try|={res_lm:.4e}  ACCEPTED")
                    dy_try = dy_lm
                    res_try = res_lm
                    accepted = True
                    break
                print(f"         | LM(lam={lam:.2e}): |G_try|={res_lm:.4e}  reject")
                lam *= 10.0

        if accepted:
            dr += dy_try[0]
            dz += dy_try[1]
            da += dy_try[2]
            phi_new = phi + dy_try[3:5]
            phi = phi_new / np.dot(l_vec, phi_new)
            print(f"         | TR: step ACCEPTED")
        else:
            print(f"         | TR: step REJECTED (TR + all LM attempts failed)")

        # Update trust-region radius based on Newton rho
        if rho < 0.25:
            Delta = max(Delta * 0.25, 1e-8)
        elif rho > eta_good and step_norm_phys >= 0.95 * Delta:
            Delta = min(Delta * 2.0, Delta_max)

        gc.collect()

    r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da
    return r_cur, z_cur, a_cur, phi, False, res


def moore_spence_solve_hat_ad(r_off_hat_eq, z_off_hat_eq, a_hat_init, shared_data, *, tol=1e-8, max_iter=20, md=None,
                               dr_init=0.0, dz_init=0.0, da_init=0.0):

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data
    r_ref = float(r_off_hat_eq) - float(dr_init)
    z_ref = float(z_off_hat_eq) - float(dz_init)
    a_ref = float(a_hat_init)   - float(da_init)

    if md is None:
        md = setup_moving_mesh_hat(r_ref, z_ref, a_ref, R_hat, H_hat, W_hat,
                                   Re, G_hat, U_m_hat, u_2d, p_2d)

    F0, J0_full = evaluate_forces_hat(float(dr_init), float(dz_init), float(da_init), md)
    eigpairs = estimate_eigenvectors(J0_full[:, :2])

    results = []
    for trial_idx, (mu, phi_ev) in enumerate(eigpairs):
        l_vec = phi_ev.copy()
        phi_start = phi_ev / np.dot(l_vec, phi_ev)

        print("\n" + "=" * 65)
        print(f"  MOORE-SPENCE AD — trial {trial_idx+1}/{len(eigpairs)}"
              f"  mu={mu:+.4e}  phi=({phi_ev[0]:+.4f}, {phi_ev[1]:+.4f})")
        print(f"  Start: r={r_off_hat_eq:.6f} z={z_off_hat_eq:.6f} a={a_hat_init:.6f}")
        print("=" * 65)

        result = _ms_trial_ad(float(dr_init), float(dz_init), float(da_init),
                              phi_start, l_vec, md, r_ref, z_ref, a_ref, L_c, tol, max_iter)
        results.append(result)

    converged = [r for r in results if r[4]]
    best = min(converged, key=lambda x: x[5]) if converged else min(results, key=lambda x: x[5])
    r_bif, z_bif, a_bif, phi_bif, ok, final_res = best

    print(f"\n  {'=' * 50}")
    print(f"  Best: converged={ok}  |G|={final_res:.4e}  a={a_bif:.8f}")
    print(f"  {'=' * 50}")
    return r_bif, z_bif, a_bif, phi_bif, ok


def newton_root_refine_hat(r_off_hat_init, z_off_hat_init, a_hat, shared_data, *, tol=1e-12, max_iter=15):

    print("\n" + "=" * 65)
    print(f"  Newton Root Refinement (a_hat = {a_hat:.6f})")
    print("=" * 65)

    R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_2d, p_2d = shared_data

    md = setup_moving_mesh_hat(r_off_hat_init, z_off_hat_init, a_hat, R_hat, H_hat, W_hat, Re, G_hat, U_m_hat, u_2d, p_2d)

    dr, dz = 0.0, 0.0

    for k in range(max_iter):
        F, J_full = evaluate_forces_hat(dr, dz, 0.0, md)
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
                F_try = evaluate_forces_hat(
                    dr + alpha * dx_n[0], dz + alpha * dx_n[1], 0.0,
                    md, compute_jacobian=False)
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

    raise RuntimeError(
        f"Newton refinement did not converge at a_hat = {a_hat} "
        f"after {max_iter} iterations (|F| = {res:.4e})."
    )


if __name__ == "__main__":

    # Initial guess from the bifurcation diagramm
    r_off_hat_init = 0.6098
    z_off_hat_init = 0.0274
    a_hat_start = 0.1375


    R_hat, H_hat, W_hat, L_c, U_c, Re = first_nondimensionalisation(R, H, W, Q, rho, mu, print_values=True)

    print("\nparticle_maxh_rel = ", particle_maxh_rel)

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)

    G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat = bg.solve_2D_background_flow()

    shared_data = (R_hat, H_hat, W_hat, L_c, U_c, Re, G_hat, U_m_hat, u_bar_2d_hat, p_bar_tilde_2d_hat)

    # refine initial guess to ensure starting MS with a root
    r_hat, z_hat, md_newton, dr_newton, dz_newton = newton_root_refine_hat(r_off_hat_init, z_off_hat_init, a_hat_start, shared_data,
                                                                            tol=1e-12, max_iter=15)

    # ══════════════════════════════════════════════════════════════════
    #  DIAGNOSTIC MOORE-SPENCE: manual iteration with full diagnostics
    #  Remove this block and replace with moore_spence_solve_hat_ad()
    #  once convergence is understood.
    # ══════════════════════════════════════════════════════════════════

    md = md_newton
    r_ref = r_hat - dr_newton
    z_ref = z_hat - dz_newton
    a_ref = a_hat_start

    F0, J0_full = evaluate_forces_hat(dr_newton, dz_newton, 0.0, md)
    eigpairs = estimate_eigenvectors(J0_full[:, :2])

    for trial_idx, (mu_start, phi_ev) in enumerate(eigpairs):
        l_vec = phi_ev.copy()
        phi = phi_ev / np.dot(l_vec, phi_ev)
        dr, dz, da = float(dr_newton), float(dz_newton), 0.0

        print("\n" + "=" * 70)
        print(f"  DIAGNOSTIC MS — trial {trial_idx+1}/{len(eigpairs)}"
              f"  mu={mu_start:+.4e}  phi=({phi_ev[0]:+.4f}, {phi_ev[1]:+.4f})")
        print("=" * 70)

        MS_MAX_ITER = 10
        Delta = 1e-2
        for k in range(MS_MAX_ITER):
            F_base, J_full, dJphi_dx = evaluate_forces_jac_hessian_hat(dr, dz, da, md, phi)
            J_sp = J_full[:, :2]
            dF_da = J_full[:, 2]
            G1 = F_base
            G2 = J_sp @ phi
            G3_val = np.dot(l_vec, phi) - 1.0
            G = np.concatenate([G1, G2, [G3_val]])
            res = np.linalg.norm(G)
            r_cur, z_cur, a_cur = r_ref + dr, z_ref + dz, a_ref + da

            print(f"\n  Iter {k:2d} | r={r_cur:+.8f} z={z_cur:+.8f} a={a_cur:.8f}")
            print(f"         | |G|={res:.4e}  F_x={G1[0]:+.4e}  F_z={G1[1]:+.4e}")
            print(f"         | (J*phi)_x={G2[0]:+.4e}  (J*phi)_z={G2[1]:+.4e}")

            dr_n, dz_n = dr, dz
            for nit in range(10):
                F_n, J_n = evaluate_forces_hat(dr_n, dz_n, da, md)
                if np.linalg.norm(F_n) < 1e-12: break
                try:
                    dx = np.linalg.solve(J_n[:, :2], -F_n)
                    dr_n += dx[0]; dz_n += dx[1]
                except np.linalg.LinAlgError: break
            print(f"         | DIAG1: Newton root at a={a_cur:.6f}: "
                  f"r={r_ref+dr_n:+.8f} z={z_ref+dz_n:+.8f}  "
                  f"|F|={np.linalg.norm(F_n):.4e}  "
                  f"dist=({dr_n-dr:+.4e}, {dz_n-dz:+.4e})")

            eig_vals, eig_vecs = np.linalg.eig(J_sp)
            idx_sort = np.argsort(np.abs(eig_vals))
            v_null = eig_vecs[:, idx_sort[0]].real
            v_null = v_null / np.linalg.norm(v_null)
            cos_ang = min(abs(np.dot(phi / np.linalg.norm(phi), v_null)), 1.0)
            angle_deg = np.degrees(np.arccos(cos_ang))
            print(f"         | DIAG2: eigs(J_sp) = [{eig_vals[idx_sort[0]].real:+.4e}, "
                  f"{eig_vals[idx_sort[1]].real:+.4e}]")
            print(f"         |        v_null = ({v_null[0]:+.4f}, {v_null[1]:+.4f})  "
                  f"phi = ({phi[0]:+.4f}, {phi[1]:+.4f})  "
                  f"angle(phi, v_null) = {angle_deg:.2f} deg")

            try:
                newton_step_F = np.linalg.solve(J_sp, -F_base)
                print(f"         | DIAG4: F-Newton step: dr={newton_step_F[0]:+.4e} "
                      f"dz={newton_step_F[1]:+.4e} |step|={np.linalg.norm(newton_step_F):.4e}")
            except np.linalg.LinAlgError:
                print(f"         | DIAG4: F-Newton step: J_sp singular")

            DG = np.zeros((5, 5))
            DG[0:2, 0:2] = J_sp;  DG[0:2, 2] = dF_da
            DG[2:4, 0] = dJphi_dx[:, 0];  DG[2:4, 1] = dJphi_dx[:, 1];  DG[2:4, 2] = dJphi_dx[:, 2]
            DG[2:4, 3:5] = J_sp;  DG[4, 3:5] = l_vec


            D = np.ones(5)
            DG_s = DG / D[np.newaxis, :]

            print(f"         | cond(DG)={np.linalg.cond(DG):.2e}  "
                  f"cond(DG_s)={np.linalg.cond(DG_s):.2e}")

            try:
                dy = np.linalg.solve(DG_s, -G) / D
            except np.linalg.LinAlgError:
                print("  !! DG singular"); break

            print(f"         | MS step: dr={dy[0]:+.4e} dz={dy[1]:+.4e} "
                  f"da={dy[2]:+.4e} dphi=({dy[3]:+.4e}, {dy[4]:+.4e})")

            print(f"         | DG matrix:")
            for row in range(5):
                print(f"         |   [{' '.join(f'{DG[row,j]:+10.4e}' for j in range(5))}]")

            Delta_max = 1.0
            eta_accept, eta_good = 0.1, 0.75

            def _eval_step_diag(dy_c):
                phi_t = phi + dy_c[3:5]
                if a_ref + da + dy_c[2] <= 0: return float('nan'), False, None
                try:
                    F_t, J_t = evaluate_forces_hat(dr+dy_c[0], dz+dy_c[1], da+dy_c[2], md)
                    G_t = np.concatenate([F_t, J_t[:,:2] @ phi_t, [np.dot(l_vec, phi_t)-1.0]])
                    return float(np.linalg.norm(G_t)), True, G_t
                except MeshFlippedError:
                    return float('nan'), False, None

            step_norm = np.linalg.norm(dy[:3])
            if step_norm > Delta:
                dy_try = dy * (Delta / step_norm)
                print(f"         | TR: clipped {step_norm:.4e} -> {Delta:.4e}")
            else:
                dy_try = dy.copy()

            G_pred = G + DG @ dy_try
            print(f"         | DIAG5: pred G = [F_x={G_pred[0]:+.4e} F_z={G_pred[1]:+.4e} "
                  f"Jphi_x={G_pred[2]:+.4e} Jphi_z={G_pred[3]:+.4e} norm={G_pred[4]:+.4e}]")

            pred = res**2 - np.linalg.norm(G_pred)**2
            res_try, step_ok, G_try = _eval_step_diag(dy_try)
            rho = (res**2 - res_try**2) / pred if (step_ok and abs(pred) > 1e-30) else -1.0

            if G_try is not None:
                print(f"         | actual G = [F_x={G_try[0]:+.4e} F_z={G_try[1]:+.4e} "
                      f"Jphi_x={G_try[2]:+.4e} Jphi_z={G_try[3]:+.4e} norm={G_try[4]:+.4e}]")
            print(f"         | |G_try|={res_try if step_ok else float('nan'):.4e} vs |G|={res:.4e}"
                  f"  rho={rho:+.4f}  Delta={Delta:.4e}")

            accepted = rho >= eta_accept and step_ok

            if not accepted:
                lam = 1e-6 * np.linalg.norm(DG_s)**2
                for _ in range(8):
                    try:
                        dy_lm = np.linalg.solve(DG_s.T @ DG_s + lam*np.eye(5), -DG_s.T @ G) / D
                    except np.linalg.LinAlgError:
                        lam *= 10; continue
                    res_lm, ok_lm, _ = _eval_step_diag(dy_lm)
                    if ok_lm and res_lm < res:
                        print(f"         | LM(lam={lam:.2e}): |G|={res_lm:.4e} ACCEPTED")
                        dy_try = dy_lm; accepted = True; break
                    print(f"         | LM(lam={lam:.2e}): |G|={res_lm:.4e} reject")
                    lam *= 10

            if accepted:
                dr += dy_try[0]; dz += dy_try[1]; da += dy_try[2]
                phi_new = phi + dy_try[3:5]
                phi = phi_new / np.dot(l_vec, phi_new)
                print(f"         | ACCEPTED")
                # DIAG 6
                print(f"         | DIAG6: phi=({phi[0]:+.6f}, {phi[1]:+.6f})")
            else:
                print(f"         | REJECTED")

            if rho < 0.25:
                Delta = max(Delta * 0.25, 1e-8)
            elif rho > eta_good and step_norm >= 0.95 * Delta:
                Delta = min(Delta * 2.0, Delta_max)

        print(f"\n  End trial {trial_idx+1}")
        print("=" * 70)