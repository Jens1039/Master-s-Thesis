import os
os.environ["OMP_NUM_THREADS"] = "1"

import math

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, taylor_test
from pyadjoint import Block, AdjFloat, get_working_tape, ReducedFunctional, Control, Tape, set_working_tape, continue_annotation

from nondimensionalization import second_nondimensionalisation
from background_flow_return_UFL import background_flow_differentiable, build_3d_background_flow_differentiable
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole


class NumpyLinSolveBlock(Block):

    def __init__(self, A_bvs, b_bvs, x_fns, n):
        super().__init__()
        self.n = n
        for bv in A_bvs:
            self.add_dependency(bv)
        for bv in b_bvs:
            self.add_dependency(bv)
        for fn in x_fns:
            self.add_output(fn.create_block_variable())

    # ---- helpers ----
    def _unpack(self, inputs):
        n = self.n
        A = np.array([float(v) for v in inputs[: n * n]]).reshape(n, n)
        b = np.array([float(v) for v in inputs[n * n :]])
        return A, b

    # ---- forward ----
    def recompute_component(self, inputs, block_variable, idx, prepared):
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)
        out = block_variable.output
        with stop_annotating():
            out.dat.data[:] = x[idx]
        return out

    # ---- adjoint ----
    def evaluate_adj_component(self, inputs, adj_inputs,
                               block_variable, idx, prepared=None):
        n = self.n
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)

        # Collect adjoint seeds from the n R-space Function outputs.
        # adj_inputs[k] is a Cofunction on the R-space (or None).
        adj_x = np.zeros(n)
        for k in range(n):
            ai = adj_inputs[k]
            if ai is not None:
                adj_x[k] = float(ai.dat.data_ro[0])

        mu = np.linalg.solve(A.T, adj_x)

        if idx < n * n:
            i, j = divmod(idx, n)
            return AdjFloat(-mu[i] * x[j])
        else:
            k = idx - n * n
            return AdjFloat(mu[k])

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        n = self.n
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)

        # Tangent of A and b
        dA = np.zeros((n, n))
        db = np.zeros(n)
        for k in range(n * n):
            if tlm_inputs[k] is not None:
                dA[k // n, k % n] = float(tlm_inputs[k])
        for k in range(n):
            if tlm_inputs[n * n + k] is not None:
                db[k] = float(tlm_inputs[n * n + k])

        # dx = A^{-1} (db - dA @ x)
        dx = np.linalg.solve(A, db - dA @ x)
        out = block_variable.output
        with stop_annotating():
            result = Function(out.function_space())
            result.dat.data[:] = dx[idx]
        return result


def numpy_lin_solve_to_R(A_adj_floats, b_adj_floats, R_space, n):

    A_np = np.array([float(a) for a in A_adj_floats]).reshape(n, n)
    b_np = np.array([float(b) for b in b_adj_floats])
    x_np = np.linalg.solve(A_np, b_np)

    x_fns = []
    for k in range(n):
        f = Function(R_space, name=f"linsys_x_{k}")
        with stop_annotating():
            f.dat.data[:] = x_np[k]
        x_fns.append(f)

    if annotate_tape():
        block = NumpyLinSolveBlock(
            A_adj_floats,
            b_adj_floats,
            x_fns, n,
        )
        get_working_tape().add_block(block)

    return x_fns


def stokes_solve(V, Q, particle_bc_expr, tags, mesh):

    W = V * Q
    v_trial, p_trial = TrialFunctions(W)
    v_test, q_test = TestFunctions(W)

    a_form = (
        2 * inner(sym(grad(v_trial)), sym(grad(v_test))) * dx
        - p_trial * div(v_test) * dx
        + q_test * div(v_trial) * dx
    )
    L_form = inner(Constant((0.0, 0.0, 0.0)), v_test) * dx  # zero RHS

    bcs = [
        DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), tags["walls"]),
        DirichletBC(W.sub(0), particle_bc_expr, tags["particle"]),
    ]

    nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True, comm=W.comm)]
    )

    w = Function(W)
    solve(
        a_form == L_form, w, bcs=bcs,
        nullspace=nullspace,
        solver_parameters={
                            "ksp_type": "preonly",
                            "pc_type": "lu",
                            "pc_factor_mat_solver_type": "mumps",
                            "mat_mumps_icntl_24": 1,
                            "mat_mumps_icntl_25": 0,
                            },
    )

    v_out = Function(V, name="v_stokes")
    p_out = Function(Q, name="p_stokes")
    v_out.assign(w.subfunctions[0])
    p_out.assign(w.subfunctions[1])

    return v_out, p_out


class perturbed_flow_differentiable:

    def __init__(self, R, H, W, L, a, Re_p, mesh3d, tags, u_bar_3d, p_bar_3d, X_ref, xi, u_cyl_3d):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.L = L
        self.Re_p = Re_p

        self.u_bar = u_bar_3d
        self.p_bar = p_bar_3d

        self.mesh3d = mesh3d
        self.tags = tags

        self.x = SpatialCoordinate(self.mesh3d)
        self.x_p = Constant(self.tags["particle_center"])

        self.e_x_prime = Constant([math.cos(self.L / self.R * 0.5), math.sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-math.sin(self.L / self.R * 0.5), math.cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        r_xy = sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.e_r_prime = as_vector([self.x[0] / r_xy, self.x[1] / r_xy, 0])
        self.e_theta_prime = as_vector([-self.x[1] / r_xy, self.x[0] / r_xy, 0])

        # --- X_ref + xi für Dirichlet-BCs (on-tape Abhängigkeit) ---
        self.x_bc = X_ref + xi

        r_xy_bc = sqrt(self.x_bc[0] ** 2 + self.x_bc[1] ** 2)
        self.e_r_bc = as_vector([self.x_bc[0] / r_xy_bc, self.x_bc[1] / r_xy_bc, 0])
        self.e_theta_bc = as_vector([-self.x_bc[1] / r_xy_bc, self.x_bc[0] / r_xy_bc, 0])

        # u_bar via x_bc für BCs
        self.u_bar_bc = as_vector([
            self.x_bc[0] / r_xy_bc * u_cyl_3d[0] - self.x_bc[1] / r_xy_bc * u_cyl_3d[2],
            self.x_bc[1] / r_xy_bc * u_cyl_3d[0] + self.x_bc[0] / r_xy_bc * u_cyl_3d[2],
            u_cyl_3d[1],
        ])

        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.R_space = FunctionSpace(self.mesh3d, "R", 0)

        self.u_cyl_3d = u_cyl_3d


    def _solve_stokes(self, particle_bc_expr):
        return stokes_solve(
            self.V, self.Q, particle_bc_expr, self.tags, self.mesh3d)


    def _force_components(self, v, q):
        n = FacetNormal(self.mesh3d)
        sigma = -q * Identity(3) + (grad(v) + grad(v).T)
        traction = dot(-n, sigma)
        return [assemble(traction[i] * ds(self.tags["particle"]))
                for i in range(3)]


    def _torque_components(self, v, q):
        n = FacetNormal(self.mesh3d)
        sigma = -q * Identity(3) + (grad(v) + grad(v).T)
        traction = dot(-n, sigma)
        moment = cross(self.x - self.x_p, traction)    # SpatialCoordinate OK in ds
        return [assemble(moment[i] * ds(self.tags["particle"]))
                for i in range(3)]


    @staticmethod
    def _dot3(e_np, comps):
        return float(e_np[0]) * comps[0] + float(e_np[1]) * comps[1] + float(e_np[2]) * comps[2]


    def F_p(self):

        V = self.V
        x_bc = self.x_bc

        u_bar_a = dot(self.u_bar, self.e_theta_prime) * self.e_theta_prime
        u_bar_s = (dot(self.u_bar, self.e_r_prime) * self.e_r_prime
                   + dot(self.u_bar, self.e_z_prime) * self.e_z_prime)

        u_bar_a_bc = dot(self.u_bar_bc, self.e_theta_bc) * self.e_theta_bc
        u_bar_s_bc = (dot(self.u_bar_bc, self.e_r_bc) * self.e_r_bc
                      + dot(self.u_bar_bc, self.e_z_prime) * self.e_z_prime)

        bc_Theta = Function(V, name="bc_Theta")
        bc_Theta.interpolate(cross(self.e_z_prime, x_bc))

        bc_Ox = Function(V, name="bc_Ox")
        bc_Ox.interpolate(cross(self.e_x_prime, x_bc - self.x_p))

        bc_Oy = Function(V, name="bc_Oy")
        bc_Oy.interpolate(cross(self.e_y_prime, x_bc - self.x_p))

        bc_Oz = Function(V, name="bc_Oz")
        bc_Oz.interpolate(cross(self.e_z_prime, x_bc - self.x_p))

        bc_bg = Function(V, name="bc_bg")
        bc_bg.interpolate(-u_bar_a_bc)

        v_Theta, q_Theta = self._solve_stokes(bc_Theta)
        v_Ox,    q_Ox    = self._solve_stokes(bc_Ox)
        v_Oy,    q_Oy    = self._solve_stokes(bc_Oy)
        v_Oz,    q_Oz    = self._solve_stokes(bc_Oz)
        v_bg,    q_bg    = self._solve_stokes(bc_bg)

        F_Theta = self._force_components(v_Theta, q_Theta)
        F_Ox    = self._force_components(v_Ox,    q_Ox)
        F_Oy    = self._force_components(v_Oy,    q_Oy)
        F_Oz    = self._force_components(v_Oz,    q_Oz)
        F_bg    = self._force_components(v_bg,    q_bg)

        T_Theta = self._torque_components(v_Theta, q_Theta)
        T_Ox    = self._torque_components(v_Ox,    q_Ox)
        T_Oy    = self._torque_components(v_Oy,    q_Oy)
        T_Oz    = self._torque_components(v_Oz,    q_Oz)
        T_bg    = self._torque_components(v_bg,    q_bg)

        e_x = np.array(self.e_x_prime.values())
        e_y = np.array(self.e_y_prime.values())
        e_z = np.array(self.e_z_prime.values())

        d = self._dot3

        A_flat = [
            d(e_y, F_Theta), d(e_y, F_Oz), d(e_y, F_Ox), d(e_y, F_Oy),
            d(e_x, T_Theta), d(e_x, T_Oz), d(e_x, T_Ox), d(e_x, T_Oy),
            d(e_y, T_Theta), d(e_y, T_Oz), d(e_y, T_Ox), d(e_y, T_Oy),
            d(e_z, T_Theta), d(e_z, T_Oz), d(e_z, T_Ox), d(e_z, T_Oy),
        ]

        b_flat = [
            -d(e_y, F_bg),
            -d(e_x, T_bg),
            -d(e_y, T_bg),
            -d(e_z, T_bg),
        ]

        Theta_fn, Omega_z_fn, Omega_x_fn, Omega_y_fn = \
            numpy_lin_solve_to_R(A_flat, b_flat, self.R_space, 4)

        v_0_a = Function(self.V, name="v_0_a")
        v_0_a.interpolate(
            Theta_fn   * v_Theta
            + Omega_x_fn * v_Ox
            + Omega_y_fn * v_Oy
            + Omega_z_fn * v_Oz
            + v_bg
        )

        bc_sym = Function(V, name="bc_sym")
        bc_sym.interpolate(-u_bar_s_bc)
        v_0_s, q_0_s = self._solve_stokes(bc_sym)

        bc_ex = Function(V, name="bc_ex")
        bc_ex.interpolate(self.e_x_prime)
        bc_ez = Function(V, name="bc_ez")
        bc_ez.interpolate(self.e_z_prime)
        u_hat_x, _ = self._solve_stokes(bc_ex)
        u_hat_z, _ = self._solve_stokes(bc_ez)

        F_s = self._force_components(v_0_s, q_0_s)
        F_s_x = d(e_x, F_s)
        F_s_z = d(e_z, F_s)

        x_p_np = np.array(self.x_p.values())
        cent_vec = np.cross(e_z, np.cross(e_z, x_p_np))
        cent_coeff_x = float(np.dot(e_x, cent_vec))
        cent_coeff_z = float(np.dot(e_z, cent_vec))

        vol = assemble(Constant(1.0) * dx(domain=self.mesh3d))
        neg4pi3 = Constant(-4.0 * np.pi / 3.0)

        centrifugal_x = (
                assemble(neg4pi3 * Constant(cent_coeff_x) * Theta_fn * Theta_fn * dx)
                / vol
        )
        centrifugal_z = (
                assemble(neg4pi3 * Constant(cent_coeff_z) * Theta_fn * Theta_fn * dx)
                / vol
        )

        n_hat = FacetNormal(self.mesh3d)
        inertial_integrand = (
            dot(u_bar_a, -n_hat) * u_bar_a
            + dot(u_bar_s, -n_hat) * u_bar_a
            + dot(u_bar_a, -n_hat) * u_bar_s
            + dot(u_bar_s, -n_hat) * u_bar_s
        )
        inertial_x = assemble(
            dot(self.e_x_prime, inertial_integrand)
            * ds(self.tags["particle"], degree=6)
        )
        inertial_z = assemble(
            dot(self.e_z_prime, inertial_integrand)
            * ds(self.tags["particle"], degree=6)
        )

        rhs = (
            cross(Theta_fn * self.e_z_prime, v_0_a)
            + dot(grad(u_bar_a), v_0_a)
            + dot(grad(v_0_a), v_0_a + u_bar_a
                  - cross(Theta_fn * self.e_z_prime, self.x))
            + dot(grad(u_bar_s), v_0_s)
            + dot(grad(v_0_s), v_0_s + u_bar_s)
            + cross(Theta_fn * self.e_z_prime, v_0_s)
            - dot(grad(v_0_s), cross(Theta_fn * self.e_z_prime, self.x))
            + dot(grad(u_bar_s), v_0_a)
            + dot(grad(u_bar_a), v_0_s)
            + dot(grad(v_0_s), v_0_a + u_bar_a)
            + dot(grad(v_0_a), v_0_s + u_bar_s)
        )

        fluid_stress_x = assemble(-dot(u_hat_x, rhs) * dx(degree=6))
        fluid_stress_z = assemble(-dot(u_hat_z, rhs) * dx(degree=6))

        inv_Re_p = float(1.0 / self.Re_p)

        F_p_x = (inv_Re_p * F_s_x
                  + fluid_stress_x + inertial_x + centrifugal_x)
        F_p_z = (inv_Re_p * F_s_z
                  + fluid_stress_z + inertial_z + centrifugal_z)

        self._debug = {}



        return F_p_x, F_p_z



if __name__ == "__main__":

    R_hat = 500
    H_hat = 2
    W_hat = 2
    a_hat = 0.135
    Re = 1.0
    L_c = H_hat / 2
    U_c = 0.008366733466944444

    set_working_tape(Tape())
    continue_annotation()

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_val, U_m_hat, u_bar, p_bar_tilde = bg.solve_2D_background_flow()

    from background_flow import background_flow, build_3d_background_flow

    bg_2 = background_flow(R_hat, H_hat, W_hat, Re)
    G_val_2, U_m_hat_2, u_bar_2, p_bar_tilde_2 = bg.solve_2D_background_flow()

    R_hat_hat, H_hat_hat, W_hat_hat, a_hat_hat, G_hat_hat, L_c_p, U_c_p, u_bar_2d_hat_hat, p_bar_2d_hat_hat, Re_p = second_nondimensionalisation(
        R_hat, H_hat, W_hat, a_hat, L_c, U_c, G_val, Re, u_bar, p_bar_tilde, U_m_hat, print_values=False)

    L_hat_hat = 4 * max(H_hat_hat, W_hat_hat)
    particle_maxh_hat_hat = 0.2 * a_hat_hat
    global_maxh_hat_hat = 0.2 * min(H_hat_hat, W_hat_hat)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
            R_hat_hat, H_hat_hat, W_hat_hat, L_hat_hat, a_hat_hat,
            particle_maxh_hat_hat, global_maxh_hat_hat, r_off=-4, z_off=2)

    R_space = FunctionSpace(mesh3d, "R", 0)
    delta_r = Function(R_space, name="delta_r").assign(0.0)
    delta_z = Function(R_space, name="delta_z").assign(0.0)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)
    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    x3d = SpatialCoordinate(mesh3d)
    dist = sqrt((X_ref[0] - cx) ** 2 + (X_ref[1] - cy) ** 2 + (X_ref[2] - cz) ** 2)

    r_cut = Constant(0.5 * min(H_hat_hat, W_hat_hat))
    bump = max_value(Constant(0.0), 1.0 - dist / r_cut)

    theta_half = tags["theta"] / 2.0
    cos_th = math.cos(theta_half)
    sin_th = math.sin(theta_half)

    xi = Function(V_def, name="xi")
    xi.interpolate(as_vector([
        delta_r * cos_th * bump,
        delta_r * sin_th * bump,
        delta_z * bump,
    ]))

    mesh3d.coordinates.assign(X_ref + xi)

    u_bar_3d, p_bar_3d, u_cyl_3d = build_3d_background_flow_differentiable(
        R_hat_hat, H_hat_hat, W_hat_hat, G_hat_hat, mesh3d, tags, u_bar_2d_hat_hat, p_bar_2d_hat_hat)

    c_r = Control(delta_r)
    c_z = Control(delta_z)
    h_r = Function(R_space).assign(0.01)
    h_z = Function(R_space).assign(0.01)
    h = [h_r, h_z]
    m0 = [Function(R_space).assign(0.0), Function(R_space).assign(0.0)]

    pf = perturbed_flow_differentiable(
        R_hat_hat, H_hat_hat, W_hat_hat, L_hat_hat, a_hat_hat, Re_p,
        mesh3d, tags, u_bar_3d, p_bar_3d,
        X_ref, xi, u_cyl_3d)

    F_p_x, F_p_z = pf.F_p()

    from perturbed_flow import perturbed_flow

    with stop_annotating():

        u_bar_3d_2, p_bar_3d_2 = build_3d_background_flow(
        R_hat_hat, H_hat_hat, W_hat_hat, G_hat_hat, mesh3d, tags, u_bar_2d_hat_hat, p_bar_2d_hat_hat)

        pf_2 = perturbed_flow(
        R_hat_hat, H_hat_hat, W_hat_hat, L_hat_hat, a_hat_hat, Re_p,
        mesh3d, tags, u_bar_3d, p_bar_3d)

        F_p_x_2, F_p_z_2 = pf_2.F_p()

    print("max difference between tape-safe and verified F_p:", np.max(np.abs([F_p_x - F_p_x_2, F_p_z - F_p_z_2])))

    J_x = ReducedFunctional(F_p_x, [c_r, c_z])
    J_z = ReducedFunctional(F_p_z, [c_r, c_z])

    dFx = J_x.derivative()
    dFx_dr = float(assemble(action(dFx[0], Function(R_space).assign(1.0))))
    dFx_dz = float(assemble(action(dFx[1], Function(R_space).assign(1.0))))
    print(f"  dF_p_x/d(delta_r) = {dFx_dr}")
    print(f"  dF_p_x/d(delta_z) = {dFx_dz}")

    dFz = J_z.derivative()
    dFz_dr = float(assemble(action(dFz[0], Function(R_space).assign(1.0))))
    dFz_dz = float(assemble(action(dFz[1], Function(R_space).assign(1.0))))
    print(f"  dF_p_z/d(delta_r) = {dFz_dr}")
    print(f"  dF_p_z/d(delta_z) = {dFz_dz}")

    taylor_test(J_x, m0, h)