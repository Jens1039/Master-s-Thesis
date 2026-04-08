import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import ufl

from firedrake import *
from firedrake.adjoint import stop_annotating, annotate_tape, taylor_test
from pyadjoint import Block, AdjFloat, get_working_tape, ReducedFunctional, Control, Tape, set_working_tape, continue_annotation
from pyadjoint.verification import convergence_rates
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock, solve_init_params

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


    def _unpack(self, inputs):
        n = self.n
        A = np.array([float(v) for v in inputs[: n * n]]).reshape(n, n)
        b = np.array([float(v) for v in inputs[n * n :]])
        return A, b


    def recompute_component(self, inputs, block_variable, idx, prepared):
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)
        out = block_variable.output
        with stop_annotating():
            out.dat.data[:] = x[idx]
        return out


    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):

        n = self.n
        A, b = self._unpack(inputs)
        x = np.linalg.solve(A, b)
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
        dA = np.zeros((n, n))
        db = np.zeros(n)
        for k in range(n * n):
            if tlm_inputs[k] is not None:
                dA[k // n, k % n] = float(tlm_inputs[k])
        for k in range(n):
            if tlm_inputs[n * n + k] is not None:
                db[k] = float(tlm_inputs[n * n + k])
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
        block = NumpyLinSolveBlock(A_adj_floats, b_adj_floats, x_fns, n)
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
    L_form = inner(Constant((0.0, 0.0, 0.0)), v_test) * dx
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
            "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "mat_mumps_icntl_24": 1, "mat_mumps_icntl_25": 0,
        },
    )
    v_out = Function(V, name="v_stokes")
    p_out = Function(Q, name="p_stokes")
    v_out.assign(w.subfunctions[0])
    p_out.assign(w.subfunctions[1])
    return v_out, p_out


_STOKES_SP = {
    "ksp_type": "preonly", "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_24": 1, "mat_mumps_icntl_25": 0,
}


class CachedStokesContext:

    def __init__(self, V, Q, tags, mesh):
        self.V = V
        self.Q = Q
        self.W = V * Q
        self.tags = tags
        self.mesh = mesh

        v_trial, p_trial = TrialFunctions(self.W)
        v_test, q_test = TestFunctions(self.W)

        self.a_form = (
            2 * inner(sym(grad(v_trial)), sym(grad(v_test))) * dx
            - p_trial * div(v_test) * dx
            + q_test * div(v_trial) * dx
        )
        self.L_form = inner(Constant((0.0, 0.0, 0.0)), v_test) * dx

        self.nullspace = MixedVectorSpaceBasis(
            self.W,
            [self.W.sub(0), VectorSpaceBasis(constant=True, comm=self.W.comm)],
        )
        self.bcs_hom = [
            DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), tags["walls"]),
            DirichletBC(self.W.sub(0), Constant((0.0, 0.0, 0.0)), tags["particle"]),
        ]

        self._fwd_solver = None
        self._adj_solver = None
        self._fp = None
        self.fwd_factor_count = 0
        self.adj_factor_count = 0

    def _fingerprint(self):
        c = self.mesh.coordinates.dat.data_ro.ravel()
        n = len(c)
        return c[0] + c[n // 4] + c[n // 2] + c[-1]

    def _invalidate(self):
        self._fwd_solver = None
        self._adj_solver = None

    def get_fwd_solver(self):
        fp = self._fingerprint()
        if self._fwd_solver is None or abs(fp - self._fp) > 1e-14:
            A = assemble(self.a_form, bcs=self.bcs_hom)
            self._fwd_solver = LinearSolver(
                A, nullspace=self.nullspace, solver_parameters=_STOKES_SP)
            self._adj_solver = None
            self._fp = fp
            self.fwd_factor_count += 1
        return self._fwd_solver

    def get_adj_solver(self):
        self.get_fwd_solver()
        if self._adj_solver is None:
            a_adj = adjoint(self.a_form)
            B = assemble(a_adj, bcs=self.bcs_hom)
            self._adj_solver = LinearSolver(
                B, nullspace=self.nullspace, solver_parameters=_STOKES_SP)
            self.adj_factor_count += 1
        return self._adj_solver

    def _lift_rhs(self, a_form, L_form, bcs):

        W = self.W
        w_lift = Function(W)
        for bc in bcs:
            bc.apply(w_lift)

        b = assemble(L_form - action(a_form, w_lift), bcs=self.bcs_hom)

        return b, w_lift


class CachedStokesSolveBlock(GenericSolveBlock):

    def __init__(self, lhs, rhs, func, bcs, ctx, *args, **kwargs):
        super().__init__(lhs, rhs, func, bcs, *args, **kwargs)
        self.ctx = ctx

    def _init_solver_parameters(self, args, kwargs):
        super()._init_solver_parameters(args, kwargs)
        solve_init_params(self, args, kwargs, varform=True)

    def _forward_solve(self, lhs, rhs, func, bcs):
        b, w_lift = self.ctx._lift_rhs(lhs, rhs, bcs)
        solver = self.ctx.get_fwd_solver()
        solver.solve(func, b)
        func += w_lift
        return func


    def _assemble_and_solve_adj_eq(self, dFdu_adj_form, dJdu, compute_bdy):
        dJdu_copy = dJdu.copy()

        # Adjoint BCs are homogeneous — zero out BC DOFs in the adjoint RHS.
        # bc.apply() fails on Cofunctions (dual space != primal space),
        # so zero the velocity DOFs directly via the underlying data array.
        for bc in self.ctx.bcs_hom:
            dJdu.dat[0].data_wo[bc.nodes] = 0.0

        adj_solver = self.ctx.get_adj_solver()
        adj_sol = Function(self.function_space)
        adj_solver.solve(adj_sol, dJdu)

        adj_sol_bdy = None
        if compute_bdy:
            adj_sol_bdy = self._compute_adj_bdy(
                adj_sol, adj_sol_bdy, dFdu_adj_form, dJdu_copy)
        return adj_sol, adj_sol_bdy


    def _assemble_and_solve_tlm_eq(self, dFdu, dFdm, dudm, bcs):
        solver = self.ctx.get_fwd_solver()
        for bc in bcs:
            bc.apply(dFdm)
            bc.apply(dudm)
        solver.solve(dudm, dFdm)
        return dudm


def stokes_solve_cached(V, Q, particle_bc_expr, tags, mesh, ctx):
    
    W = ctx.W
    a_form = ctx.a_form
    L_form = ctx.L_form

    bcs = [
        DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), tags["walls"]),
        DirichletBC(W.sub(0), particle_bc_expr, tags["particle"]),
    ]

    w = Function(W)


    with stop_annotating():
        b, w_lift = ctx._lift_rhs(a_form, L_form, bcs)
        solver = ctx.get_fwd_solver()
        solver.solve(w, b)
        w += w_lift


    if annotate_tape():
        block = CachedStokesSolveBlock(
            a_form, L_form, w, bcs, ctx,
            solver_parameters=_STOKES_SP,
        )
        get_working_tape().add_block(block)
        # GenericSolveBlock.__init__ may already register w as an output.
        # Only add_output if it was NOT already registered by super().__init__.
        if not any(bv.output is w for bv in block.get_outputs()):
            block.add_output(w.create_block_variable())

    v_out = Function(V, name="v_stokes")
    p_out = Function(Q, name="p_stokes")
    v_out.assign(w.subfunctions[0])
    p_out.assign(w.subfunctions[1])
    return v_out, p_out


class perturbed_flow_differentiable:

    def __init__(self, R, H, W, L, a, Re_p, mesh3d, tags, u_bar_3d,
                 p_bar_3d, X_ref, xi, u_cyl_3d):
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

        self.e_x_prime = Constant([math.cos(self.L / self.R * 0.5),
                                   math.sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-math.sin(self.L / self.R * 0.5),
                                   math.cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        r_xy = sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.e_r_prime = as_vector([self.x[0] / r_xy, self.x[1] / r_xy, 0])
        self.e_theta_prime = as_vector([-self.x[1] / r_xy,
                                        self.x[0] / r_xy, 0])

        self.x_bc = X_ref + xi
        r_xy_bc = sqrt(self.x_bc[0] ** 2 + self.x_bc[1] ** 2)
        self.e_r_bc = as_vector([self.x_bc[0] / r_xy_bc,
                                 self.x_bc[1] / r_xy_bc, 0])
        self.e_theta_bc = as_vector([-self.x_bc[1] / r_xy_bc,
                                     self.x_bc[0] / r_xy_bc, 0])
        self.u_bar_bc = as_vector([
            self.x_bc[0] / r_xy_bc * u_cyl_3d[0]
            - self.x_bc[1] / r_xy_bc * u_cyl_3d[2],
            self.x_bc[1] / r_xy_bc * u_cyl_3d[0]
            + self.x_bc[0] / r_xy_bc * u_cyl_3d[2],
            u_cyl_3d[1],
        ])

        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.R_space = FunctionSpace(self.mesh3d, "R", 0)
        self.u_cyl_3d = u_cyl_3d

        with stop_annotating():
            self.stokes_ctx = CachedStokesContext(
                self.V, self.Q, self.tags, self.mesh3d)

    def _solve_stokes(self, particle_bc_expr):
        return stokes_solve_cached(
            self.V, self.Q, particle_bc_expr, self.tags,
            self.mesh3d, self.stokes_ctx)

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
        moment = cross(self.x - self.x_p, traction)
        return [assemble(moment[i] * ds(self.tags["particle"]))
                for i in range(3)]

    @staticmethod
    def _dot3(e_np, comps):
        return (float(e_np[0]) * comps[0] + float(e_np[1]) * comps[1]
                + float(e_np[2]) * comps[2])

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
        v_Ox, q_Ox = self._solve_stokes(bc_Ox)
        v_Oy, q_Oy = self._solve_stokes(bc_Oy)
        v_Oz, q_Oz = self._solve_stokes(bc_Oz)
        v_bg, q_bg = self._solve_stokes(bc_bg)

        F_Theta = self._force_components(v_Theta, q_Theta)
        F_Ox = self._force_components(v_Ox, q_Ox)
        F_Oy = self._force_components(v_Oy, q_Oy)
        F_Oz = self._force_components(v_Oz, q_Oz)
        F_bg = self._force_components(v_bg, q_bg)

        T_Theta = self._torque_components(v_Theta, q_Theta)
        T_Ox = self._torque_components(v_Ox, q_Ox)
        T_Oy = self._torque_components(v_Oy, q_Oy)
        T_Oz = self._torque_components(v_Oz, q_Oz)
        T_bg = self._torque_components(v_bg, q_bg)

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
            -d(e_y, F_bg), -d(e_x, T_bg),
            -d(e_y, T_bg), -d(e_z, T_bg),
        ]

        Theta_fn, Omega_z_fn, Omega_x_fn, Omega_y_fn = \
            numpy_lin_solve_to_R(A_flat, b_flat, self.R_space, 4)

        v_0_a = Function(self.V, name="v_0_a")
        v_0_a.interpolate(
            Theta_fn * v_Theta + Omega_x_fn * v_Ox
            + Omega_y_fn * v_Oy + Omega_z_fn * v_Oz + v_bg)

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
            assemble(neg4pi3 * Constant(cent_coeff_x) * Theta_fn
                     * Theta_fn * dx) / vol)
        centrifugal_z = (
            assemble(neg4pi3 * Constant(cent_coeff_z) * Theta_fn
                     * Theta_fn * dx) / vol)

        n_hat = FacetNormal(self.mesh3d)
        inertial_integrand = (
            dot(u_bar_a, -n_hat) * u_bar_a
            + dot(u_bar_s, -n_hat) * u_bar_a
            + dot(u_bar_a, -n_hat) * u_bar_s
            + dot(u_bar_s, -n_hat) * u_bar_s
        )
        inertial_x = assemble(
            dot(self.e_x_prime, inertial_integrand)
            * ds(self.tags["particle"], degree=6))
        inertial_z = assemble(
            dot(self.e_z_prime, inertial_integrand)
            * ds(self.tags["particle"], degree=6))

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
        return F_p_x, F_p_z


def _banner(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _pass_fail(name, passed, detail=""):
    tag = "PASS" if passed else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    return passed


if __name__ == "__main__":

    results = []

    R_hat = 500; H_hat = 2; W_hat = 2; a_hat = 0.135
    Re = 1.0; L_c = H_hat / 2; U_c = 0.008366733466944444

    set_working_tape(Tape())
    continue_annotation()

    bg = background_flow_differentiable(R_hat, H_hat, W_hat, Re)
    G_val, U_m_hat, u_bar, p_bar_tilde = bg.solve_2D_background_flow()

    from background_flow import background_flow, build_3d_background_flow

    R_hh, H_hh, W_hh, a_hh, G_hh, L_c_p, U_c_p, u_2d_hh, p_2d_hh, Re_p = \
        second_nondimensionalisation(
            R_hat, H_hat, W_hat, a_hat, L_c, U_c,
            G_val, Re, u_bar, p_bar_tilde, U_m_hat, print_values=False)

    L_hh = 4 * max(H_hh, W_hh)
    particle_maxh = 0.2 * a_hh
    global_maxh = 0.2 * min(H_hh, W_hh)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R_hh, H_hh, W_hh, L_hh, a_hh,
        particle_maxh, global_maxh, r_off=-4, z_off=2)

    R_space = FunctionSpace(mesh3d, "R", 0)
    delta_r = Function(R_space, name="delta_r").assign(0.0)
    delta_z = Function(R_space, name="delta_z").assign(0.0)

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)
    with stop_annotating():
        X_ref = Function(V_def, name="X_ref")
        X_ref.interpolate(SpatialCoordinate(mesh3d))

    cx, cy, cz = tags["particle_center"]
    dist = sqrt((X_ref[0]-cx)**2 + (X_ref[1]-cy)**2 + (X_ref[2]-cz)**2)
    r_cut = Constant(0.5 * min(H_hh, W_hh))
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
        R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh)

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    V_stokes = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_stokes = FunctionSpace(mesh3d, "CG", 1)

    # =====================================================================
    #  TEST 1: Individual Stokes solves — cached vs uncached
    # =====================================================================

    _banner("TEST 1: INDIVIDUAL STOKES SOLVES (cached vs uncached)")

    ctx_test1 = CachedStokesContext(V_stokes, Q_stokes, tags, mesh3d)

    pf_tmp = perturbed_flow_differentiable(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
        mesh3d, tags, u_bar_3d, p_bar_3d, X_ref, xi, u_cyl_3d)
    x_bc = pf_tmp.x_bc

    bc_exprs = {}
    with stop_annotating():
        bc_exprs["bc_Theta"] = Function(V_stokes).interpolate(
            cross(pf_tmp.e_z_prime, x_bc))
        bc_exprs["bc_Ox"] = Function(V_stokes).interpolate(
            cross(pf_tmp.e_x_prime, x_bc - pf_tmp.x_p))
        bc_exprs["bc_Oy"] = Function(V_stokes).interpolate(
            cross(pf_tmp.e_y_prime, x_bc - pf_tmp.x_p))
        bc_exprs["bc_Oz"] = Function(V_stokes).interpolate(
            cross(pf_tmp.e_z_prime, x_bc - pf_tmp.x_p))

        u_bar_a_bc = dot(pf_tmp.u_bar_bc, pf_tmp.e_theta_bc) * pf_tmp.e_theta_bc
        u_bar_s_bc = (dot(pf_tmp.u_bar_bc, pf_tmp.e_r_bc) * pf_tmp.e_r_bc
                      + dot(pf_tmp.u_bar_bc, pf_tmp.e_z_prime) * pf_tmp.e_z_prime)
        bc_exprs["bc_bg"] = Function(V_stokes).interpolate(-u_bar_a_bc)
        bc_exprs["bc_sym"] = Function(V_stokes).interpolate(-u_bar_s_bc)
        bc_exprs["bc_ex"] = Function(V_stokes).interpolate(pf_tmp.e_x_prime)
        bc_exprs["bc_ez"] = Function(V_stokes).interpolate(pf_tmp.e_z_prime)

    test1_all_pass = True
    tol1 = 1e-10
    for name, bc_fn in bc_exprs.items():
        with stop_annotating():
            v_cached, q_cached = stokes_solve_cached(
                V_stokes, Q_stokes, bc_fn, tags, mesh3d, ctx_test1)
            v_uncached, q_uncached = stokes_solve(
                V_stokes, Q_stokes, bc_fn, tags, mesh3d)

            v_norm = norm(v_uncached, "L2")
            q_norm = norm(q_uncached, "L2")
            v_err = errornorm(v_cached, v_uncached, "L2")
            q_err = errornorm(q_cached, q_uncached, "L2")
            v_rel = v_err / v_norm if v_norm > 0 else v_err
            q_rel = q_err / q_norm if q_norm > 0 else q_err

            ok = v_rel < tol1 and q_rel < tol1
            test1_all_pass &= _pass_fail(
                f"{name:10s}", ok,
                f"v_rel={v_rel:.2e}, q_rel={q_rel:.2e}")

    results.append(("Test 1: Individual Stokes solves", test1_all_pass))

    # =====================================================================
    #  TEST 2: Forward F_p — cached vs reference class
    # =====================================================================

    _banner("TEST 2: FORWARD F_p (cached vs reference)")

    set_working_tape(Tape())
    continue_annotation()

    delta_r.assign(0.0)
    delta_z.assign(0.0)
    xi.interpolate(as_vector([
        delta_r * cos_th * bump,
        delta_r * sin_th * bump,
        delta_z * bump,
    ]))
    mesh3d.coordinates.assign(X_ref + xi)

    pf = perturbed_flow_differentiable(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
        mesh3d, tags, u_bar_3d, p_bar_3d, X_ref, xi, u_cyl_3d)
    F_p_x, F_p_z = pf.F_p()

    from perturbed_flow import perturbed_flow
    with stop_annotating():
        pf_ref = perturbed_flow(
            R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
            mesh3d, tags, u_bar_3d, p_bar_3d)
        F_ref_x, F_ref_z = pf_ref.F_p()

    tol2 = 1e-10
    rel_x = abs(float(F_p_x) - float(F_ref_x)) / abs(float(F_ref_x))
    rel_z = abs(float(F_p_z) - float(F_ref_z)) / abs(float(F_ref_z))
    print(f"  F_p_x  cached={float(F_p_x):+.12e}  ref={float(F_ref_x):+.12e}  rel={rel_x:.2e}")
    print(f"  F_p_z  cached={float(F_p_z):+.12e}  ref={float(F_ref_z):+.12e}  rel={rel_z:.2e}")
    test2_pass = _pass_fail("F_p_x", rel_x < tol2, f"rel={rel_x:.2e}")
    test2_pass &= _pass_fail("F_p_z", rel_z < tol2, f"rel={rel_z:.2e}")
    results.append(("Test 2: Forward F_p comparison", test2_pass))

    # =====================================================================
    #  TEST 3: AD Jacobian — cached vs uncached
    # =====================================================================

    _banner("TEST 3: AD JACOBIAN (cached vs uncached)")

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    J_x = ReducedFunctional(F_p_x, [c_r, c_z])
    J_z = ReducedFunctional(F_p_z, [c_r, c_z])

    dFx = J_x.derivative()
    dFx_dr = float(assemble(action(dFx[0], Function(R_space).assign(1.0))))
    dFx_dz = float(assemble(action(dFx[1], Function(R_space).assign(1.0))))
    dFz = J_z.derivative()
    dFz_dr = float(assemble(action(dFz[0], Function(R_space).assign(1.0))))
    dFz_dz = float(assemble(action(dFz[1], Function(R_space).assign(1.0))))

    print(f"  Cached AD:")
    print(f"    dF_x/dr={dFx_dr:+.10e}  dF_x/dz={dFx_dz:+.10e}")
    print(f"    dF_z/dr={dFz_dr:+.10e}  dF_z/dz={dFz_dz:+.10e}")

    # Uncached AD reference
    set_working_tape(Tape())
    continue_annotation()

    delta_r_uc = Function(R_space, name="delta_r_uc").assign(0.0)
    delta_z_uc = Function(R_space, name="delta_z_uc").assign(0.0)

    xi_uc = Function(V_def, name="xi_uc")
    xi_uc.interpolate(as_vector([
        delta_r_uc * cos_th * bump,
        delta_r_uc * sin_th * bump,
        delta_z_uc * bump,
    ]))
    mesh3d.coordinates.assign(X_ref + xi_uc)

    u_bar_3d_uc, p_bar_3d_uc, u_cyl_3d_uc = build_3d_background_flow_differentiable(
        R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh)

    from perturbed_flow_return_UFL import (
        perturbed_flow_differentiable as pf_diff_uncached)

    c_r_uc = Control(delta_r_uc)
    c_z_uc = Control(delta_z_uc)

    pf_uc = pf_diff_uncached(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
        mesh3d, tags, u_bar_3d_uc, p_bar_3d_uc, X_ref, xi_uc, u_cyl_3d_uc)
    F_uc_x, F_uc_z = pf_uc.F_p()

    J_x_uc = ReducedFunctional(F_uc_x, [c_r_uc, c_z_uc])
    J_z_uc = ReducedFunctional(F_uc_z, [c_r_uc, c_z_uc])

    dFx_uc = J_x_uc.derivative()
    dFx_dr_uc = float(assemble(action(dFx_uc[0], Function(R_space).assign(1.0))))
    dFx_dz_uc = float(assemble(action(dFx_uc[1], Function(R_space).assign(1.0))))
    dFz_uc = J_z_uc.derivative()
    dFz_dr_uc = float(assemble(action(dFz_uc[0], Function(R_space).assign(1.0))))
    dFz_dz_uc = float(assemble(action(dFz_uc[1], Function(R_space).assign(1.0))))

    print(f"  Uncached AD:")
    print(f"    dF_x/dr={dFx_dr_uc:+.10e}  dF_x/dz={dFx_dz_uc:+.10e}")
    print(f"    dF_z/dr={dFz_dr_uc:+.10e}  dF_z/dz={dFz_dz_uc:+.10e}")

    tol3 = 1e-6
    jac_cached = [dFx_dr, dFx_dz, dFz_dr, dFz_dz]
    jac_uncached = [dFx_dr_uc, dFx_dz_uc, dFz_dr_uc, dFz_dz_uc]
    jac_names = ["dF_x/dr", "dF_x/dz", "dF_z/dr", "dF_z/dz"]
    test3_pass = True
    for jn, jc, ju in zip(jac_names, jac_cached, jac_uncached):
        rel = abs(jc - ju) / abs(ju) if abs(ju) > 0 else abs(jc - ju)
        ok = rel < tol3
        test3_pass &= _pass_fail(jn, ok, f"cached={jc:+.10e} uncached={ju:+.10e} rel={rel:.2e}")
    results.append(("Test 3: AD Jacobian cached vs uncached", test3_pass))

    # =====================================================================
    #  TEST 4: Taylor tests at m0 = (0, 0)
    # =====================================================================

    _banner("TEST 4: TAYLOR TESTS at m0=(0,0)")

    # Re-setup cached tape for Taylor tests
    set_working_tape(Tape())
    continue_annotation()

    delta_r.assign(0.0)
    delta_z.assign(0.0)
    xi.interpolate(as_vector([
        delta_r * cos_th * bump,
        delta_r * sin_th * bump,
        delta_z * bump,
    ]))
    mesh3d.coordinates.assign(X_ref + xi)

    u_bar_3d_t4, p_bar_3d_t4, u_cyl_3d_t4 = build_3d_background_flow_differentiable(
        R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh)

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    pf_t4 = perturbed_flow_differentiable(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
        mesh3d, tags, u_bar_3d_t4, p_bar_3d_t4, X_ref, xi, u_cyl_3d_t4)
    F_t4_x, F_t4_z = pf_t4.F_p()

    J_x_t4 = ReducedFunctional(F_t4_x, [c_r, c_z])
    J_z_t4 = ReducedFunctional(F_t4_z, [c_r, c_z])

    m0 = [Function(R_space).assign(0.0), Function(R_space).assign(0.0)]
    h = [Function(R_space).assign(0.1), Function(R_space).assign(0.1)]

    tol_taylor = 1.9
    rate_x = taylor_test(J_x_t4, m0, h)
    rate_z = taylor_test(J_z_t4, m0, h)
    print(f"  J_x min rate = {rate_x:.4f}")
    print(f"  J_z min rate = {rate_z:.4f}")

    test4_pass = _pass_fail("taylor_test(J_x)", rate_x >= tol_taylor, f"rate={rate_x:.4f}")
    test4_pass &= _pass_fail("taylor_test(J_z)", rate_z >= tol_taylor, f"rate={rate_z:.4f}")
    results.append(("Test 4: Taylor tests at m0=(0,0)", test4_pass))

    # =====================================================================
    #  TEST 5: Taylor tests at shifted point m0 = (0.5, 0.3)
    # =====================================================================

    _banner("TEST 5: TAYLOR TESTS at m0=(0.5, 0.3)")

    set_working_tape(Tape())
    continue_annotation()

    delta_r.assign(0.5)
    delta_z.assign(0.3)
    xi.interpolate(as_vector([
        delta_r * cos_th * bump,
        delta_r * sin_th * bump,
        delta_z * bump,
    ]))
    mesh3d.coordinates.assign(X_ref + xi)

    u_bar_3d_t5, p_bar_3d_t5, u_cyl_3d_t5 = build_3d_background_flow_differentiable(
        R_hh, H_hh, W_hh, G_hh, mesh3d, tags, u_2d_hh, p_2d_hh)

    c_r = Control(delta_r)
    c_z = Control(delta_z)

    pf_t5 = perturbed_flow_differentiable(
        R_hh, H_hh, W_hh, L_hh, a_hh, Re_p,
        mesh3d, tags, u_bar_3d_t5, p_bar_3d_t5, X_ref, xi, u_cyl_3d_t5)
    F_t5_x, F_t5_z = pf_t5.F_p()

    J_x_t5 = ReducedFunctional(F_t5_x, [c_r, c_z])
    J_z_t5 = ReducedFunctional(F_t5_z, [c_r, c_z])

    m0_shifted = [Function(R_space).assign(0.5), Function(R_space).assign(0.3)]
    h_shifted = [Function(R_space).assign(0.01), Function(R_space).assign(0.01)]

    rate_x5 = taylor_test(J_x_t5, m0_shifted, h_shifted)
    rate_z5 = taylor_test(J_z_t5, m0_shifted, h_shifted)
    print(f"  J_x min rate = {rate_x5:.4f}")
    print(f"  J_z min rate = {rate_z5:.4f}")

    test5_pass = _pass_fail("taylor_test(J_x) shifted", rate_x5 >= tol_taylor, f"rate={rate_x5:.4f}")
    test5_pass &= _pass_fail("taylor_test(J_z) shifted", rate_z5 >= tol_taylor, f"rate={rate_z5:.4f}")
    results.append(("Test 5: Taylor tests at m0=(0.5,0.3)", test5_pass))

    # =====================================================================
    #  TEST 6: LU factorisation counts
    # =====================================================================

    _banner("TEST 6: LU FACTORISATION COUNTS")

    fwd_count = pf_t5.stokes_ctx.fwd_factor_count
    adj_count = pf_t5.stokes_ctx.adj_factor_count
    print(f"  Forward factorisations : {fwd_count}")
    print(f"  Adjoint factorisations : {adj_count}")
    test6_pass = _pass_fail("fwd_factor_count == 1", fwd_count == 1, f"got {fwd_count}")
    test6_pass &= _pass_fail("adj_factor_count == 1", adj_count == 1, f"got {adj_count}")
    results.append(("Test 6: LU factorisation counts", test6_pass))

    # =====================================================================
    #  SUMMARY
    # =====================================================================

    _banner("SUMMARY")
    all_pass = True
    for name, passed in results:
        tag = "PASS" if passed else "FAIL"
        print(f"  [{tag}] {name}")
        all_pass &= passed

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)