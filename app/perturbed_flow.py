import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np



class perturbed_flow:

    def __init__(self, mesh3d, tags, background_flow, walls_id, particle_id):
        self.mesh3d = mesh3d
        self.tags = tags
        self.background_flow = background_flow
        self.walls_id = walls_id
        self.particle_id = particle_id
        self.mu = float(getattr(background_flow, "mu", 1.0))
        self.rho = float(getattr(background_flow, "rho", 1.0))
        self.a = float(getattr(background_flow, "a", 1.0))
        self.H = float(getattr(background_flow, "H", 1.0))
        self.W = float(getattr(background_flow, "W", self.H))
        self.inlet_id = tags.get("inlet", None)
        self.outlet_id = tags.get("outlet", None)
        Q = getattr(background_flow, "Q", None)
        Re_attr = getattr(background_flow, "Re", None)
        if Re_attr is not None:
            self.Re = float(Re_attr)
        elif Q is not None:
            self.Re = float(self.rho * Q / (self.mu * self.W))
        else:
            self.Re = 1.0
        self.Re_p = float(self.Re * (self.a / self.H) ** 2)


        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.W_mixed = self.V * self.Q

        u, p = TrialFunctions(self.W_mixed)
        v, q = TestFunctions(self.W_mixed)
        a_form = self.mu * inner(grad(u), grad(v)) * dx - p * div(v) * dx + q * div(u) * dx

        self._bcs_hom = [
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.walls_id),
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.particle_id),
        ]
        A = assemble(a_form, bcs=self._bcs_hom)

        nullspace = MixedVectorSpaceBasis(
            self.W_mixed, [self.W_mixed.sub(0), VectorSpaceBasis(constant=True, comm=self.W_mixed.comm)]
        )
        self.solver = LinearSolver(
            A,
            nullspace=nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        self.v_bc = Function(self.V)
        self.u_bar_3d, self.p_bar_3d = self.background_flow.build_3d_background_flow(self.mesh3d)

    def Stokes_solver_3d(self, particle_bcs):

        v_test, q_test = TestFunctions(self.W_mixed)

        self.v_bc.assign(0.0)

        self.v_bc.interpolate(particle_bcs)

        DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.walls_id).apply(self.v_bc)
        if self.inlet_id is not None:
            DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.inlet_id).apply(self.v_bc)
        if self.outlet_id is not None:
            DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.outlet_id).apply(self.v_bc)

        DirichletBC(self.V, particle_bcs, self.particle_id).apply(self.v_bc)

        Lg = -(
                self.mu * inner(grad(self.v_bc), grad(v_test)) * dx
                + q_test * div(self.v_bc) * dx
        )
        b = assemble(Lg, tensor=Cofunction(self.W_mixed.dual()), bcs=self._bcs_hom)

        w = Function(self.W_mixed, name="stokes_solution")
        self.solver.solve(w, b)
        u0, p = w.subfunctions

        u_total = Function(self.V, name="u_total")
        u_total.assign(u0)
        u_total += self.v_bc
        return u_total, p

    def F_minus_1(self, v_0, q_0, mesh3d, particle_id, mu=1.0):
        n = FacetNormal(mesh3d)
        traction = -dot(n, -q_0 * Identity(3) + mu * (grad(v_0) + grad(v_0).T))
        comps = [assemble(traction[i] * ds(particle_id)) for i in range(3)]
        return np.array([float(c) for c in comps])

    def T_minus_1(self, v_0_a, q_0_a, mesh3d, particle_id, x, x_p, mu=1.0):
        n = FacetNormal(mesh3d)
        traction = -dot(n, -q_0_a * Identity(3) + mu * (grad(v_0_a) + grad(v_0_a).T))
        moment_density = cross(x - x_p, traction)
        comps = [assemble(moment_density[i] * ds(particle_id)) for i in range(3)]
        return np.array([float(c) for c in comps])

    def compute_F_0_a(self, v0a, u_hat_x, u_hat_z, u_bar_3d_a, x, Theta):
        e_z = as_vector((0.0, 0.0, 1.0))
        ThetaC = Constant(float(Theta))
        term1 = ThetaC * cross(e_z, v0a)
        term2 = dot(grad(u_bar_3d_a), v0a)
        adv_vec = v0a + u_bar_3d_a - ThetaC * cross(e_z, x)
        term3 = dot(grad(v0a), adv_vec)
        integrand = term1 + term2 + term3
        F0_x = assemble(dot(u_hat_x, integrand) * dx(degree=6))
        F0_z = assemble(dot(u_hat_z, integrand) * dx(degree=6))

        return np.array([float(F0_x), 0.0, float(F0_z)])

    def compute_F_0_s(self, v0s, u_hat_x, u_hat_z, u_bar_3d_s):
        termA = dot(grad(u_bar_3d_s), v0s)
        adv_s = v0s + u_bar_3d_s
        termB = dot(grad(v0s), adv_s)
        integrand = termA + termB
        F0s_x = assemble(dot(u_hat_x, integrand) * dx(degree=6))
        F0s_z = assemble(dot(u_hat_z, integrand) * dx(degree=6))
        return np.array([float(F0s_x), 0.0, float(F0s_z)])

    def F_p(self):
        x = as_vector(SpatialCoordinate(self.mesh3d))
        x_p = Constant(self.tags["center"])

        rmag = sqrt(x[0] ** 2 + x[1] ** 2)
        rmag_eps = conditional(rmag > 1e-14, rmag, 1.0)
        e_theta = as_vector((-x[1] / rmag_eps, x[0] / rmag_eps, 0.0))

        u_bar_3d = self.u_bar_3d
        u_bar_3d_a = dot(u_bar_3d, e_theta) * e_theta
        u_bar_3d_s = u_bar_3d - u_bar_3d_a

        x0, y0, z0 = self.tags["center"]
        r0 = float(np.hypot(x0, y0))
        if r0 == 0.0:
            ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            ex0 = np.array([x0 / r0, y0 / r0, 0.0], dtype=float)
        ez0 = np.array([0.0, 0.0, 1.0], dtype=float)

        bcs_Theta = cross(as_vector((0.0, 0.0, 1.0)), x)
        bcs_Omega = cross(as_vector((0.0, 0.0, 1.0)), x - x_p)
        bcs_bg = -u_bar_3d_a

        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(bcs_Theta)
        v_0_a_Omega, q_0_a_Omega = self.Stokes_solver_3d(bcs_Omega)
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(bcs_bg)

        Fm1_Theta = self.F_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d, self.particle_id, mu=self.mu)
        Fm1_Omega = self.F_minus_1(v_0_a_Omega, q_0_a_Omega, self.mesh3d, self.particle_id, mu=self.mu)
        Fm1_bg = self.F_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d, self.particle_id, mu=self.mu)

        T_Theta = self.T_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d, self.particle_id, x, x_p, mu=self.mu)
        T_Omega = self.T_minus_1(v_0_a_Omega, q_0_a_Omega, self.mesh3d, self.particle_id, x, x_p, mu=self.mu)
        T_bg = self.T_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d, self.particle_id, x, x_p, mu=self.mu)

        e_t0 = np.array([-y0 / r0, x0 / r0, 0.0], dtype=float) if r0 != 0.0 else np.array([0.0, 1.0, 0.0], float)

        A = np.array([[np.dot(e_t0, Fm1_Theta), np.dot(e_t0, Fm1_Omega)],
                      [np.dot(ez0, T_Theta), np.dot(ez0, T_Omega)]], dtype=float)
        b = -np.array([np.dot(e_t0, Fm1_bg),
                       np.dot(ez0, T_bg)], dtype=float)
        try:
            Theta, Omega_p_abs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            Theta, Omega_p_abs = np.linalg.lstsq(A, b, rcond=None)[0]
        Theta = float(Theta)
        OmegaC = Constant(float(Omega_p_abs))
        ThetaC = Constant(Theta)

        Vloc = v_0_a_Theta.function_space()
        v_0_a = Function(Vloc, name="v0a")
        v_0_a.interpolate(ThetaC * v_0_a_Theta + OmegaC * v_0_a_Omega + v_0_a_bg)

        v_0_s, q_0_s = self.Stokes_solver_3d(-u_bar_3d_s)
        Fm1_s = self.F_minus_1(v_0_s, q_0_s, self.mesh3d, self.particle_id, mu=self.mu)

        u_hat_r, _ = self.Stokes_solver_3d(Constant((float(ex0[0]), float(ex0[1]), float(ex0[2]))))
        u_hat_z, _ = self.Stokes_solver_3d(Constant((0.0, 0.0, -1.0)))

        F0_a = self.compute_F_0_a(v_0_a, u_hat_r, u_hat_z, self.mesh3d, self.walls_id, self.particle_id, u_bar_3d_a, x,
                                  Theta)
        F0_s = self.compute_F_0_s(v_0_s, u_hat_r, u_hat_z, self.mesh3d, self.walls_id, self.particle_id, u_bar_3d_s)
        F0 = F0_a + F0_s

        Ftot = (1.0 / float(self.Re_p)) * np.asarray(Fm1_s, dtype=float) + np.asarray(F0, dtype=float)

        F_p_vec = (ex0 @ Ftot) * ex0 + (ez0 @ Ftot) * ez0

        return F_p_vec
