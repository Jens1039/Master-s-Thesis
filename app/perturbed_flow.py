import os

os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np


class perturbed_flow:

    def __init__(self, mesh3d, tags, a, Re_p, background_flow):
        self.mesh3d = mesh3d
        self.tags = tags
        self.background_flow = background_flow

        self.H = float(getattr(background_flow, "H"))
        self.W = float(getattr(background_flow, "W"))
        self.Re = float(getattr(background_flow, "Re"))
        self.a = a
        self.Re_p = Re_p

        self.inlet_id = tags.get("inlet", None)
        self.outlet_id = tags.get("outlet", None)

        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.W_mixed = self.V * self.Q

        u, p = TrialFunctions(self.W_mixed)
        v, q = TestFunctions(self.W_mixed)

        a_form = inner(grad(u), grad(v)) * dx - p * div(v) * dx + q * div(u) * dx

        self._bcs_hom = [
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["walls"]),
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["particle"]),
        ]
        if self.inlet_id:
            self._bcs_hom.append(DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.inlet_id))
        if self.outlet_id:
            self._bcs_hom.append(DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.outlet_id))

        A = assemble(a_form, bcs=self._bcs_hom)
        nullspace = MixedVectorSpaceBasis(self.W_mixed, [self.W_mixed.sub(0),
                                                         VectorSpaceBasis(constant=True, comm=self.W_mixed.comm)])

        self.solver = LinearSolver(
            A,
            nullspace=nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
                "mat_mumps_icntl_25": 0,
            },
        )

        self.v_bc = Function(self.V)
        self.u_bar_3d, self.p_bar_3d = self.background_flow.build_3d_background_flow(self.mesh3d)

    def Stokes_solver_3d(self, particle_bcs):
        v_test, q_test = TestFunctions(self.W_mixed)
        self.v_bc.assign(0.0)
        self.v_bc.interpolate(particle_bcs)

        DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.tags["walls"]).apply(self.v_bc)
        if self.inlet_id: DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.inlet_id).apply(self.v_bc)
        if self.outlet_id: DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.outlet_id).apply(self.v_bc)
        DirichletBC(self.V, particle_bcs, self.tags["particle"]).apply(self.v_bc)

        L_bcs = -(inner(grad(self.v_bc), grad(v_test)) * dx + q_test * div(self.v_bc) * dx)
        b = assemble(L_bcs, tensor=Cofunction(self.W_mixed.dual()), bcs=self._bcs_hom)

        w = Function(self.W_mixed)
        self.solver.solve(w, b)
        u0, p = w.subfunctions
        u_total = Function(self.V)
        u_total.assign(u0)
        u_total += self.v_bc
        return u_total, p

    def F_minus_1(self, v_0, q_0, mesh3d):
        n = FacetNormal(mesh3d)
        traction = -dot(n, -q_0 * Identity(3) + grad(v_0) + grad(v_0).T)
        components = [assemble(traction[i] * ds(self.tags["particle"])) for i in range(3)]
        return np.array([float(c) for c in components])

    def T_minus_1(self, v_0_a, q_0_a, mesh3d, x, x_p):
        n = FacetNormal(mesh3d)
        traction = -dot(n, -q_0_a * Identity(3) + grad(v_0_a) + grad(v_0_a).T)
        moment_density = cross(x - x_p, traction)
        components = [assemble(moment_density[i] * ds(self.tags["particle"])) for i in range(3)]
        return np.array([float(c) for c in components])

    def compute_F_0_a(self, v_0_a, u_hat_x, u_hat_z, u_bar_3d_a, x, Theta):

        e_z = as_vector((0.0, 0.0, 1.0))
        ThetaC = Constant(float(Theta))
        term1 = ThetaC * cross(e_z, v_0_a)
        term2 = dot(grad(u_bar_3d_a), v_0_a)
        adv_vec = v_0_a + u_bar_3d_a - ThetaC * cross(e_z, x)
        term3 = dot(grad(v_0_a), adv_vec)

        integrand = term1 + term2 + term3
        F0_x = assemble(dot(u_hat_x, integrand) * dx(degree=6))
        F0_z = assemble(dot(u_hat_z, integrand) * dx(degree=6))
        return np.array([float(F0_x), 0.0, float(F0_z)])

    def compute_F_0_s(self, v0s, u_hat_x, u_hat_z, u_bar_3d_s):
        termA = dot(grad(u_bar_3d_s), v0s)
        adv_s = v0s + u_bar_3d_s
        termB = dot(grad(v0s), adv_s)
        integrand = termA + termB

        F_0_s_x = assemble(dot(u_hat_x, integrand) * dx(degree=6))
        F_0_s_z = assemble(dot(u_hat_z, integrand) * dx(degree=6))
        return np.array([float(F_0_s_x), 0.0, float(F_0_s_z)])

    def F_p(self):
        x = as_vector(SpatialCoordinate(self.mesh3d))
        x_p = Constant(self.tags["particle_center"])

        rmag = sqrt(x[0] ** 2 + x[1] ** 2)
        rmag_eps = conditional(rmag > 1e-14, rmag, 1.0)
        e_theta = as_vector((-x[1] / rmag_eps, x[0] / rmag_eps, 0.0))

        u_bar_3d_a = dot(self.u_bar_3d, e_theta) * e_theta
        u_bar_3d_s = self.u_bar_3d - u_bar_3d_a

        ell = min(self.H, self.W)
        U_m = float(self.background_flow.U_m)
        scale_bg = ell / (self.a * U_m)
        u_bar_3d_a_scaled = scale_bg * u_bar_3d_a
        u_bar_3d_s_scaled = scale_bg * u_bar_3d_s

        x0, y0, z0 = self.tags["particle_center"]
        r0 = float(np.hypot(x0, y0))
        if r0 == 0.0:
            ex0, e_t0 = np.array([1., 0., 0.]), np.array([0., 1., 0.])
        else:
            ex0, e_t0 = np.array([x0 / r0, y0 / r0, 0.]), np.array([-y0 / r0, x0 / r0, 0.])
        ez0 = np.array([0., 0., 1.])

        bcs_Theta = cross(as_vector((0., 0., 1.)), x)
        bcs_Omega = cross(as_vector((0., 0., 1.)), x - x_p)
        bcs_bg = -u_bar_3d_a_scaled

        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(bcs_Theta)
        v_0_a_Omega, q_0_a_Omega = self.Stokes_solver_3d(bcs_Omega)
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(bcs_bg)

        F_m1_Theta = self.F_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        F_m1_Omega = self.F_minus_1(v_0_a_Omega, q_0_a_Omega, self.mesh3d)
        F_m1_bg = self.F_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d)
        T_Theta = self.T_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d, x, x_p)
        T_Omega = self.T_minus_1(v_0_a_Omega, q_0_a_Omega, self.mesh3d, x, x_p)
        T_bg = self.T_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d, x, x_p)

        A_mat = np.array([[np.dot(e_t0, F_m1_Theta), np.dot(e_t0, F_m1_Omega)],
                          [np.dot(ez0, T_Theta), np.dot(ez0, T_Omega)]])
        b_vec = -np.array([np.dot(e_t0, F_m1_bg), np.dot(ez0, T_bg)])

        Theta, Omega_val = np.linalg.solve(A_mat, b_vec)
        Theta, Omega_val = float(Theta), float(Omega_val)

        ThetaC, OmegaC = Constant(Theta), Constant(Omega_val)
        v_0_a = Function(v_0_a_Theta.function_space())
        v_0_a.interpolate(ThetaC * v_0_a_Theta + OmegaC * v_0_a_Omega + v_0_a_bg)

        v_0_s, q_0_s = self.Stokes_solver_3d(-u_bar_3d_s_scaled)
        F_m1_s_raw = self.F_minus_1(v_0_s, q_0_s, self.mesh3d)

        u_hat_r, _ = self.Stokes_solver_3d(Constant((float(ex0[0]), float(ex0[1]), float(ex0[2]))))
        u_hat_z, _ = self.Stokes_solver_3d(Constant((0., 0., 1.)))

        F_0_a_raw = self.compute_F_0_a(v_0_a, u_hat_r, u_hat_z, u_bar_3d_a_scaled, x, Theta)
        F_0_s_raw = self.compute_F_0_s(v_0_s, u_hat_r, u_hat_z, u_bar_3d_s_scaled)

        r_vec = np.array([x0, y0, 0.0])
        R_loc = np.linalg.norm(r_vec)
        e_r = r_vec / R_loc

        r_local = R_loc - self.background_flow.R
        z_local = z0
        X_2d = r_local + 0.5 * self.W
        Y_2d = z_local + 0.5 * self.H

        try:
            u_vec_2d = self.background_flow.u_bar.at([X_2d, Y_2d])
            u_theta_mag = u_vec_2d[2] * scale_bg
        except:
            u_theta_mag = 0.0

        val_centrifugal = (4.0 / 3.0) * np.pi * (Theta ** 2) * R_loc
        val_inertia = -(4.0 / 3.0) * np.pi * (u_theta_mag ** 2 / R_loc)
        F_missing_dim = (val_centrifugal + val_inertia) * e_r

        F_missing_coeff = F_missing_dim * (self.a ** 2)

        F_drag_coeff = np.asarray(F_m1_s_raw, dtype=float) / self.a
        F_lift_fem_coeff = (np.asarray(F_0_a_raw, dtype=float) + np.asarray(F_0_s_raw, dtype=float)) / (self.a ** 2)
        F_lift_total = F_lift_fem_coeff + np.asarray(F_missing_coeff, dtype=float)

        Ftot = (1.0 / float(self.Re_p)) * F_drag_coeff + F_lift_total

        self.F_p = (ex0 @ Ftot) * ex0 + (ez0 @ Ftot) * ez0
        return self.F_p