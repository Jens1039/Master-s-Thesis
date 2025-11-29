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

        a_form = ((a ** 2) * inner(grad(u), grad(v)) * dx - a * p * div(v) * dx + a * q * div(u) * dx)
        # grad -> 2 * (sym(grad(u)), sym(grad(v))), => implicit no stress

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

        L_bcs = -((self.a ** 2) * inner(grad(self.v_bc), grad(v_test)) * dx + self.a * q_test * div(self.v_bc) * dx)

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
        sigma_hat = -q_0 * Identity(3) + self.a * (grad(v_0) + grad(v_0).T)
        traction_hat = -dot(n, sigma_hat)
        components = [assemble(traction_hat[i] * ds(self.tags["particle"])) for i in range(3)]
        F_m1 = (1 / self.a ** 2) * np.array([float(c) for c in components])
        return F_m1


    def T_minus_1(self, v_0_a, q_0_a, mesh3d, x, x_p):

        n = FacetNormal(mesh3d)
        sigma_hat = -q_0_a * Identity(3) + self.a * (grad(v_0_a) + grad(v_0_a).T)
        traction_hat = -dot(n, sigma_hat)
        moment_density_hat = cross((x - x_p) / self.a, traction_hat)
        components = [assemble(moment_density_hat[i] * ds(self.tags["particle"])) for i in range(3)]
        T_m1 = (1 / self.a ** 2) * np.array([float(c) for c in components])
        return T_m1


    def compute_F_0_a(self, v_0_a, u_hat_x, u_hat_z, u_bar_3d_a_scaled, x, Theta):

        e_z = as_vector((0.0, 0.0, 1.0))
        ThetaC = Constant(float(Theta))
        term1 = ThetaC * cross(e_z, v_0_a)
        term2 = self.a * dot(grad(u_bar_3d_a_scaled), v_0_a)
        adv_vec = v_0_a + u_bar_3d_a_scaled - ThetaC * cross(e_z, x / self.a)
        term3 = self.a * dot(grad(v_0_a), adv_vec)
        integrand = term1 + term2 + term3
        sign = -1.0
        volume_scale = 1.0 / (self.a ** 3)
        F0_x = sign * volume_scale * assemble(dot(u_hat_x, integrand) * dx(degree=6))
        F0_z = sign * volume_scale * assemble(dot(u_hat_z, integrand) * dx(degree=6))
        return np.array([float(F0_x), 0.0, float(F0_z)])


    def compute_F_0_s(self, v0s, u_hat_x, u_hat_z, u_bar_3d_s_scaled):

        termA = self.a * dot(grad(u_bar_3d_s_scaled), v0s)
        adv_s = v0s + u_bar_3d_s_scaled
        termB = self.a * dot(grad(v0s), adv_s)
        integrand = termA + termB
        sign = -1.0
        volume_scale = 1.0 / (self.a ** 3)
        F_0_s_x = sign * volume_scale * assemble(dot(u_hat_x, integrand) * dx(degree=6))
        F_0_s_z = sign * volume_scale * assemble(dot(u_hat_z, integrand) * dx(degree=6))
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

        ex0 = np.array([x0 / r0, y0 / r0, 0.])
        et0 = np.array([-y0 / r0, x0 / r0, 0.])
        ez0 = np.array([0., 0., 1.])

        bcs_Theta = cross(as_vector((0., 0., 1.)), x) / self.a
        bcs_Omega_Z = cross(as_vector((0., 0., 1.)), x - x_p) / self.a
        bcs_Omega_R = cross(as_vector((float(ex0[0]), float(ex0[1]), 0.0)), x - x_p) / self.a
        bcs_Omega_T = cross(as_vector((float(et0[0]), float(et0[1]), 0.0)), x - x_p) / self.a
        bcs_bg = -u_bar_3d_a_scaled

        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(bcs_Theta)
        v_0_a_Omega_Z, q_0_a_Omega_Z = self.Stokes_solver_3d(bcs_Omega_Z)
        v_0_a_Omega_R, q_0_a_Omega_R = self.Stokes_solver_3d(bcs_Omega_R)
        v_0_a_Omega_T, q_0_a_Omega_T = self.Stokes_solver_3d(bcs_Omega_T) # NEW
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(bcs_bg)

        F_m1_Theta = self.F_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        T_Theta = self.T_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d, x, x_p)
        F_m1_Omega_Z = self.F_minus_1(v_0_a_Omega_Z, q_0_a_Omega_Z, self.mesh3d)
        T_Omega_Z = self.T_minus_1(v_0_a_Omega_Z, q_0_a_Omega_Z, self.mesh3d, x, x_p)
        F_m1_Omega_R = self.F_minus_1(v_0_a_Omega_R, q_0_a_Omega_R, self.mesh3d)
        T_Omega_R = self.T_minus_1(v_0_a_Omega_R, q_0_a_Omega_R, self.mesh3d, x, x_p)
        F_m1_Omega_T = self.F_minus_1(v_0_a_Omega_T, q_0_a_Omega_T, self.mesh3d)
        T_Omega_T = self.T_minus_1(v_0_a_Omega_T, q_0_a_Omega_T, self.mesh3d, x, x_p)
        F_m1_bg = self.F_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d)
        T_bg = self.T_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d, x, x_p)

        A_mat = np.array([
            [np.dot(et0, F_m1_Theta), np.dot(et0, F_m1_Omega_Z), np.dot(et0, F_m1_Omega_R), np.dot(et0, F_m1_Omega_T)],
            [np.dot(ez0, T_Theta),    np.dot(ez0, T_Omega_Z),    np.dot(ez0, T_Omega_R),    np.dot(ez0, T_Omega_T)],
            [np.dot(ex0, T_Theta),    np.dot(ex0, T_Omega_Z),    np.dot(ex0, T_Omega_R),    np.dot(ex0, T_Omega_T)],
            [np.dot(et0, T_Theta),    np.dot(et0, T_Omega_Z),    np.dot(et0, T_Omega_R),    np.dot(et0, T_Omega_T)]
        ])

        b_vec = -np.array([
            np.dot(et0, F_m1_bg),
            np.dot(ez0, T_bg),
            np.dot(ex0, T_bg),
            np.dot(et0, T_bg)
        ])

        solution = np.linalg.solve(A_mat, b_vec)
        Theta_val = float(solution[0])
        Omega_Z_val = float(solution[1])
        Omega_R_val = float(solution[2])
        Omega_T_val = float(solution[3]) # NEW

        v_0_a = Function(v_0_a_Theta.function_space())
        v_0_a.interpolate(Constant(Theta_val) * v_0_a_Theta +
                          Constant(Omega_Z_val) * v_0_a_Omega_Z +
                          Constant(Omega_R_val) * v_0_a_Omega_R +
                          Constant(Omega_T_val) * v_0_a_Omega_T + # NEW
                          v_0_a_bg)

        v_0_s, q_0_s = self.Stokes_solver_3d(-u_bar_3d_s_scaled)
        F_m1_s_dimless = self.F_minus_1(v_0_s, q_0_s, self.mesh3d)

        u_hat_r, _ = self.Stokes_solver_3d(Constant((float(ex0[0]), float(ex0[1]), float(ex0[2]))))
        u_hat_z, _ = self.Stokes_solver_3d(Constant((0., 0., 1.)))

        F_0_a_lift = self.compute_F_0_a(v_0_a, u_hat_r, u_hat_z, u_bar_3d_a_scaled, x, Theta_val)
        F_0_s_lift = self.compute_F_0_s(v_0_s, u_hat_r, u_hat_z, u_bar_3d_s_scaled)

        r_vec = np.array([x0, y0, 0.0])
        R_loc = np.linalg.norm(r_vec)

        e_r = r_vec / R_loc
        R_loc_hat = R_loc / self.a
        val_centrifugal_magnitude = -(4.0 / 3.0) * np.pi * (Theta_val ** 2) * R_loc_hat
        F_centrifugal_vec = val_centrifugal_magnitude * e_r


        u_bar_3d_scaled = self.u_bar_3d * scale_bg
        n = FacetNormal(self.mesh3d)


        integrand = dot(u_bar_3d_scaled, -n) * u_bar_3d_scaled

        F_inertial_sym = [assemble(integrand[i] * ds(self.tags["particle"])) for i in range(3)]
        F_inertial_vec = np.array([float(x) for x in F_inertial_sym]) / (self.a**2)

        F_0_body = F_centrifugal_vec + F_inertial_vec

        F_lift_total = F_0_a_lift + F_0_s_lift + F_0_body
        Ftot = (1.0 / float(self.Re_p)) * F_m1_s_dimless + F_lift_total
        self.F_p = (ex0 @ Ftot) * ex0 + (ez0 @ Ftot) * ez0
        return self.F_p