import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np


'''
class perturbed_flow:

    def __init__(self, R, H, W, a, Re_p, mesh3d, tags, u_bar, p_bar):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.L = 4*max(H, W)
        self.Re_p = Re_p

        self.u_bar = u_bar
        self.p_bar = p_bar

        self.mesh3d = mesh3d
        self.tags = tags

        # Note that the netgen coordinate system can be interpreted as the rotating coordinate system (x', y', z'),
        # whereby we lag behind the particle by 0.5*L/R
        self.x = SpatialCoordinate(self.mesh3d)
        # We therefore formulate all of our vectors in this coordinate system

        # Define the particle center
        self.x_p = Constant(self.tags["particle_center"])

        # Define the unit vectors of the rotating cartesian coordinate system (x', y', z')
        self.e_x_prime = Constant([cos(self.L / self.R * 0.5), sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-sin(self.L / self.R * 0.5), cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        # Define the unit vectors of the rotating cylindrical coordinate system
        self.e_r_prime = as_vector([self.x[0] / sqrt(self.x[0]**2 + self.x[1]**2), self.x[1] / sqrt(self.x[0]**2 + self.x[1]**2), 0])
        self.e_theta_prime = as_vector([- self.x[1]/ sqrt(self.x[0]**2 + self.x[1]**2), self.x[0] / sqrt(self.x[0]**2 + self.x[1]**2), 0])

        # We solve the homogenous part of the Stokes Problem and assemble the matrix
        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.W_mixed = self.V * self.Q

        v_0_hom, q_0 = TrialFunctions(self.W_mixed)
        v, q = TestFunctions(self.W_mixed)

        a_form = 2 * inner(sym(grad(v_0_hom)), sym(grad(v))) * dx - q_0 * div(v) * dx + q * div(v_0_hom) * dx

        self._bcs_hom = [
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["walls"]),
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["particle"]),
        ]

        A = assemble(a_form, bcs=self._bcs_hom)

        nullspace = MixedVectorSpaceBasis(self.W_mixed, [self.W_mixed.sub(0), VectorSpaceBasis(constant=True, comm=self.W_mixed.comm)])

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

        self.v_0_bc = Function(self.V)


    def Stokes_solver_3d(self, particle_bcs):

        v_test, q_test = TestFunctions(self.W_mixed)
        # v_bc is the function capturing the bcs while having arbitrary values on the interior of the domain
        self.v_0_bc.assign(0.0)
        self.v_0_bc.interpolate(particle_bcs)

        DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.tags["walls"]).apply(self.v_0_bc)
        DirichletBC(self.V, particle_bcs, self.tags["particle"]).apply(self.v_0_bc)

        L_bcs = - 2 * inner(sym(grad(self.v_0_bc)), sym(grad(v_test))) * dx - q_test * div(self.v_0_bc) * dx

        b = assemble(L_bcs, tensor=Cofunction(self.W_mixed.dual()), bcs=self._bcs_hom)

        w = Function(self.W_mixed)
        self.solver.solve(w, b)
        v_0_hom, q_0 = w.subfunctions
        v_0 = Function(self.V)
        v_0.assign(v_0_hom)
        v_0 += self.v_0_bc

        return v_0, q_0


    def F_minus_1(self, v_0, q_0, mesh3d):

        # In the paper n points into the centre of the particle
        # FacetNormal(mesh) always points out of the mesh => Therefore they both point in the same direction
        n = FacetNormal(mesh3d)
        sigma = -q_0 * Identity(3) + (grad(v_0) + grad(v_0).T)
        traction = dot(-n, sigma)
        components = [assemble(traction[i] * ds(self.tags["particle"])) for i in range(3)]
        F_minus_1 = np.array([float(c) for c in components])
        return F_minus_1


    def T_minus_1(self, v_0_a, q_0_a, mesh3d):

        n = FacetNormal(mesh3d)
        sigma = -q_0_a * Identity(3) + (grad(v_0_a) + grad(v_0_a).T)
        traction = dot(-n, sigma)
        moment_density = cross((self.x - self.x_p), traction)
        components = [assemble(moment_density[i] * ds(self.tags["particle"])) for i in range(3)]
        T_minus_1 = np.array([float(c) for c in components])
        return T_minus_1


    def compute_F_0(self, v_0_a, v_0_s, u_x, u_z, u_bar_a, u_bar_s, Theta_np):

        x_p_np = np.array(self.x_p.values())
        e_x_prime_np = np.array(self.e_x_prime.values())
        e_z_np = np.array([0, 0, 1])
        Theta = Constant(Theta_np)
        n = FacetNormal(self.mesh3d)

        centrifugal_term = - (4*np.pi)/3 * (Theta_np**2) * np.cross(e_z_np, np.cross(e_z_np, x_p_np))

        inertial_integrand = dot(u_bar_a, -n) * u_bar_a + (dot(u_bar_s, -n) * u_bar_a + dot(u_bar_a, -n) * u_bar_s) + dot(u_bar_s, -n) * u_bar_s
        # With the assemble command UFL becomes a python list
        inertial_integral = [assemble(inertial_integrand[i] * ds(self.tags["particle"], degree=6)) for i in range(3)]
        # We want numpy arrays to be able to still apply computational operations later like np.cross
        inertial_term = np.array(inertial_integral)

        fluid_stress_right_inner = (
                cross(Theta * self.e_z_prime, v_0_a)
                      + dot(grad(u_bar_a), v_0_a)
                      + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * self.e_z_prime, self.x))
                + dot(grad(u_bar_s), v_0_s) + dot(grad(v_0_s), v_0_s + u_bar_s)
                + cross(Theta * self.e_z_prime, v_0_s)
                        - dot(grad(v_0_s), cross(Theta * self.e_z_prime, self.x))
                        + dot(grad(u_bar_s), v_0_a)
                        + dot(grad(u_bar_a), v_0_s)
                        + dot(grad(v_0_s), v_0_a + u_bar_a)
                        + dot(grad(v_0_a), v_0_s + u_bar_s)
        )

        fluid_stress_x = inner(u_x, fluid_stress_right_inner)
        fluid_stress_z = dot(u_z, fluid_stress_right_inner)

        fluid_stress_integral_x = assemble(fluid_stress_x * dx(degree=6))
        fluid_stress_integral_z = assemble(fluid_stress_z * dx(degree=6))

        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        F_0 = fluid_stress_term + inertial_term + centrifugal_term

        return F_0


    def compute_F_0_low_flow_regime(self, v_0_a, v_0_s, u_x, u_z, u_bar_a, u_bar_s, Theta_np):
        """
        Computes the inertial lift force F_0 using the low flow rate approximation.
        Per Section 3 (Page 16) of the paper, terms involving secondary flow
        (u_bar_s, v_0_s) are neglected in this calculation as they scale
        with higher orders of epsilon = L_c/2R.
        """

        x_p_np = np.array(self.x_p.values())
        e_x_prime_np = np.array(self.e_x_prime.values())
        e_z = Constant([0, 0, 1])
        e_z_np = np.array([0, 0, 1])
        Theta = Constant(Theta_np)
        n = FacetNormal(self.mesh3d)

        centrifugal_term = - (4 * np.pi) / 3 * (Theta_np ** 2) * np.cross(e_z_np, np.cross(e_z_np, x_p_np))

        inertial_integrand = dot(u_bar_a, -n) * u_bar_a

        inertial_integral_x = assemble(inertial_integrand[0] * ds(self.tags["particle"], degree=6))
        inertial_integral_y = assemble(inertial_integrand[1] * ds(self.tags["particle"], degree=6))
        inertial_integral_z = assemble(inertial_integrand[2] * ds(self.tags["particle"], degree=6))
        inertial_term = np.array([inertial_integral_x, inertial_integral_y, inertial_integral_z])

        fluid_stress_term_x = (
            inner(u_x,

                  cross(Theta * e_z, v_0_a)

                  + dot(grad(u_bar_a), v_0_a)



                  + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * e_z, self.x))
                  )
        )
        fluid_stress_integral_x = assemble(fluid_stress_term_x * dx(degree=6))

        fluid_stress_term_z = (
            inner(u_z,
                  cross(Theta * e_z, v_0_a)
                  + dot(grad(u_bar_a), v_0_a)
                  + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * e_z, self.x))
                  )
        )
        fluid_stress_integral_z = assemble(fluid_stress_term_z * dx(degree=6))

        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        F_0 = fluid_stress_term + centrifugal_term + inertial_term

        return F_0


    def F_p(self):

        u_bar_a = dot(self.u_bar, self.e_theta_prime) * self.e_theta_prime
        u_bar_s = dot(self.u_bar, self.e_r_prime) * self.e_r_prime + dot(self.u_bar, self.e_z_prime) * self.e_z_prime

        bcs_Theta = cross(self.e_z_prime, self.x)
        bcs_Omega_p_x = cross(self.e_x_prime, self.x - self.x_p)
        bcs_Omega_p_y = cross(self.e_y_prime, self.x - self.x_p)
        bcs_Omega_p_z = cross(self.e_z_prime, self.x - self.x_p)
        bcs_bg = -u_bar_a

        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(bcs_Theta)
        v_0_a_Omega_p_x, q_0_a_Omega_p_x = self.Stokes_solver_3d(bcs_Omega_p_x)
        v_0_a_Omega_p_y, q_0_a_Omega_p_y = self.Stokes_solver_3d(bcs_Omega_p_y)
        v_0_a_Omega_p_z, q_0_a_Omega_p_z = self.Stokes_solver_3d(bcs_Omega_p_z)
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(bcs_bg)

        F_minus_1_a_Theta = self.F_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        F_minus_1_a_Omega_p_x = self.F_minus_1(v_0_a_Omega_p_x, q_0_a_Omega_p_x, self.mesh3d)
        F_minus_1_a_Omega_p_y = self.F_minus_1(v_0_a_Omega_p_y, q_0_a_Omega_p_y, self.mesh3d)
        F_minus_1_a_Omega_p_z = self.F_minus_1(v_0_a_Omega_p_z, q_0_a_Omega_p_z, self.mesh3d)
        F_minus_1_a_Omega_bg = self.F_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d)

        T_minus_1_a_Theta = self.T_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        T_minus_1_a_Omega_p_x = self.T_minus_1(v_0_a_Omega_p_x, q_0_a_Omega_p_x, self.mesh3d)
        T_minus_1_a_Omega_p_y = self.T_minus_1(v_0_a_Omega_p_y, q_0_a_Omega_p_y, self.mesh3d)
        T_minus_1_a_Omega_p_z = self.T_minus_1(v_0_a_Omega_p_z, q_0_a_Omega_p_z, self.mesh3d)
        T_minus_1_a_Omega_bg = self.T_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d)

        e_x_np = np.array(self.e_x_prime.values())
        e_y_np = np.array(self.e_y_prime.values())
        e_z_np = np.array(self.e_z_prime.values())

        A = np.array([

            [np.dot(e_y_np, F_minus_1_a_Theta), np.dot(e_y_np, F_minus_1_a_Omega_p_z),
             np.dot(e_y_np, F_minus_1_a_Omega_p_x), np.dot(e_y_np, F_minus_1_a_Omega_p_y)],

            [np.dot(e_x_np, T_minus_1_a_Theta), np.dot(e_x_np, T_minus_1_a_Omega_p_z),
             np.dot(e_x_np, T_minus_1_a_Omega_p_x), np.dot(e_x_np, T_minus_1_a_Omega_p_y)],

            [np.dot(e_y_np, T_minus_1_a_Theta), np.dot(e_y_np, T_minus_1_a_Omega_p_z),
             np.dot(e_y_np, T_minus_1_a_Omega_p_x), np.dot(e_y_np, T_minus_1_a_Omega_p_y)],

            [np.dot(e_z_np, T_minus_1_a_Theta), np.dot(e_z_np, T_minus_1_a_Omega_p_z),
             np.dot(e_z_np, T_minus_1_a_Omega_p_x), np.dot(e_z_np, T_minus_1_a_Omega_p_y)]
        ])


        b = -np.array([
            np.dot(e_y_np, F_minus_1_a_Omega_bg),
            np.dot(e_x_np, T_minus_1_a_Omega_bg),
            np.dot(e_y_np, T_minus_1_a_Omega_bg),
            np.dot(e_z_np, T_minus_1_a_Omega_bg)
        ])

        Theta_and_Omega_p = np.linalg.solve(A, b)

        Theta = float(Theta_and_Omega_p[0])
        Omega_p_z = float(Theta_and_Omega_p[1])
        Omega_p_x = float(Theta_and_Omega_p[2])
        Omega_p_y = float(Theta_and_Omega_p[3])

        v_0_a = Function(v_0_a_Theta.function_space())

        v_0_a.interpolate(Constant(Theta) * v_0_a_Theta +
                          Constant(Omega_p_x) * v_0_a_Omega_p_x +
                          Constant(Omega_p_y) * v_0_a_Omega_p_y +
                          Constant(Omega_p_z) * v_0_a_Omega_p_z +
                          v_0_a_bg)

        v_0_s, q_0_s = self.Stokes_solver_3d(-u_bar_s)

        F_minus_1_s = self.F_minus_1(v_0_s, q_0_s, self.mesh3d)

        u_hat_x, _ = self.Stokes_solver_3d(self.e_x_prime)
        u_hat_z, _ = self.Stokes_solver_3d(self.e_z_prime)

        F_0 = self.compute_F_0(v_0_a, v_0_s, u_hat_x, u_hat_z, u_bar_a, u_bar_s, Theta)

        F_p_x = np.dot(e_x_np, 1/self.Re_p * F_minus_1_s + F_0)
        F_p_z = np.dot(e_z_np, 1/self.Re_p * F_minus_1_s + F_0)

        return F_p_x, F_p_z
'''






class perturbed_flow:

    def __init__(self, R, H, W, a, Re_p, mesh3d, tags, u_bar, p_bar):
        self.mesh3d = mesh3d
        self.tags = tags

        self.H = H
        self.W = W
        self.R = R
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
        self.u_bar_3d, self.p_bar_3d = u_bar, p_bar

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
        U_m = 2.1
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

        r_local = R_loc - self.R
        z_local = z0
        X_2d = r_local + 0.5 * self.W
        Y_2d = z_local + 0.5 * self.H

        try:
            # u_vec_2d = self.u_bar_3d.at([X_2d, Y_2d])
            u_theta_mag = 2.1 * scale_bg
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

        F_r = np.dot(Ftot, ex0)
        F_z = np.dot(Ftot, ez0)

        return F_r, F_z