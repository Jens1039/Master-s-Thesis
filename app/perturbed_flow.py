import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np


class perturbed_flow:

    def __init__(self, R, H, W, a, Re_p, mesh3d, tags, u_bar, p_bar):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.Re_p = Re_p

        self.u_bar = u_bar
        self.p_bar = p_bar

        self.mesh3d = mesh3d
        self.tags = tags

        # The netgen coordinate system perfectly coincides with the (x, y, z) coordinate system from the paper
        x = SpatialCoordinate(self.mesh3d)
        # Therefore in the lab coordinate system x_p = particle center
        x_p = Constant(self.tags["particle_center"])
        # In the rotating coordinate system (x', y', z') we use the sentence from the paper
        # "It is always the case that x_p' = R + r_p, y_p' = 0 and z_p' = z_p"
        # Since r_p = sqrt(x_p^2 + y_p^2) - R, we get x_p' = (sqrt(x_p^2 + y_p^2), 0, z_p)
        self.x_p_prime = Constant([sqrt(x_p[0]**2 + x_p[1]**2), 0, x_p[2]])
        # theta_p = arctan2(y_p, x_p)
        theta_p_np = atan2(self.tags["particle_center"][1], self.tags["particle_center"][0])
        # This is the coordinate of theta over which we are later going to integrate
        theta_coordinate = atan2(x[1], x[0])
        # x' = (cos(theta_p) - sin(theta_p), sin(theta_p) + cos(theta_p), z)
        self.x_prime = as_vector([cos(theta_p_np) * x[0] + sin(theta_p_np) * x[1], -sin(theta_p_np) * x[0] + cos(theta_p_np) * x[1], x[2]])
        # e_x_prime = cos(theta_p_np)*e_x + cos(theta_p_np)*e_y
        self.e_x_prime = Constant((cos(theta_p_np), sin(theta_p_np), 0.0))
        #
        self.e_theta_hat_prime = as_vector((-sin(theta_coordinate), cos(theta_coordinate), 0.0))
        self.e_r_hat_prime = as_vector((cos(theta_coordinate), sin(theta_coordinate), 0.0))
        self.e_z = Constant([0, 0, 1])
        self.e_x_hat_prime_particle = Constant((cos(theta_p_np), sin(theta_p_np), 0.0))
        self.e_theta_hat_prime_particle = Constant((-sin(theta_p_np), cos(theta_p_np), 0.0))

        # Stokes Solver setup in the constructor to enable LU-caching
        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.W_mixed = self.V * self.Q

        u, p = TrialFunctions(self.W_mixed)
        v, q = TestFunctions(self.W_mixed)

        a_form = 2 * inner(sym(grad(u)), sym(grad(v))) * dx - p * div(v) * dx + q * div(u) * dx

        self._bcs_hom = [
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["walls"]),
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["particle"]),
        ]

        # self._bcs_hom.append(DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["inlet"]))
        # self._bcs_hom.append(DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["outlet"]))

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


    def Stokes_solver_3d(self, particle_bcs):

        v_test, q_test = TestFunctions(self.W_mixed)
        self.v_bc.assign(0.0)
        self.v_bc.interpolate(particle_bcs)

        DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.tags["walls"]).apply(self.v_bc)
        # DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.tags["inlet"]).apply(self.v_bc)
        # DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.tags["outlet"]).apply(self.v_bc)
        DirichletBC(self.V, particle_bcs, self.tags["particle"]).apply(self.v_bc)

        L_bcs = - 2 * inner(sym(grad(self.v_bc)), sym(grad(v_test))) * dx - q_test * div(self.v_bc) * dx

        b = assemble(L_bcs, tensor=Cofunction(self.W_mixed.dual()), bcs=self._bcs_hom)

        w = Function(self.W_mixed)
        self.solver.solve(w, b)
        u0, p = w.subfunctions
        u_total = Function(self.V)
        u_total.assign(u0)
        u_total += self.v_bc
        return u_total, p


    def F_minus_1(self, v_0, q_0, mesh3d):

        # In the paper n points into the centre of the particle
        # FacetNormal(mesh) always points out of the mesh => Therefore they both point in the same direction
        n = FacetNormal(mesh3d)
        sigma_hat = -q_0 * Identity(3) + (grad(v_0) + grad(v_0).T)
        traction_hat = dot(-n, sigma_hat)
        components = [assemble(traction_hat[i] * ds(self.tags["particle"])) for i in range(3)]
        F_minus_1 = np.array([float(c) for c in components])
        return F_minus_1


    def T_minus_1(self, v_0_a, q_0_a, mesh3d):

        n = FacetNormal(mesh3d)
        sigma_hat = -q_0_a * Identity(3) + (grad(v_0_a) + grad(v_0_a).T)
        traction_hat = dot(-n, sigma_hat)
        moment_density_hat = cross((self.x_prime - self.x_p_prime), traction_hat)
        components = [assemble(moment_density_hat[i] * ds(self.tags["particle"])) for i in range(3)]
        T_minus_1 = np.array([float(c) for c in components])
        return T_minus_1


    def compute_F_0(self, v_0_a, v_0_s, u_hat_x, u_hat_z, u_bar_a, u_bar_s, Theta_np):

        x_p_np = np.array(self.x_p_prime.values())
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

        fluid_stress_term_x = (
                inner(u_hat_x, cross(Theta * self.e_z, v_0_a)
                      + dot(grad(u_bar_a), v_0_a)
                      + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * self.e_z, self.x_prime)))
                + inner(u_hat_x, dot(grad(u_bar_s), v_0_s) + dot(grad(v_0_s),
                                                                 v_0_s + u_bar_s))
                + inner(u_hat_x, cross(Theta * self.e_z, v_0_s)
                        - dot(grad(v_0_s), cross(Theta * self.e_z, self.x_prime))
                        + dot(grad(u_bar_s), v_0_a)
                        + dot(grad(u_bar_a), v_0_s)
                        + dot(grad(v_0_s), v_0_a + u_bar_a)
                        + dot(grad(v_0_a), v_0_s + u_bar_s))
        )
        fluid_stress_integral_x = assemble(fluid_stress_term_x * dx(degree=6))

        fluid_stress_term_z = (
                inner(u_hat_z, cross(Theta * self.e_z, v_0_a)
                      + dot(grad(u_bar_a), v_0_a)
                      + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * self.e_z, self.x_prime)))
                + inner(u_hat_z, dot(grad(u_bar_s), v_0_s) + dot(grad(v_0_s), v_0_s + u_bar_s))
                + inner(u_hat_z, cross(Theta * self.e_z, v_0_s)
                        - dot(grad(v_0_s), cross(Theta * self.e_z, self.x_prime))
                        + dot(grad(u_bar_s), v_0_a)
                        + dot(grad(u_bar_a), v_0_s)
                        + dot(grad(v_0_s), v_0_a + u_bar_a)
                        + dot(grad(v_0_a), v_0_s + u_bar_s))
        )
        fluid_stress_integral_z = assemble(fluid_stress_term_z * dx(degree=6))

        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        F_0 = fluid_stress_term + centrifugal_term + inertial_term

        return F_0


    def compute_F_0_low(self, v_0_a, v_0_s, u_hat_x, u_hat_z, u_bar_a, u_bar_s, Theta_np):
        """
        Computes the inertial lift force F_0 using the low flow rate approximation.
        Per Section 3 (Page 16) of the paper, terms involving secondary flow
        (u_bar_s, v_0_s) are neglected in this calculation as they scale
        with higher orders of epsilon.
        """

        x_p_np = np.array(self.x_p_prime.values())
        e_x_prime_np = np.array(self.e_x_prime.values())
        e_z = Constant([0, 0, 1])
        e_z_np = np.array([0, 0, 1])
        Theta = Constant(Theta_np)
        n = FacetNormal(self.mesh3d)

        centrifugal_term = - (4 * np.pi) / 3 * (Theta_np ** 2) * np.cross(e_z_np, np.cross(e_z_np, x_p_np))

        # We approximate volume integral of u . grad(u) over the particle.
        # Only axial flow (u_bar_a) is retained.
        # Logic: Divergence theorem converts volume integral to surface integral.
        # n points INTO particle (FEniCS convention), so outward normal is -n.
        inertial_integrand = dot(u_bar_a, -n) * u_bar_a

        inertial_integral_x = assemble(inertial_integrand[0] * ds(self.tags["particle"], degree=6))
        inertial_integral_y = assemble(inertial_integrand[1] * ds(self.tags["particle"], degree=6))
        inertial_integral_z = assemble(inertial_integrand[2] * ds(self.tags["particle"], degree=6))
        inertial_term = np.array([inertial_integral_x, inertial_integral_y, inertial_integral_z])

        # We only retain terms involving v_0_a and u_bar_a.
        # Terms with v_0_s and u_bar_s are dropped.

        # X-Component
        fluid_stress_term_x = (
            inner(u_hat_x,
                  # Coriolis-like term on disturbance
                  cross(Theta * e_z, v_0_a)
                  # Advection of disturbance by background
                  + dot(grad(u_bar_a), v_0_a)
                  # Advection of background by disturbance (linearized)
                  # Note: The term (v_0_a + u_bar_a - cross(...)) represents the total velocity
                  # relative to the rotating frame contribution.
                  + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * e_z, self.x_prime))
                  )
        )
        fluid_stress_integral_x = assemble(fluid_stress_term_x * dx(degree=6))

        # Z-Component
        fluid_stress_term_z = (
            inner(u_hat_z,
                  cross(Theta * e_z, v_0_a)
                  + dot(grad(u_bar_a), v_0_a)
                  + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * e_z, self.x_prime))
                  )
        )
        fluid_stress_integral_z = assemble(fluid_stress_term_z * dx(degree=6))

        # Combine components
        # Note: The integrals are scalars that scale the basis vectors e_x_prime and e_z
        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        # Total F_0
        F_0 = fluid_stress_term + centrifugal_term + inertial_term

        return F_0


    def F_p(self):

        u_bar_a = dot(self.u_bar, self.e_theta_hat_prime) * self.e_theta_hat_prime
        u_bar_s = dot(self.u_bar, self.e_r_hat_prime) * self.e_r_hat_prime + dot(self.u_bar, self.e_z) * self.e_z
        ex0 = np.array(self.e_x_hat_prime_particle.values())
        et0 = np.array(self.e_theta_hat_prime_particle.values())
        ez0 = np.array([0, 0, 1])

        bcs_Theta = cross(self.e_z, self.x_prime)
        bcs_Omega_Z = cross(self.e_z, self.x_prime - self.x_p_prime)
        bcs_Omega_R = cross(as_vector((float(ex0[0]), float(ex0[1]), 0.0)), self.x_prime - self.x_p_prime)
        bcs_Omega_T = cross(as_vector((float(et0[0]), float(et0[1]), 0.0)), self.x_prime - self.x_p_prime)
        bcs_bg = -u_bar_a

        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(bcs_Theta)
        v_0_a_Omega_Z, q_0_a_Omega_Z = self.Stokes_solver_3d(bcs_Omega_Z)
        v_0_a_Omega_R, q_0_a_Omega_R = self.Stokes_solver_3d(bcs_Omega_R)
        v_0_a_Omega_T, q_0_a_Omega_T = self.Stokes_solver_3d(bcs_Omega_T)
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(bcs_bg)

        F_m1_Theta = self.F_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        T_Theta = self.T_minus_1(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        F_m1_Omega_Z = self.F_minus_1(v_0_a_Omega_Z, q_0_a_Omega_Z, self.mesh3d)
        T_Omega_Z = self.T_minus_1(v_0_a_Omega_Z, q_0_a_Omega_Z, self.mesh3d)
        F_m1_Omega_R = self.F_minus_1(v_0_a_Omega_R, q_0_a_Omega_R, self.mesh3d)
        T_Omega_R = self.T_minus_1(v_0_a_Omega_R, q_0_a_Omega_R, self.mesh3d)
        F_m1_Omega_T = self.F_minus_1(v_0_a_Omega_T, q_0_a_Omega_T, self.mesh3d)
        T_Omega_T = self.T_minus_1(v_0_a_Omega_T, q_0_a_Omega_T, self.mesh3d)
        F_m1_bg = self.F_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d)
        T_bg = self.T_minus_1(v_0_a_bg, q_0_a_bg, self.mesh3d)

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
        Omega_T_val = float(solution[3])

        v_0_a = Function(v_0_a_Theta.function_space())

        v_0_a.interpolate(Constant(Theta_val) * v_0_a_Theta +
                          Constant(Omega_Z_val) * v_0_a_Omega_Z +
                          Constant(Omega_R_val) * v_0_a_Omega_R +
                          Constant(Omega_T_val) * v_0_a_Omega_T +
                          v_0_a_bg)

        v_0_s, q_0_s = self.Stokes_solver_3d(-u_bar_s)

        F_minus_1_s = self.F_minus_1(v_0_s, q_0_s, self.mesh3d)

        u_hat_x, _ = self.Stokes_solver_3d(Constant((float(ex0[0]), float(ex0[1]), float(ex0[2]))))
        u_hat_z, _ = self.Stokes_solver_3d(Constant((0., 0., 1.)))

        F_0 = self.compute_F_0_low(v_0_a, v_0_s, u_hat_x, u_hat_z, u_bar_a, u_bar_s, Theta_val)

        F_p = (np.dot(ex0, 1/self.Re_p * F_minus_1_s + F_0)) * ex0 + (np.dot(ez0, 1/self.Re_p * F_minus_1_s + F_0)) * ez0

        self.F_p = F_p

        return F_p