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
        self.L = 4*max(H, W)
        self.Re_p = Re_p

        self.u_bar = u_bar
        self.p_bar = p_bar

        self.mesh3d = mesh3d
        self.tags = tags

        # Note that the netgen coordinate system can be interpreted as the rotating coordinate system (x', y', z'),
        # whereby we lag behind the particle by 0.5*L/R
        # We therefore formulate all of our vectors in this coordinate system
        self.x = SpatialCoordinate(self.mesh3d)

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

        # Refactor the background inertial volume integral into a surface integral using the divergence theorem and incompressibility
        inertial_integrand = dot(u_bar_a, -n) * u_bar_a + (dot(u_bar_s, -n) * u_bar_a + dot(u_bar_a, -n) * u_bar_s) + dot(u_bar_s, -n) * u_bar_s

        inertial_integral = [assemble(inertial_integrand[i] * ds(self.tags["particle"], degree=6)) for i in range(3)]

        # With the assemble command UFL becomes a python list.
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

        # It seems there is a sign error in the paper in equation (2.24). The minus in front of the RHS is missing.
        fluid_stress_x = - dot(u_x, fluid_stress_right_inner)
        fluid_stress_z = - dot(u_z, fluid_stress_right_inner)

        fluid_stress_integral_x = assemble(fluid_stress_x * dx(degree=6))
        fluid_stress_integral_z = assemble(fluid_stress_z * dx(degree=6))

        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        F_0 = fluid_stress_term + inertial_term + centrifugal_term

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