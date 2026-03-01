import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import math


class perturbed_flow:
    """
    Computes the inertial lift force on a particle in a curved duct using a full
    6x6 resistance matrix to enforce F_{-1} = T_{-1} = 0 in all directions.

    This is mathematically equivalent to the 4x4 a/s-decomposed approach in
    perturbed_flow.py, but avoids the 1/Re_p amplification of mesh-induced errors
    in the secondary-flow Stokes drag F_{-1,s}.

    The key identity used is (see Brendan's code comments):
        Theta*cross(e_z, x) + cross(Omega'_p, x - x_p)
            = U_y * e_y' + cross(Omega'_p + Theta*e_z, x - x_p)
    which allows replacing the 4 unknowns (Theta, Omega'_p) with the 6 Cartesian
    DOFs (U_x', U_y', U_z', Omega_total_x, Omega_total_y, Omega_total_z).
    Theta is then recovered as U_y' / |x_p|_{xy}.
    """

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

        self.x = SpatialCoordinate(self.mesh3d)

        self.x_p = Constant(self.tags["particle_center"])

        self.e_x_prime = Constant([cos(self.L / self.R * 0.5), sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-sin(self.L / self.R * 0.5), cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        self.e_r_prime = as_vector([self.x[0] / sqrt(self.x[0]**2 + self.x[1]**2), self.x[1] / sqrt(self.x[0]**2 + self.x[1]**2), 0])
        self.e_theta_prime = as_vector([- self.x[1]/ sqrt(self.x[0]**2 + self.x[1]**2), self.x[0] / sqrt(self.x[0]**2 + self.x[1]**2), 0])

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

        n = FacetNormal(mesh3d)
        sigma = -q_0 * Identity(3) + (grad(v_0) + grad(v_0).T)
        traction = dot(-n, sigma)
        components = [assemble(traction[i] * ds(self.tags["particle"])) for i in range(3)]
        F_minus_1 = np.array([float(c) for c in components])
        return F_minus_1


    def T_minus_1(self, v_0, q_0, mesh3d):

        n = FacetNormal(mesh3d)
        sigma = -q_0 * Identity(3) + (grad(v_0) + grad(v_0).T)
        traction = dot(-n, sigma)
        moment_density = cross((self.x - self.x_p), traction)
        components = [assemble(moment_density[i] * ds(self.tags["particle"])) for i in range(3)]
        T_minus_1 = np.array([float(c) for c in components])
        return T_minus_1


    def compute_F_0(self, v_0, u_hat_x, u_hat_z, Theta_np):
        """
        Compute the O(1) force correction F_0 using the Lorentz reciprocal theorem
        with the full (non-decomposed) v_0 and u_bar.

        Eq. (2.35b) + (2.37) from the paper, simplified because we use the combined
        v_0 = v_{0,a} + v_{0,s} directly.
        """

        x_p_np = np.array(self.x_p.values())
        e_x_prime_np = np.array(self.e_x_prime.values())
        e_z_np = np.array([0, 0, 1])
        Theta = Constant(Theta_np)
        n = FacetNormal(self.mesh3d)

        centrifugal_term = - (4*np.pi)/3 * (self.a**3) * (Theta_np**2) * np.cross(e_z_np, np.cross(e_z_np, x_p_np))

        # Background inertia (centripetal): ∫ (u_bar · ∇)u_bar dV over the particle,
        # converted to a surface integral via the divergence theorem + incompressibility
        inertial_integrand = dot(self.u_bar, -n) * self.u_bar
        inertial_integral = [assemble(inertial_integrand[i] * ds(self.tags["particle"], degree=6)) for i in range(3)]
        inertial_term = np.array(inertial_integral)

        # Reciprocal theorem (eq 2.37): the combined inertia term using full v_0
        # This equals the expanded a/s form in eqs. (2.46a-c) but is numerically simpler.
        inertia = (
            cross(Theta * self.e_z_prime, v_0)
            + dot(grad(self.u_bar), v_0)
            + dot(grad(v_0), v_0 + self.u_bar - cross(Theta * self.e_z_prime, self.x))
        )

        fluid_stress_x = - dot(u_hat_x, inertia)
        fluid_stress_z = - dot(u_hat_z, inertia)

        fluid_stress_integral_x = assemble(fluid_stress_x * dx(degree=6))
        fluid_stress_integral_z = assemble(fluid_stress_z * dx(degree=6))

        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        F_0 = fluid_stress_term + inertial_term + centrifugal_term

        return F_0


    def F_p(self):
        """
        Compute the total cross-sectional force on the particle.

        Uses a 6x6 resistance matrix to enforce F_{-1} = T_{-1} = 0 in all
        directions, then computes the O(1) inertial lift F_0 via the reciprocal
        theorem. The total force is simply F_0 (no 1/Re_p * F_{-1,s} term).

        The 6 DOFs are translations (U_x', U_y', U_z') and combined rotations
        (Omega_total_x, Omega_total_y, Omega_total_z) where
        Omega_total = Omega'_p + Theta * e_z.

        Returns (F_p_x, F_p_z) — the force components in the e_x' (radial) and
        e_z' (vertical) directions.
        """

        e_x_np = np.array(self.e_x_prime.values())
        e_y_np = np.array(self.e_y_prime.values())
        e_z_np = np.array(self.e_z_prime.values())
        directions = [e_x_np, e_y_np, e_z_np]

        # --- Solve 7 Stokes problems (vs 8 in the 4x4 approach) ---
        # 3 translations + 3 rotations + 1 background flow
        # The translation solutions v_Ux and v_Uz double as the dual solutions
        # u_hat_x and u_hat_z for the reciprocal theorem.
        bcs_motions = [
            self.e_x_prime,                                     # U_x'
            self.e_y_prime,                                     # U_y'
            self.e_z_prime,                                     # U_z'
            cross(self.e_x_prime, self.x - self.x_p),          # Omega_x
            cross(self.e_y_prime, self.x - self.x_p),          # Omega_y
            cross(self.e_z_prime, self.x - self.x_p),          # Omega_z
        ]

        solutions = []
        for bc in bcs_motions:
            v, q = self.Stokes_solver_3d(bc)
            solutions.append((v, q))

        v_bg, q_bg = self.Stokes_solver_3d(-self.u_bar)

        # --- Build the 6x6 resistance matrix ---
        # R_mat[i, j] = i-th force/torque component from j-th unit motion
        # Rows 0-2: F · e_x', F · e_y', F · e_z'
        # Rows 3-5: T · e_x', T · e_y', T · e_z'
        R_mat = np.zeros((6, 6))
        b_vec = np.zeros(6)

        for j, (v_j, q_j) in enumerate(solutions):
            F_j = self.F_minus_1(v_j, q_j, self.mesh3d)
            T_j = self.T_minus_1(v_j, q_j, self.mesh3d)
            for i in range(3):
                R_mat[i, j] = np.dot(directions[i], F_j)
                R_mat[i + 3, j] = np.dot(directions[i], T_j)

        F_bg = self.F_minus_1(v_bg, q_bg, self.mesh3d)
        T_bg = self.T_minus_1(v_bg, q_bg, self.mesh3d)
        for i in range(3):
            b_vec[i] = np.dot(directions[i], F_bg)
            b_vec[i + 3] = np.dot(directions[i], T_bg)

        # Solve: R_mat @ alpha = -b_vec  (particle motions must cancel bg-flow forces)
        alpha = np.linalg.solve(R_mat, -b_vec)

        U_x  = float(alpha[0])
        U_y  = float(alpha[1])
        U_z  = float(alpha[2])
        Omega_x = float(alpha[3])
        Omega_y = float(alpha[4])
        Omega_z = float(alpha[5])

        # Recover Theta = U_y / (R + r_p) where R + r_p = |x_p|_{xy}
        x_p_np = np.array(self.x_p.values())
        R_p = np.sqrt(x_p_np[0]**2 + x_p_np[1]**2)
        Theta = U_y / R_p

        # --- Reconstruct the full leading-order perturbation v_0 ---
        v_0 = Function(self.V)
        v_0.interpolate(
            Constant(U_x) * solutions[0][0]
            + Constant(U_y) * solutions[1][0]
            + Constant(U_z) * solutions[2][0]
            + Constant(Omega_x) * solutions[3][0]
            + Constant(Omega_y) * solutions[4][0]
            + Constant(Omega_z) * solutions[5][0]
            + v_bg
        )

        # --- Compute F_0 via the reciprocal theorem ---
        # Reuse translation solutions as the dual fields u_hat_x', u_hat_z'
        u_hat_x = solutions[0][0]
        u_hat_z = solutions[2][0]

        F_0 = self.compute_F_0(v_0, u_hat_x, u_hat_z, Theta)

        # Total force: F'_p = F_0  (F_{-1} = 0 by construction of the 6x6 system)
        F_p_x = np.dot(e_x_np, F_0)
        F_p_z = np.dot(e_z_np, F_0)

        return F_p_x, F_p_z
