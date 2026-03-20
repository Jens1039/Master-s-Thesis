import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np


class perturbed_flow:
    """
    Adjoint-friendly perturbed Stokes solver.
    - Uses Nitsche boundary treatment on walls and particle.
    - Keeps returned force scalars on the AD tape (no numpy solve path).
    """

    def __init__(self, R, H, W, L, a, Re_p, mesh3d, tags, u_bar, p_bar):
        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.L = L
        self.Re_p = Re_p

        self.u_bar = u_bar
        self.p_bar = p_bar
        self.mesh3d = mesh3d
        self.tags = tags

        self.x = SpatialCoordinate(self.mesh3d)
        self.x_p = Constant(self.tags["particle_center"])

        self.e_x_prime = Constant([cos(self.L / self.R * 0.5), sin(self.L / self.R * 0.5), 0.0])
        self.e_y_prime = Constant([-sin(self.L / self.R * 0.5), cos(self.L / self.R * 0.5), 0.0])
        self.e_z_prime = Constant([0.0, 0.0, 1.0])

        r_abs = sqrt(self.x[0] ** 2 + self.x[1] ** 2)
        self.e_r_prime = as_vector([self.x[0] / r_abs, self.x[1] / r_abs, 0.0])
        self.e_theta_prime = as_vector([-self.x[1] / r_abs, self.x[0] / r_abs, 0.0])

        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.R0 = FunctionSpace(self.mesh3d, "R", 0)
        self.W_mixed = self.V * self.Q

        self.gamma_nitsche = Constant(50.0)
        self.h_cell = CellDiameter(self.mesh3d)

    def _sigma(self, u, p):
        return -p * Identity(3) + (grad(u) + grad(u).T)

    def _solve_stokes_with_particle_bc(self, particle_bcs):
        u, p = TrialFunctions(self.W_mixed)
        v, q = TestFunctions(self.W_mixed)
        n = FacetNormal(self.mesh3d)

        sigma_u = self._sigma(u, p)
        sigma_v = self._sigma(v, q)

        a_bulk = (
            2.0 * inner(sym(grad(u)), sym(grad(v))) * dx
            - p * div(v) * dx
            + q * div(u) * dx
        )

        a_wall = (
            -inner(dot(sigma_u, n), v) * ds(self.tags["walls"])
            -inner(dot(sigma_v, n), u) * ds(self.tags["walls"])
            + (self.gamma_nitsche / self.h_cell) * inner(u, v) * ds(self.tags["walls"])
        )

        a_particle = (
            -inner(dot(sigma_u, n), v) * ds(self.tags["particle"])
            -inner(dot(sigma_v, n), u) * ds(self.tags["particle"])
            + (self.gamma_nitsche / self.h_cell) * inner(u, v) * ds(self.tags["particle"])
        )

        L_particle = (
            -inner(dot(sigma_v, n), particle_bcs) * ds(self.tags["particle"])
            + (self.gamma_nitsche / self.h_cell) * inner(particle_bcs, v) * ds(self.tags["particle"])
        )

        a_form = a_bulk + a_wall + a_particle
        L_form = L_particle

        w = Function(self.W_mixed)

        nullspace = MixedVectorSpaceBasis(
            self.W_mixed,
            [self.W_mixed.sub(0), VectorSpaceBasis(constant=True, comm=self.W_mixed.comm)],
        )

        solve(
            a_form == L_form,
            w,
            nullspace=nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
                "mat_mumps_icntl_25": 0,
            },
        )

        u_sol, p_sol = w.subfunctions
        return u_sol, p_sol

    def _force_components(self, v_0, q_0):
        n = FacetNormal(self.mesh3d)
        sigma = self._sigma(v_0, q_0)
        traction = dot(-n, sigma)
        return tuple(assemble(traction[i] * ds(self.tags["particle"])) for i in range(3))

    def _torque_components(self, v_0, q_0):
        n = FacetNormal(self.mesh3d)
        sigma = self._sigma(v_0, q_0)
        traction = dot(-n, sigma)
        moment_density = cross((self.x - self.x_p), traction)
        return tuple(assemble(moment_density[i] * ds(self.tags["particle"])) for i in range(3))

    def _compute_F0_components(self, v_0_a, v_0_s, u_x, u_z, u_bar_a, u_bar_s, Theta):
        n = FacetNormal(self.mesh3d)
        e_z = self.e_z_prime

        inertial_integrand = (
            dot(u_bar_a, -n) * u_bar_a
            + (dot(u_bar_s, -n) * u_bar_a + dot(u_bar_a, -n) * u_bar_s)
            + dot(u_bar_s, -n) * u_bar_s
        )
        inertial_components = tuple(
            assemble(inertial_integrand[i] * ds(self.tags["particle"], degree=6)) for i in range(3)
        )

        centrifugal_vec = -(4.0 * pi / 3.0) * (Theta ** 2) * cross(e_z, cross(e_z, self.x_p))

        fluid_rhs = (
            cross(Theta * self.e_z_prime, v_0_a)
            + dot(grad(u_bar_a), v_0_a)
            + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * self.e_z_prime, self.x))
            + dot(grad(u_bar_s), v_0_s)
            + dot(grad(v_0_s), v_0_s + u_bar_s)
            + cross(Theta * self.e_z_prime, v_0_s)
            - dot(grad(v_0_s), cross(Theta * self.e_z_prime, self.x))
            + dot(grad(u_bar_s), v_0_a)
            + dot(grad(u_bar_a), v_0_s)
            + dot(grad(v_0_s), v_0_a + u_bar_a)
            + dot(grad(v_0_a), v_0_s + u_bar_s)
        )

        fluid_stress_x = -dot(u_x, fluid_rhs)
        fluid_stress_z = -dot(u_z, fluid_rhs)
        fluid_stress_integral_x = assemble(fluid_stress_x * dx(degree=6))
        fluid_stress_integral_z = assemble(fluid_stress_z * dx(degree=6))

        F0 = []
        for i in range(3):
            val = (
                fluid_stress_integral_x * self.e_x_prime[i]
                + fluid_stress_integral_z * self.e_z_prime[i]
                + inertial_components[i]
                + centrifugal_vec[i]
            )
            F0.append(val)
        return tuple(F0)

    def _solve_anti_symmetric_coupled(self, u_bar_a):
        W_a = self.V * self.Q * self.R0 * self.R0 * self.R0 * self.R0
        w = Function(W_a)
        u, p, Theta, Omega_z, Omega_x, Omega_y = split(w)
        v, q, tTheta, tOmega_z, tOmega_x, tOmega_y = TestFunctions(W_a)

        n = FacetNormal(self.mesh3d)
        sigma_u = self._sigma(u, p)
        sigma_v = self._sigma(v, q)

        bcs_Theta = cross(self.e_z_prime, self.x)
        bcs_Omega_p_x = cross(self.e_x_prime, self.x - self.x_p)
        bcs_Omega_p_y = cross(self.e_y_prime, self.x - self.x_p)
        bcs_Omega_p_z = cross(self.e_z_prime, self.x - self.x_p)
        bcs_bg = -u_bar_a

        g_particle = (
            Theta * bcs_Theta
            + Omega_x * bcs_Omega_p_x
            + Omega_y * bcs_Omega_p_y
            + Omega_z * bcs_Omega_p_z
            + bcs_bg
        )

        R_bulk = (
            2.0 * inner(sym(grad(u)), sym(grad(v))) * dx
            - p * div(v) * dx
            + q * div(u) * dx
        )

        R_wall = (
            -inner(dot(sigma_u, n), v) * ds(self.tags["walls"])
            -inner(dot(sigma_v, n), u) * ds(self.tags["walls"])
            + (self.gamma_nitsche / self.h_cell) * inner(u, v) * ds(self.tags["walls"])
        )

        R_particle = (
            -inner(dot(sigma_u, n), v) * ds(self.tags["particle"])
            -inner(dot(sigma_v, n), u - g_particle) * ds(self.tags["particle"])
            + (self.gamma_nitsche / self.h_cell) * inner(u - g_particle, v) * ds(self.tags["particle"])
        )

        traction = dot(-n, sigma_u)
        moment = cross(self.x - self.x_p, traction)

        R_constraints = (
            tTheta * dot(traction, self.e_y_prime) * ds(self.tags["particle"])
            + tOmega_z * dot(moment, self.e_x_prime) * ds(self.tags["particle"])
            + tOmega_x * dot(moment, self.e_y_prime) * ds(self.tags["particle"])
            + tOmega_y * dot(moment, self.e_z_prime) * ds(self.tags["particle"])
        )

        R_total = R_bulk + R_wall + R_particle + R_constraints

        nullspace = MixedVectorSpaceBasis(
            W_a,
            [
                W_a.sub(0),
                VectorSpaceBasis(constant=True, comm=W_a.comm),
                W_a.sub(2),
                W_a.sub(3),
                W_a.sub(4),
                W_a.sub(5),
            ],
        )

        solve(
            R_total == 0,
            w,
            nullspace=nullspace,
            solver_parameters={
                "snes_type": "newtonls",
                "snes_linesearch_type": "l2",
                "mat_type": "aij",
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )

        return w.subfunctions

    def F_p(self):
        u_bar_a = dot(self.u_bar, self.e_theta_prime) * self.e_theta_prime
        u_bar_s = dot(self.u_bar, self.e_r_prime) * self.e_r_prime + dot(self.u_bar, self.e_z_prime) * self.e_z_prime

        v_0_a, q_0_a, Theta, Omega_p_z, Omega_p_x, Omega_p_y = self._solve_anti_symmetric_coupled(u_bar_a)
        v_0_s, q_0_s = self._solve_stokes_with_particle_bc(-u_bar_s)

        u_hat_x, _ = self._solve_stokes_with_particle_bc(self.e_x_prime)
        u_hat_z, _ = self._solve_stokes_with_particle_bc(self.e_z_prime)

        F_minus_1_s = self._force_components(v_0_s, q_0_s)
        F_0 = self._compute_F0_components(v_0_a, v_0_s, u_hat_x, u_hat_z, u_bar_a, u_bar_s, Theta)

        ex = tuple(float(v) for v in self.e_x_prime.values())
        ez = tuple(float(v) for v in self.e_z_prime.values())

        F_total = tuple((1.0 / self.Re_p) * F_minus_1_s[i] + F_0[i] for i in range(3))
        F_p_x = sum(ex[i] * F_total[i] for i in range(3))
        F_p_z = sum(ez[i] * F_total[i] for i in range(3))

        return F_p_x, F_p_z


if __name__ == "__main__":
    pass
