import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np

from background_flow import make_curved_channel_section_with_spherical_hole


class perturbed_flow:

    def __init__(self, mesh3d, mu):
        self.mesh3d = mesh3d
        self.mu = mu

    def Stokes_solver_3d(self, mesh3d, particle_bc, walls_id, particle_id, mu, walls_bc = Constant((0, 0, 0))):
        """
        Solves
            -grad(q) + mu * div(grad(v)) = 0
             div(v) = 0
             particle_bc
             walls_bc
        """
        # Creates a FunctionSpace with dim = mesh.geometric_dimension() Lagrange-Elements (CG) and polynomial degree 2
        V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        Q = FunctionSpace(self.mesh3d, "DG", 1)
        W = V * Q
        print(W.dim())

        # the unknown functions for which the problem shall be solved
        v, q = TrialFunctions(W)
        f, g = TestFunctions(W)

        # weak form of -grad(q) + mu * div(grad(v)) + weak form of div(v)
        a = mu * inner(grad(v), grad(f)) * dx - q * div(f) * dx + g * div(v) * dx

        w = Function(W, name="stokes_solution")

        bcs = [DirichletBC(W.sub(0), walls_bc, walls_id), DirichletBC(W.sub(0), particle_bc, particle_id)]

        nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis(constant=True, comm=W.comm)])

        problem = LinearVariationalProblem(a, 0, w, bcs=bcs)

        solver = LinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
        )

        solver.solve()

        v_0, q_0 = w.subfunctions
        return v_0, q_0


    def F_minus_1(self, v_0, q_0, mesh3d, particle_id, mu=1.0):
        # a UFL normal vector which adapts dynamically to the domain over which an integral is taken
        n = FacetNormal(mesh3d)

        traction = -dot(n, -q_0*Identity(3) + mu*(grad(v_0) + grad(v_0).T))
        # assemble can just integrate scalar values so we integrate over each component separately
        comps = [assemble(traction[i] * ds(particle_id)) for i in range(3)]
        # we want to return the value and not the firedrake object, therefore float(c)
        return np.array([float(c) for c in comps])


    def T_minus_1(self, v_0_a, q_0_a, mesh3d, particle_id, x, x_p, mu=1.0):

        n = FacetNormal(mesh3d)

        traction = -dot(n, -q_0_a*Identity(3) + mu*(grad(v_0_a) + grad(v_0_a).T))

        moment_density = cross(x - x_p, traction)

        comps = [assemble(moment_density[i] * ds(particle_id)) for i in range(3)]

        return np.array([float(c) for c in comps])


    def compute_F_0_a(self, v0a, mesh3d, walls_id, particle_id, u_bar_3d_a, x, Theta, mu=1.0):


        e_z = as_vector((0.0, 0.0, 1.0))
        ThetaC = Constant(float(Theta))

        u_hat_x, _ = self.Stokes_solver_3d(
            mesh3d,
            particle_bc=Constant((1.0, 0.0, 0.0)),
            walls_id=walls_id,
            particle_id=particle_id,
            mu=mu,
            walls_bc=Constant((0.0, 0.0, 0.0)),
        )
        u_hat_z, _ = self.Stokes_solver_3d(
            mesh3d,
            particle_bc=Constant((0.0, 0.0, 1.0)),
            walls_id=walls_id,
            particle_id=particle_id,
            mu=mu,
            walls_bc=Constant((0.0, 0.0, 0.0)),
        )

        term1 = ThetaC * cross(e_z, v0a)

        term2 = dot(grad(u_bar_3d_a), v0a)

        adv_vec = v0a + u_bar_3d_a - ThetaC * cross(e_z, x)
        term3 = dot(grad(v0a), adv_vec)

        integrand = term1 + term2 + term3

        F0_x = assemble(dot(u_hat_x, integrand) * dx)
        F0_z = assemble(dot(u_hat_z, integrand) * dx)

        return np.array([float(F0_x), 0.0, float(F0_z)])


    def compute_F_0_s(self, v0s, mesh3d, walls_id, particle_id, u_bar_3d_s, mu=1.0):

        x = SpatialCoordinate(mesh3d)

        u_hat_x, _ = self.Stokes_solver_3d(
            mesh3d,
            particle_bc=Constant((1.0, 0.0, 0.0)),
            walls_id=walls_id,
            particle_id=particle_id,
            mu=mu,
            walls_bc=Constant((0.0, 0.0, 0.0)),
        )
        u_hat_z, _ = self.Stokes_solver_3d(
            mesh3d,
            particle_bc=Constant((0.0, 0.0, 1.0)),
            walls_id=walls_id,
            particle_id=particle_id,
            mu=mu,
            walls_bc=Constant((0.0, 0.0, 0.0)),
        )

        termA = dot(grad(u_bar_3d_s), v0s)

        adv_s = v0s + u_bar_3d_s
        termB = dot(grad(v0s), adv_s)

        integrand = termA + termB

        F0s_x = assemble(dot(u_hat_x, integrand) * dx)
        F0s_z = assemble(dot(u_hat_z, integrand) * dx)

        return np.array([float(F0s_x), 0.0, float(F0s_z)])


    def solve_particle_dynamics(self, u_bar_3d, tags, Re):

        mesh3d = self.mesh3d
        mu = self.mu


        x = as_vector(SpatialCoordinate(mesh3d))
        x_p = Constant(tags["center"])

        r = sqrt(x[0] ** 2 + x[1] ** 2)
        e_theta = as_vector((-x[1] / r, x[0] / r, 0.0))


        u_bar_3d_a = dot(u_bar_3d, e_theta) * e_theta
        u_bar_3d_s = u_bar_3d - u_bar_3d_a


        bcs_Theta = cross(as_vector((0.0, 0.0, 1.0)), x)
        bcs_Omega = cross(as_vector((0.0, 0.0, 1.0)), x - x_p)
        bcs_bg = -u_bar_3d_a


        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(mesh3d, bcs_Theta, tags["walls"], tags["particle"], mu)
        v_0_a_Omega, q_0_a_Omega = self.Stokes_solver_3d(mesh3d, bcs_Omega, tags["walls"], tags["particle"], mu)
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(mesh3d, bcs_bg, tags["walls"], tags["particle"], mu)


        Fm1_Theta = self.F_minus_1(v_0_a_Theta, q_0_a_Theta, mesh3d, tags["particle"])
        Fm1_Omega = self.F_minus_1(v_0_a_Omega, q_0_a_Omega, mesh3d, tags["particle"])
        Fm1_bg = self.F_minus_1(v_0_a_bg, q_0_a_bg, mesh3d, tags["particle"])

        T_Theta = self.T_minus_1(v_0_a_Theta, q_0_a_Theta, mesh3d, tags["particle"], x, x_p)
        T_Omega = self.T_minus_1(v_0_a_Omega, q_0_a_Omega, mesh3d, tags["particle"], x, x_p)
        T_bg = self.T_minus_1(v_0_a_bg, q_0_a_bg, mesh3d, tags["particle"], x, x_p)


        x0, y0, z0 = tags["center"]
        r0 = np.hypot(x0, y0)
        e_t0 = np.array([-y0 / r0, x0 / r0, 0.0], dtype=float)
        e_z0 = np.array([0.0, 0.0, 1.0])

        A = np.array([
            [np.dot(e_t0, Fm1_Theta), np.dot(e_t0, Fm1_Omega)],
            [np.dot(e_z0, T_Theta), np.dot(e_z0, T_Omega)]
        ], dtype=float)

        b = -np.array([
            np.dot(e_t0, Fm1_bg),
            np.dot(e_z0, T_bg)
        ], dtype=float)

        Theta, Omega_p_abs = np.linalg.solve(A, b)
        Theta = float(Theta)
        Omega_p_abs = float(Omega_p_abs)


        OmegaC = Constant(Omega_p_abs)
        ThetaC = Constant(Theta)

        V = v_0_a_Theta.function_space()
        v_0_a = Function(V, name="v0a")
        v_0_a.interpolate(ThetaC * v_0_a_Theta + OmegaC * v_0_a_Omega + v_0_a_bg)


        bcs_s = -u_bar_3d_s
        v_0_s, q_0_s = self.Stokes_solver_3d(mesh3d, Constant((0.0, 0.0, 0.0)), tags["walls"], tags["particle"], mu,
                                             walls_bc=bcs_s)
        Fm1_s = self.F_minus_1(v_0_s, q_0_s, mesh3d, tags["particle"])


        F0_a = self.compute_F_0_a(v_0_a, mesh3d, tags["walls"], tags["particle"], u_bar_3d_a, x, Theta, mu)
        F0_s = self.compute_F_0_s(v_0_s, mesh3d, tags["walls"], tags["particle"], u_bar_3d_s, mu)

        F0 = F0_a + F0_s


        Ftot = (1.0 / float(Re)) * np.asarray(Fm1_s, dtype=float) + np.asarray(F0, dtype=float)
        ex0 = np.array([x0 / r0, y0 / r0, 0.0], dtype=float)
        ez0 = np.array([0.0, 0.0, 1.0], dtype=float)
        F_p = (ex0 @ Ftot) * ex0 + (ez0 @ Ftot) * ez0


        self.Theta = Theta
        self.Omega_p_abs = Omega_p_abs
        self.F_p = F_p
        self.Ftot = Ftot
        self.v_0_a = v_0_a
        self.v_0_s = v_0_s
        self.q_0_s = q_0_s

        return F_p


def F_p_r_z(background_flow, R, W, H, L, a, h, r_off, z_off, Re):

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, W, H, L, a, h, r_off, z_off)

    u_bar_3d, p_bar_3d = background_flow.build_background_flow(mesh3d)

    particle_flow = perturbed_flow(mesh3d, background_flow.mu)

    F_p = particle_flow.solve_particle_dynamics(u_bar_3d, tags, Re)

    x0, y0, z0 = tags["center"]

    r0 = np.hypot(x0, y0)

    if r0 == 0.0:
            e_r0 = np.array([1.0, 0.0, 0.0])
    else:
            e_r0 = np.array([x0 / r0, y0 / r0, 0.0])
    e_z0 = np.array([0.0, 0.0, 1.0])

    F_r = float(e_r0 @ F_p)
    F_z = float(e_z0 @ F_p)

    return (F_r, F_z)