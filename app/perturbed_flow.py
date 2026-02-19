import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import math



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



if __name__ == "__main__":

    from mpi4py import MPI

    MPI_rank = MPI.COMM_WORLD.rank
    MPI_size = MPI.COMM_WORLD.size

    from build_3d_geometry import *
    from background_flow import *

    # Parameters
    W = 2.0
    H = 2.0
    R = 160.0
    Re = 1.0
    DL = 8.0  # 12.0
    px = 0.0
    py = 0.0  # always 0.0
    pz = 0.0
    pr = 0.05
    a = pr
    L = DL

    bg = background_flow(R, H, W, Re, comm=MPI.COMM_SELF)
    G, _, u_bar_2d, p_bar_2d = bg.solve_2D_background_flow()

    Re_p = Re * (a**2)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, H, W, L, a, particle_maxh=0.2*a, global_maxh=0.2*H)
    particle_marker = tags["particle"]

    n = FacetNormal(mesh3d)
    Xp = Constant(np.array(tags["particle_center"], dtype=PETSc.ScalarType))
    Xc = SpatialCoordinate(mesh3d)

    # Optionally, do various checks of the generated mesh geometry
    if True:
        # Check that the area of various tag surfaces is correct
        for i in ["walls", "particle", "inlet", "outlet"]:
            result = assemble(dot(n, n) * ds(tags[i]))
            if MPI_size > 1:
                result = MPI.COMM_WORLD.allreduce(result, MPI.SUM)
            if MPI_rank == 0:
                print("Calculated surface area of surface", i, "is:", result)
        # Print the expected values for comparison
        if MPI_rank == 0:
            print("Expected inlet/outlet surface area:", W * H)
            print("Expected side wall surface area approximately:", 2 * DL * (W + H))
            print("Expected particle surface area:", 4 * np.pi * pr ** 2)

    if True:
        # Determine if the particle is placed correctly using surface integrals
        c1 = assemble((Xc - Xp)[0] * ds(particle_marker)) / (4 * np.pi * pr ** 2)
        c2 = assemble((Xc - Xp)[1] * ds(particle_marker)) / (4 * np.pi * pr ** 2)
        c3 = assemble((Xc - Xp)[2] * ds(particle_marker)) / (4 * np.pi * pr ** 2)
        if MPI_size > 1:
            c1, c2, c3 = MPI.COMM_WORLD.allreduce(np.array([c1, c2, c3]), MPI.SUM)
        if MPI_rank == 0:
            print("Error in particle surface coordinate mean:", c1, c2, c3)

    if False:
        # Print the number of cells distributed to each MPI process
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        print("Number of cells on rank", MPI_rank, "is:", num_cells)

    u_bar_3d, p_bar_3d = build_3d_background_flow(R, H, W, G, mesh3d, u_bar_2d, p_bar_2d)

    pf = perturbed_flow(R, H, W, a, Re_p, mesh3d, tags, u_bar_3d, p_bar_3d)

    alpha = 0.5 * L / R  # = (1/2)*L/R
    ca, sa = math.cos(alpha), math.sin(alpha)

    Q = np.array([[ca, -sa, 0.0],
                  [sa, ca, 0.0],
                  [0.0, 0.0, 1.0]], dtype=float)


    def to_global_3(v3):
        return Q.T @ np.asarray(v3, dtype=float)


    ex_g_mesh = Constant((ca, sa, 0.0))
    ey_g_mesh = Constant((-sa, ca, 0.0))
    ez_g_mesh = Constant((0.0, 0.0, 1.0))


    def drag_torque_fd(v, q, mesh, tags, x_p, quad_degree=None):
        n = FacetNormal(mesh)
        sigma = -q * Identity(3) + (grad(v) + grad(v).T)
        traction = dot(sigma, -n)

        if quad_degree is None:
            dS = ds(tags["particle"])
        else:
            dS = ds(tags["particle"], degree=quad_degree)

        F_mesh = np.array([float(assemble(traction[i] * dS)) for i in range(3)], dtype=float)

        x = SpatialCoordinate(mesh)
        moment_density = cross(x - x_p, traction)
        T_mesh = np.array([float(assemble(moment_density[i] * dS)) for i in range(3)], dtype=float)

        F_glob = to_global_3(F_mesh)
        T_glob = to_global_3(T_mesh)
        return np.concatenate([F_glob, T_glob])


    motions = [
        ("U_x", ex_g_mesh),
        ("U_y", ey_g_mesh),
        ("U_z", ez_g_mesh),
        ("Omega_x", cross(ex_g_mesh, Xc - Xp)),
        ("Omega_y", cross(ey_g_mesh, Xc - Xp)),
        ("Omega_z", cross(ez_g_mesh, Xc - Xp)),
        ("U_bg", pf.u_bar),
    ]

    results = {}
    for label, bc_expr in motions:
        v, q = pf.Stokes_solver_3d(bc_expr)

        quad = 8 if label == "U_bg" else None

        results[label] = drag_torque_fd(v, q, pf.mesh3d, pf.tags, pf.x_p, quad_degree=quad)

    if pf.mesh3d.comm.rank == 0:
        # Brendan-style lines
        print("U_x coefficients:", *results["U_x"])
        print("U_y coefficients:", *results["U_y"])
        print("U_z coefficients:", *results["U_z"])
        print("Omega_x coefficients:", *results["Omega_x"])
        print("Omega_y coefficients:", *results["Omega_y"])
        print("Omega_z coefficients:", *results["Omega_z"])
        print("Background flow induced force/torque:", *results["U_bg"])