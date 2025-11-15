import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
from math import atan2, hypot, cos, sin



class background_flow:

    def __init__(self, R, H, W, Q, Re):
        self.R = R
        self.H = H
        self.W = W
        self.Q = Q
        self.Re = Re

        # This needs to be adapted once we know how
        self.G = Constant(1)

        self.mesh2d = RectangleMesh(120, 120, self.W, self.H, quadrilateral=False)

    def solve_2D_background_flow(self, H_1_seminorm=False):

        # "P": "Polynomial" Complete Lagrange-polynoms e.g. P(dim=2) = span{1, x, y, xy, x^2, y^2} => triangle in 2d, tetraeder in 3d
        # "Q": "Quadrilateral" Tensor-product polynoms e.g. Q(dim=2) = span{1, x, y, x^2, xy, y^2, x^2y, y^2x, x^2y^2} => squares in 2d, hexaeder in 3d
        # "CG": "Continuous Galerkin" = "P" different names just for historical reasons (FENICS/Firedrake) (continuity on nodes is internally ensured)
        # "DG": "Discontinuous Galerkin" the basis functions are the same, but when assembling the matrix continuity on the nodes is not ensured
        # Taylor - Hood: ("CG" k, "CG" k-1) => LBB - stable, approximatively divergence free, not local mass preservance
        # Scott - Vogelius ("CG" k, "DG" k-1) => LBB - stable on special meshes (e.g. barycentrically refined)
        # but divergence free, local mass preservance

        # Velocity space (3 components: u_r, u_z, u_theta), pressure space, and scalar space for \hat{G}
        V = VectorFunctionSpace(self.mesh2d, "CG", 2, dim=3)
        Q = FunctionSpace(self.mesh2d, "CG", 1)

        # Just the Cartesian product of V, Q and R
        W_mixed = V * Q

        # extracts a UFL expression for the coordinates of the mesh (non-dimensional!)
        x_mesh, z_mesh = SpatialCoordinate(self.mesh2d)

        # Initialise r and z so that (r,z) = (0,0) is at the center of the cross-section
        r = x_mesh - 0.5 * self.W
        z = z_mesh - 0.5 * self.H

        w = Function(W_mixed)
        u, p = split(w)
        v, q = TestFunctions(W_mixed)

        # Transformation of the velocity components to our local coordinate system
        # u = (u_x, u_z, u_theta) in the mesh coordinates, we interpret them as:
        u_r = u[0]
        u_z = u[1]
        u_theta = u[2]

        v_r = v[0]
        v_z = v[1]
        v_theta = v[2]

        # Shortcuts for readability
        def del_r(f):  return Dx(f, 0)
        def del_z(f):  return Dx(f, 1)

        # weak form of the radial-component
        F_r = ((u_r * del_r(u_r) + u_z * del_z(u_r) - (u_theta**2) / (self.R + r)) * v_r
                + del_r(p) * v_r
                + 1/self.Re * dot(grad(u_r), grad(v_r))
                + 1/self.Re * ((1.0/(self.R + r)) * del_r(u_r) - (u_r / (self.R + r)**2)) * v_r) * (self.R + r) * dx

        # weak form of the azimuthal component
        F_theta = ((u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_theta * u_r) / (self.R +r)) * v_theta
                - ((self.G*self.R) / (self.R + r)) * v_theta
                + 1/self.Re * dot(grad(u_theta), grad(v_theta))
                + 1/self.Re * ((1.0/(self.R + r)) * del_r(u_theta) - (u_theta / (self.R + r)**2)) * v_theta) * (self.R + r) * dx

        # weak form of the z-component
        F_z = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
            + del_z(p) * v_z
            + 1/self.Re * dot(grad(u_z), grad(v_z))
            + 1/self.Re * ( (1.0/(self.R + r)) * del_r(u_z) ) * v_z) * (self.R + r) * dx

        # weak form of the continuity equation
        F_cont = q * (del_r(u_r) + del_z(u_z) + u_r / (self.R + r)) * (self.R + r) * dx

        # Total residual
        F = F_r + F_theta + F_z + F_cont

        # When you define the problem the boundary conditions must apply to components of W_mixed, not to the stand-alone V,
        # thats why we're using W_mixed.sub(0) = "First component of W_mixed"
        no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")

        # In Navier-Stokes, the pressure just appears in a differentiated form and is therefore unique except for a constant.
        # A constant pressure is (because of a lack of boundary conditions for p) part of the kernel of the linearised operator.
        # If we give the solver the information where nullspaces are, it can ignore this direction while inverting the matrix
        # in our Newton method.
        nullspace = MixedVectorSpaceBasis(
            W_mixed,
            [
                W_mixed.sub(0),  # velocity block: no special nullspace (we don't tell the solver anything here)
                VectorSpaceBasis(constant=True, comm=W_mixed.comm),  # pressure: constants
            ],
        )

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        # Firedrake calculates the Gateaux-Derivative and transfers it into the Jacobian matrix on the given FEM mesh.
        # The resulting LSE is solved by using LU - decomposition.
        solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters={
                # SNES = Scalable Nonlinear Equation Solver  here: Newton with Line Search as step range criteria
                "snes_type": "newtonls",
                # bt: test the descent criteria ||F(w_k + a_k*d_k)|| < (1 - ca_k)||F(w_k)|| with decreasing a_k until it is valid
                # l2: solves min ||F(x_k + alpha*d_k||_(L^2) over alpha
                "snes_linesearch_type": "l2",
                # Tests for ||F(w_k)||/||F(w_0)|| and stops algorithm once ratio has fallen below
                "snes_rtol": 1e-8,
                # Tests for ||F(w_k)|| and stops algorithm once absolute value has fallen below
                "snes_atol": 1e-10,
                # Maximal Newton steps
                "snes_max_it": 50,
                # ksp = krylov subspace solver  here: We just want to solve our resulting LSE with LU decomposition
                "ksp_type": "preonly",
                # pc = Preconditioner  here: LU decomposition
                "pc_type": "lu",
                # MUMPS = MUltifrontal Massively Parallel sparse direct Solver (highly optimized Solver Implementation)
                "pc_factor_mat_solver_type": "mumps",
                # Theese two parameters can be used for monitoring the optimization process
                # "snes_linesearch_monitor": None,
                "snes_monitor": None,
                "snes_linesearch_maxlambda": 1,
                "snes_monitor_convergence_criteria": None,
            },
        )

        solver.solve()
        u_bar, p_bar = w.subfunctions
        self.u_bar = u_bar
        self.p_bar = p_bar

        if H_1_seminorm:
            print(sqrt(assemble(inner(grad(self.u_bar), grad(self.u_bar)) * dx)))

        return u_bar, p_bar

    def build_3d_background_flow(self, mesh3d):

        u_2d = self.u_bar
        p_2d = self.p_bar

        # ufl_element() gets the information for the family (e.g. "Lagrange"), the cell-type (e.g. "quadrilateral") and
        # the polynomial degree out of a function space like VectorFunctionSpace(mesh, "CG", 2)
        u_el = u_2d.function_space().ufl_element()
        p_el = p_2d.function_space().ufl_element()
        # We now create 3d function spaces with the same family and degree as our 2d function spaces
        V = VectorFunctionSpace(mesh3d, u_el.family(), u_el.degree())
        u_bar_3d = Function(V)
        Q = FunctionSpace(mesh3d, p_el.family(), p_el.degree())
        p_bar_3d = Function(Q)

        x = SpatialCoordinate(mesh3d)

        # creates an empty function in the V Vector Space
        coord_vec = Function(V)
        # fills coord_vec with the true mesh coordinates meaning coord_vec = id_x
        coord_vec.interpolate(as_vector((x[0], x[1], x[2])))
        # .dat grants access to the container, which holds the PETCs-vectors containing the actual data
        # .data_ro grants acces to theese data with ro = read_only
        coords_u = coord_vec.dat.data_ro

        eps = 1e-10
        u_points = []
        u_angles = []
        # iterates over all N grid points of our mesh3d and assigns the associated 2d-coordinate
        for j in range(coords_u.shape[0]):
            # Reads out the spatial coordinates of the current grid point
            X, Y, Z = coords_u[j, :]
            # We transform from global cartesian coordinates to global cylindric coordinates
            r_glob = hypot(X, Y)
            theta = atan2(Y, X)
            z = Z
            # numerical correction to ensure the functionality of PointEvaluator
            r_2d = min(max(self.W / 2 + (r_glob - self.R), 0.0 + eps), self.W - eps)
            z_2d = min(max(self.H / 2 + z, 0.0 + eps), self.H - eps)

            u_points.append((r_2d, z_2d))
            u_angles.append(theta)

        ur_uz_utheta_all = PointEvaluator(mesh=u_2d.function_space().mesh(), points=u_points,
                                          tolerance=1e-10).evaluate(u_2d)

        for j, vals in enumerate(ur_uz_utheta_all):
            u_r_val, u_z_val, u_theta_val = [float(x) for x in np.asarray(vals).reshape(-1)[:3]]
            phi = u_angles[j]
            Ux = u_r_val * cos(phi) - u_theta_val * sin(phi)
            Uy = u_r_val * sin(phi) + u_theta_val * cos(phi)
            Uz = u_z_val

            u_bar_3d.dat.data[j, 0] = Ux
            u_bar_3d.dat.data[j, 1] = Uy
            u_bar_3d.dat.data[j, 2] = Uz

        VQ = VectorFunctionSpace(mesh3d, p_el.family(), p_el.degree(), dim=3)
        coord_sca = Function(VQ)
        coord_sca.interpolate(as_vector((x[0], x[1], x[2])))
        coords_p = coord_sca.dat.data_ro

        p_points = []
        for j in range(coords_p.shape[0]):
            X, Y, Z = coords_p[j, :]
            R_glob = hypot(X, Y)
            x2d = min(max(self.W / 2.0 + (R_glob - self.R), 0.0 + eps), self.W - eps)
            z2d = min(max(self.H / 2.0 + Z, 0.0 + eps), self.H - eps)
            p_points.append((x2d, z2d))

        p_vals_raw = PointEvaluator(mesh=p_2d.function_space().mesh(), points=p_points, tolerance=1e-10).evaluate(
            p_2d)

        p_data = p_bar_3d.dat.data
        for j, v in enumerate(p_vals_raw):
            p_data[j] = float(np.asarray(v).reshape(-1)[0])

        self.u_bar_3d = u_bar_3d
        self.p_bar_3d = p_bar_3d

        return u_bar_3d, p_bar_3d