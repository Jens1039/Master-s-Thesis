import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
from math import atan2, hypot, cos, sin
from netgen.csg import *
from firedrake import Mesh, COMM_WORLD
import netgen
import math, netgen.libngpy
from netgen.occ import Pnt, ArcOfCircle, Wire, WorkPlane, Axes, Pipe, Sphere, OCCGeometry, X, Y, Z

class background_flow:

    def __init__(self, R, H, W, Q, rho, mu, a):
        self.R = R
        self.H = H
        self.W = W
        self.Q = Q
        self.rho = rho
        self.mu = mu
        self.a = a

    def get_G(self, mesh, calculate_G=True, G = None):
        '''
        We solve the part of the (reduced) Navier-Stokes equation, where G appears, with G = 1:

        rho * (u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_theta * u_r) / (R + r))
        = (G * R) / (R + r) + mu * (sp.diff(u_theta, r, 2) + sp.diff(u_theta, z, 2) + (1 / (R + r)) * sp.diff(u_theta, r) - u_theta / (R + r)**2

        u_theta = 0 on the boundaries (no-slip)

        With our solution, we solve for the volumetric flow rate Q_1 with G = 1:

        Q_1 = integrate over H^2 u_theta(r, z) dz dr

        => G = Q / Q_1

        '''

        if calculate_G == True:
            x = SpatialCoordinate(mesh)

            r = self.R + x[0] - 0.5 * self.W

            V = FunctionSpace(mesh, "CG", 2)
            u = TrialFunction(V)
            v = TestFunction(V)

            a = ( inner(grad(u), grad(v)) - (u.dx(0) / r) * v + (u / (r**2)) * v ) * dx

            L = ( (self.R/ r) * v ) * dx

            bc = DirichletBC(V, Constant(0.0), "on_boundary")

            u_sol = Function(V, name="uhat")

            solve(a == L, u_sol, bcs=bc,
                  solver_parameters=
                  {
                    "ksp_type": "preonly",
                    "pc_type": "lu"
                    })

            x = SpatialCoordinate(mesh)
            r = self.R + x[0] - 0.5 * self.W
            Q_1 = assemble(u_sol * (r) * dx)  # weight with (R + r_local)
            G = self.Q/Q_1

            self.G = G
        else:
            assert G != None; "external value for G required"
            G = G
            self.G = G
        return G


    def solve_2D_background_flow(self, mesh2d):
        # "Q" => Tensorproduct-elements, 2 = degrees of the polynomials
        V = VectorFunctionSpace(mesh2d, "P", 2, dim=3)
        Q = FunctionSpace(mesh2d, "P", 1)
        # Just the Cartesian product of V and Q
        W_mixed = V * Q

        # extracts a UFL expression for the coordinates of the mesh
        x_mesh, z_mesh  = SpatialCoordinate(mesh2d)
        # Initialise r and z so that (r,z) = (0,0) it is at the center of the cross-section
        r = x_mesh - self.W/2
        z = z_mesh - self.H/2

        w = Function(W_mixed)
        u, p = split(w)
        v, q = TestFunctions(W_mixed)

        # Transformation of the velocity components to our local coordinate system
        u_r = u[0]
        u_theta = u[2]
        u_z = u[1]
        v_r = v[0]
        v_theta = v[2]
        v_z = v[1]

        # Shortcuts for readability
        def del_r(f):  return Dx(f, 0)
        def del_z(f):  return Dx(f, 1)

        # weak form of the continuity equation
        F_cont = q * ( del_r(u_r) + del_z(u_z) + u_r/(self.R + r) ) * (self.R + r) * dx

        # weak form of the radial-component
        F_r = (self.rho * (u_r * del_r(u_r) + u_z * del_z(u_r) - (u_theta**2) / (self.R + r)) * v_r
            + del_r(p) * v_r
            + self.mu * dot(grad(u_r), grad(v_r))
            + self.mu * ((1.0/(self.R + r)) * del_r(u_r) - (u_r / (self.R + r)**2)) * v_r) * (self.R + r) * dx

        # weak form of the azimuthal-component of our transformed equation
        F_theta = (self.rho * (u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_theta * u_r) / (self.R +r)) * v_theta
                - ((self.G*self.R) / (self.R + r)) * v_theta
                + self.mu * dot(grad(u_theta), grad(v_theta))
                + self.mu * ((1.0/(self.R + r)) * del_r(u_theta) - (u_theta / (self.R + r)**2)) * v_theta) * (self.R + r) * dx

        # weak form of the z-component of our transformed equation
        F_z = (self.rho * (u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
            + del_z(p) * v_z
            + self.mu * dot(grad(u_z), grad(v_z))
            + self.mu * ( (1.0/(self.R + r)) * del_r(u_z) ) * v_z) * (self.R + r) * dx

        F = F_cont + F_theta + F_r + F_z

        # When you define the problem the boundary conditions must apply to components of W_mixed, not to the stand-alone V, thats why we're using W_mixed.sub(0) = "First component of W_mixed"
        no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")

        # In Navier-Stokes, the pressure just appears in a differentiated form and is therefore unique except for a constant.
        # A constant pressure is (because of a lack of boundary conditions for p) part of the kernel of the linearised operator
        # If we give the solver the information where nullspaces are, it can ignore this direction while inverting the matrix in our Newton method
        nullspace = MixedVectorSpaceBasis(W_mixed, [W_mixed.sub(0), VectorSpaceBasis(constant=True, comm=W_mixed.comm)])

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        # Firedrake calculates the Gateaux-Derivative and transferres it into the Jacobian matrix on the given FEM mesh.
        # The resulting LSE is solved by using LU - decomposition.

        solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters={
                # SNES = Scalable Nonlinear Equation Solver  here: Newton with Line Search as step range criteria
                "snes_type": "newtonls",
                # The type of linesearch is an (Armijo) backtracking strategy where we test the descent criteria ||F(w_k + a_k*d_k)|| < (1 - ca_k)||F(w_k)|| with decreasing a_k until it is valid
                "snes_linesearch_type": "bt",
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
                # "snes_monitor": None,
                "snes_linesearch_maxlambda": 1,
            },
        )

        solver.solve()
        u_sol, p_sol = w.subfunctions
        self.u_sol = u_sol
        self.p_sol = p_sol
        return u_sol, p_sol


    def build_background_flow(self, mesh3d):

        u_2d = self.u_sol
        p_2d = self.p_sol

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
            r_2d = min(max(self.W/2 + (r_glob - self.R), 0.0 + eps), self.W - eps)
            z_2d = min(max(self.H/2 + z, 0.0 + eps), self.H - eps)

            u_points.append((r_2d, z_2d))
            u_angles.append(theta)

        ur_uz_utheta_all = PointEvaluator(mesh=u_2d.function_space().mesh(), points=u_points, tolerance=1e-10).evaluate(u_2d)

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
            x2d = min(max(self.W/2.0 + (R_glob - self.R), 0.0 + eps), self.W - eps)
            z2d = min(max(self.H/2.0 + Z,          0.0 + eps), self.H - eps)
            p_points.append((x2d, z2d))

        p_vals_raw = PointEvaluator(mesh=p_2d.function_space().mesh(), points=p_points, tolerance=1e-10).evaluate(p_2d)

        p_data = p_bar_3d.dat.data
        for j, v in enumerate(p_vals_raw):
            p_data[j] = float(np.asarray(v).reshape(-1)[0])

        return u_bar_3d, p_bar_3d



def make_curved_channel_section_with_spherical_hole(R, W, H, L, a, h, r_off=0.0, z_off=0.0, order=3, global_maxh=None):

    theta = L / R

    p0 = Pnt(R*math.cos(0.0),       R*math.sin(0.0),       0.0)
    pm = Pnt(R*math.cos(theta*0.5), R*math.sin(theta*0.5), 0.0)
    p1 = Pnt(R*math.cos(theta),     R*math.sin(theta),     0.0)
    spine = Wire([ArcOfCircle(p0, pm, p1)])

    wp = WorkPlane(Axes((p0.x, p0.y, p0.z), n=Y, h=Z))
    rect_face = wp.RectangleC(W, H).Face()   # <-- zentriert (wichtig!)  :contentReference[oaicite:0]{index=0}

    channel = Pipe(spine, rect_face)
    channel.faces.name = "walls"

    cx = (R + r_off) * math.cos(theta*0.5)
    cy = (R + r_off) * math.sin(theta*0.5)
    cz = z_off
    sph = Sphere(Pnt(cx, cy, cz), a)
    sph.faces.name = "particle"

    fluid = channel - sph

    fluid.faces.Nearest((p0.x, p0.y, p0.z)).name = "inlet"
    fluid.faces.Nearest((p1.x, p1.y, p1.z)).name = "outlet"


    nrx = math.cos(theta*0.5)
    nry = math.sin(theta*0.5)
    probe = (cx + a*nrx, cy + a*nry, cz)
    fpart = fluid.faces.Nearest(probe)
    fpart.name = "particle"
    fpart.maxh = h

    if global_maxh is None:
        global_maxh = 0.3*max(W, H)

    if COMM_WORLD.rank == 0:
        ngmesh = OCCGeometry(fluid, dim=3).GenerateMesh(maxh=global_maxh)

    else:
        ngmesh = netgen.libngpy._meshing.Mesh(3)

    if order and order >= 2:
        mesh3d = Mesh(Mesh(ngmesh, comm=COMM_WORLD).curve_field(order))
    else:
        mesh3d = Mesh(ngmesh, comm=COMM_WORLD)

    names = ngmesh.GetRegionNames(codim=1)
    def _id(name):
        return names.index(name) + 1 if name in names else None

    tags = {
        "walls":    _id("walls"),
        "inlet":    _id("inlet"),
        "outlet":   _id("outlet"),
        "particle": _id("particle"),
        "theta":    theta,
        "center":   (cx, cy, cz),
        "backend":  "netgen_occ",
    }
    return mesh3d, tags