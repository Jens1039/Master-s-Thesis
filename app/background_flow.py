import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
import math
import gmsh
from math import atan2, hypot, cos, sin


def get_G(R, W, Q, mesh):
    '''
    We solve the part of the (reduced) Navier-Stokes equation, where G appears, with G = 1:

    rho * (u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_theta * u_r) / (R + r))
    = (G * R) / (R + r) + mu * (sp.diff(u_theta, r, 2) + sp.diff(u_theta, z, 2) + (1 / (R + r)) * sp.diff(u_theta, r) - u_theta / (R + r)**2

    u_theta = 0 on the boundaries (no-slip)

    With our solution, we solve for the volumetric flow rate Q_1 with G = 1:

    Q_1 = integrate over H^2 u_theta(r, z) dz dr

    => G = Q / Q_1

    '''

    x = SpatialCoordinate(mesh)

    r = R + x[0] - 0.5 * W

    V = FunctionSpace(mesh, "CG", 2)
    u = TrialFunction(V)
    v = TestFunction(V)

    a = ( inner(grad(u), grad(v)) - (u.dx(0) / r) * v + (u / (r**2)) * v ) * dx

    L = ( (R/ r) * v ) * dx

    bc = DirichletBC(V, Constant(0.0), "on_boundary")

    u_sol = Function(V, name="uhat")

    solve(a == L, u_sol, bcs=bc,
          solver_parameters=
          {
            "ksp_type": "preonly",
            "pc_type": "lu"
            })

    Q_1 = assemble(u_sol * dx)
    G = Q/Q_1

    return G

def solve_2D_background_flow(mesh2d, rho, mu, R, G, H, W):
    # "Q" => Tensorproduct-elements, 2 = degrees of the polynomials
    V = VectorFunctionSpace(mesh2d, "P", 2, dim=3)
    Q = FunctionSpace(mesh2d, "P", 1)
    # Just the Cartesian product of V and Q
    W_mixed = V * Q


    # extracts a UFL expression for the coordinates of the mesh
    x_mesh, z_mesh  = SpatialCoordinate(mesh2d)
    # Initialise r and z so that (r,z) = (0,0) it is at the center of the cross-section
    r = x_mesh - W/2
    z = z_mesh - H/2


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
    F_cont = q * ( del_r(u_r) + del_z(u_z) + u_r/(R + r) ) * (R + r) * dx

    # weak form of the radial-component
    F_r = (rho * (u_r * del_r(u_r) + u_z * del_z(u_r) - (u_theta**2) / (R + r)) * v_r
        + del_r(p) * v_r
        + mu * dot(grad(u_r), grad(v_r))
        + mu * ((1.0/(R + r)) * del_r(u_r) - (u_r / (R + r)**2)) * v_r) * (R + r) * dx


    # weak form of the azimuthal-component of our transformed equation
    F_theta = (rho * (u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_theta * u_r) / (R +r)) * v_theta
            - ((G*R) / (R + r)) * v_theta
            + mu * dot(grad(u_theta), grad(v_theta))
            + mu * ((1.0/(R + r)) * del_r(u_theta) - (u_theta / (R + r)**2)) * v_theta) * (R + r) * dx


    # weak form of the z-component of our transformed equation
    F_z = (rho * (u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
        + del_z(p) * v_z
        + mu * dot(grad(u_z), grad(v_z))
        + mu * ( (1.0/(R + r)) * del_r(u_z) ) * v_z) * (R + r) * dx

    F = F_cont + F_theta + F_r + F_z


    # When you define the problem the boundary conditions must apply to components of W_mixed, not to the stand-alone V, thats why we're using W_mixed.sub(0) = "First component of W_mixed"
    no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")

    # In Navier-Stokes, the pressure just appears in a differentiated form and is therefore unique except for a constant.
    # A constant pressure is (because of a lack of boundary conditions for p) part of the kernel of the linearised operator
    # If we give the solver the information where nullspaces are, it can ignore this direction while inverting the matrix in our Newton method
    nullspace = MixedVectorSpaceBasis(W_mixed, [W_mixed.sub(0), VectorSpaceBasis(constant=True)])

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
    return u_sol, p_sol

def make_curved_channel_section_with_spherical_hole(R, W, H, L, a, h, msh_file="curved_channel_with_sphere.msh"):

    theta = L/R

    # Loads the internal structures, modules and C++ libraries of Gmsh, necessary to be able to execute any other gmsh commands
    gmsh.initialize()
    # supresses any communication of gmsh with the terminal
    gmsh.option.setNumber("General.Verbosity", 0)
    # creates in gmsh a new empty model with the name "curved_channel"
    gmsh.model.add("curved_channel")
    # creates in gmshs occ-geomtriekernel a rectangle and returns its ID.
    # The rectangles lives in the z=0 plane and x \in [R - W/2, R + W/2], y \in
    rect = gmsh.model.occ.addRectangle(R - W/2, -H/2, 0.0, W, H)
    # rotates the rectangle so that it lives now in the y=0 plane
    gmsh.model.occ.rotate([(2, rect)], 0, 0, 0, 1, 0, 0, math.pi/2)
    # revolves the object with tag "rect" with dimension 2 around the z-axis with angle theta to create a section of the duct
    out = out = gmsh.model.occ.revolve([(2, rect)], 0, 0, 0, 0, 0, 1, theta)

    # assert if a volume has been created
    vol_tags = [t for (d, t) in out if d == 3]
    if not vol_tags:
        gmsh.finalize()
        raise RuntimeError("revolve didn't create a volume")
    vol = (3, vol_tags[0])

    # locate the center of the mesh
    cx = R * cos(0.5 * theta)
    cy = R * sin(0.5 * theta)
    cz = 0.0

    # adds sphere to the model "curved_channel"
    sph = gmsh.model.occ.addSphere(cx, cy, cz, a)

    # boolean cut of sphere
    cut_obj, _ = gmsh.model.occ.cut([vol], [(3, sph)], removeObject=True, removeTool=True)
    if not cut_obj:
        gmsh.finalize()
        raise RuntimeError("boolean cut failed")
    vol = cut_obj[0]

    # transfers changes made in the occ kernel to the actual model
    gmsh.model.occ.synchronize()

    # control edge length of the elements of our mesh
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(h))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(h))

    #
    angle_tol = min(0.1, abs(theta)/8.0)

    surf_inlet, surf_outlet, surf_particle, surf_walls = [], [], [], []

    # We try to determine, which 2d entity represents which surface in our curved channel section
    # loops over all 2d entities in the model "curved_channel"
    for (dim, tag) in gmsh.model.getEntities(2):
        # tries to locate the surface of the particle by measuring the distance between the com of our entity and of the particle
        cx_s, cy_s, cz_s = gmsh.model.occ.getCenterOfMass(dim, tag)
        if (cx_s - cx) ** 2 + (cy_s - cy) ** 2 + (cz_s - cz) ** 2 < (2 * a) ** 2: surf_particle.append(tag); continue

        theta_c = math.atan2(cy_s, cx_s)
        if abs(theta_c - 0.0) < angle_tol:
            surf_inlet.append(tag)
        elif abs(theta_c - theta) < angle_tol:
            surf_outlet.append(tag)
        else:
            surf_walls.append(tag)

    # apparently this is always necessary in gmsh
    phy = {}
    phy["fluid"]    = gmsh.model.addPhysicalGroup(3, [vol[1]])
    gmsh.model.setPhysicalName(3, phy["fluid"], "fluid")
    def add_pg(name, tags, hint):
        if not tags: return None
        gid = gmsh.model.addPhysicalGroup(2, tags, hint)
        gmsh.model.setPhysicalName(2, gid, name)
        return gid
    phy["walls"]    = add_pg("walls",    surf_walls,   10)
    phy["inlet"]    = add_pg("inlet",    surf_inlet,   11)
    phy["outlet"]   = add_pg("outlet",   surf_outlet,  12)
    phy["particle"] = add_pg("particle", surf_particle,13)

    # sets the version of the .msh file used to store the mesh3d
    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    # takes the geometry created by gmsh.model.occ.* and transforms it into an FEM mesh
    gmsh.model.mesh.generate(3)
    gmsh.write(msh_file)
    # close gmsh
    gmsh.finalize()

    mesh3d = Mesh(msh_file)

    tags = {
        "fluid": phy["fluid"],
        "walls": phy["walls"],
        "inlet": phy["inlet"],
        "outlet": phy["outlet"],
        "particle": phy["particle"],
        "R": R, "L": L, "W": W, "H": H, "a": a, "theta": theta, "center": (cx, cy, cz),
        "msh_file": msh_file
    }

    return mesh3d, tags

def build_background_flow(u_2d, p_2d, mesh2d, mesh3d, R, W, H):

    r_center = W/2
    z_center = H/2
    eps = 1e-10

    # ufl_element() gets the information for the family (e.g. "Lagrange"), the cell-type (e.g. "quadrilateral") and
    # the polynomial degree out of a function space like VectorFunctionSpace(mesh, "CG", 2)
    u_family_2d = u_2d.function_space().ufl_element().family()
    u_degree = u_2d.function_space().ufl_element().degree()
    p_family_2d = p_2d.function_space().ufl_element().family()
    p_degree = p_2d.function_space().ufl_element().degree()

    u_family_3d = u_family_2d
    p_family_3d = p_family_2d

    V3 = VectorFunctionSpace(mesh3d, u_family_3d, u_degree, dim=3)
    u_bar_3d = Function(V3, name="u_background")

    Q3 = FunctionSpace(mesh3d, p_family_3d, p_degree)
    p_bar_3d = Function(Q3, name="p_background")

    x = SpatialCoordinate(mesh3d)

    coord_vec = Function(V3, name="_coord_vec")
    coord_vec.interpolate(as_vector((x[0], x[1], x[2])))
    coords_u = coord_vec.dat.data_ro

    VQ = VectorFunctionSpace(mesh3d, p_family_3d, p_degree, dim=3)
    coord_sca = Function(VQ, name="_coord_sca")
    coord_sca.interpolate(as_vector((x[0], x[1], x[2])))
    coords_p = coord_sca.dat.data_ro

    u_data = u_bar_3d.dat.data

    for j in range(coords_u.shape[0]):
        X, Y, Z = coords_u[j, :]

        phi = atan2(Y, X)
        R_glob = hypot(X, Y)

        x2d_raw = r_center + (R_glob - R)
        z2d_raw = z_center + Z

        x2d = min(max(x2d_raw, 0.0 + eps), W - eps)
        z2d = min(max(z2d_raw, 0.0 + eps), H - eps)

        ur_uz_utheta = u_2d.at([x2d, z2d], tolerance=1e-10)
        u_r_val     = ur_uz_utheta[0]
        u_z_val     = ur_uz_utheta[1]
        u_theta_val = ur_uz_utheta[2]

        Ux = u_r_val * cos(phi) + u_theta_val * (-sin(phi))
        Uy = u_r_val * sin(phi) + u_theta_val * ( cos(phi))
        Uz = u_z_val

        u_data[j, 0] = Ux
        u_data[j, 1] = Uy
        u_data[j, 2] = Uz

    p_data = p_bar_3d.dat.data

    for j in range(coords_p.shape[0]):
        X, Y, Z = coords_p[j, :]

        R_glob = hypot(X, Y)

        x2d_raw = r_center + (R_glob - R)
        z2d_raw = z_center + Z

        x2d = min(max(x2d_raw, 0.0 + eps), W - eps)
        z2d = min(max(z2d_raw, 0.0 + eps), H - eps)

        p_val_raw = p_2d.at([x2d, z2d], tolerance=1e-10)
        if isinstance(p_val_raw, (list, tuple, np.ndarray)):
            p_val = float(np.asarray(p_val_raw).reshape(-1)[0])
        else:
            p_val = float(p_val_raw)

        p_data[j] = p_val

    return u_bar_3d, p_bar_3d