################################################
### Step 0.1: Imports
################################################

import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
from scipy.interpolate import RectBivariateSpline as RBS
from MeshGeneration_firedrake import GenerateCurvedDuctParticleMesh
from BackgroundFlow_firedrake import ComputeBackgroundFlow

from mpi4py import MPI


################################################
### Step 0.2: Parameter definitions
################################################


def InertialLiftCalculation(W = 2.0,
                            H = 2.0,
                            R = 160,
                            Re = 1.0,
                            DL = 8.0,
                            px = 0.0,
                            py = 0.0, # always 0.0
                            pz = 0.25,
                            pr = 0.05,
                            comm = None):

    # Use provided communicator or default to COMM_WORLD
    if comm is None:
        comm = MPI.COMM_WORLD
    
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()




    ################################################
    ### Step 1.1: Background flow computation
    ################################################

    # Compute background flow (in 2D) and setup an interpolant function
    RZ, UVW = ComputeBackgroundFlow(W, H, R, Re = 1.0, nW = 32, nH = 32, scaling = "mean", comm=comm)
    U_RBS = RBS(RZ[0][0,:],RZ[1][:,0],UVW[0].T)
    V_RBS = RBS(RZ[0][0,:],RZ[1][:,0],UVW[1].T)
    W_RBS = RBS(RZ[0][0,:],RZ[1][:,0],UVW[2].T)
    def ubar(x):
        rad = ((R+x[0])**2+x[1]**2)**0.5
        r = -R+rad
        z = x[2]
        u_th = U_RBS.ev(r,z)
        u_r = V_RBS.ev(r,z)
        u_z = W_RBS.ev(r,z)
        cos_th = (x[0]+R)/rad
        sin_th = x[1]/rad
        return [-u_th*sin_th+u_r*cos_th,u_th*cos_th+u_r*sin_th,u_z]

    #################################################
    ### Step 1.2: Optional checks on the flow
    #################################################

    # Check velocity in centre
    if True:
        if MPI_rank==0:
            print("Background flow velocity at cross-section centre:",ubar([0,0,0]))
            print("Background flow velocity at particle centre:",ubar([px,py,pz]))

    ################################################
    ### Step 2.1: 3D particle mesh generation
    ################################################

    facet_markers, mesh, cell_tags, facet_tags = GenerateCurvedDuctParticleMesh(W,H,R,px,pz,pr,DL,comm=comm)

    # Define various mesh related quantities
    particle_marker = facet_markers["particle"]
    ds = Measure("ds", domain=mesh)
    n = FacetNormal(mesh)
    Xp = Constant(np.array([px, py, pz], dtype=ScalarType))
    Xc = ufl.SpatialCoordinate(mesh)

    #################################################
    ### Step 2.2: Optional various checks on the mesh
    #################################################

    # Optionally, do various checks of the generated mesh geometry
    if True:
        # Check that the area of various tag surfaces is correct
        for i in range(1,5):
            result = assemble(dot(n, n) * ds(i))
            if MPI_size>1:
                result = comm.allreduce(result,MPI.SUM)
            if MPI_rank==0:
                print("Calculated surface area of surface",i,"is:",result)
        # Print the expected values for comparison
        if MPI_rank==0:
            print("Expected inlet/outlet surface area:",W*H)
            print("Expected side wall surface area approximately:",2*DL*(W+H))
            print("Expected particle surface area:",4*np.pi*pr**2)

    if True:
        # Determine if the particle is placed correctly using surface integrals
        c1 = assemble((Xc-Xp)[0]*ds(particle_marker))/(4*np.pi*pr**2)
        c2 = assemble((Xc-Xp)[1]*ds(particle_marker))/(4*np.pi*pr**2)
        c3 = assemble((Xc-Xp)[2]*ds(particle_marker))/(4*np.pi*pr**2)
        if MPI_size>1:
            c1,c2,c3 = comm.allreduce(np.array([c1,c2,c3]),MPI.SUM)
        if MPI_rank==0:
            print("Error in particle surface coordinate mean:",c1,c2,c3)

    if True:
        # Print the number of cells distributed to each MPI process
        num_cells = mesh.num_cells()
        print("Number of cells on rank", MPI_rank, "is:", num_cells)

    ############################################################################
    ### Step 3: Set up finite element spaces and boundary conditions
    ############################################################################

    # Setup function spaces and boundary conditions
    V, Q = VectorFunctionSpace(mesh, "Lagrange", 2), FunctionSpace(mesh, "Lagrange", 1)
    # For firedrake, we have to formulate the problem on a mixed space
    Z = V * Q

    # Define some convenience constants/functions
    ex = np.array([1.0,0.0,0.0], dtype=PETSc.ScalarType)
    ey = np.array([0.0,1.0,0.0], dtype=PETSc.ScalarType)
    ez = np.array([0.0,0.0,1.0], dtype=PETSc.ScalarType)
    k = Constant(ez)
    zero_const = Constant(PETSc.ScalarType(0))
    zero_vec = Constant([PETSc.ScalarType(0)]*mesh.geometric_dimension())
    R_vec = Constant([PETSc.ScalarType(R),PETSc.ScalarType(0),PETSc.ScalarType(0)])
    wx_fun = as_vector((0.0*Xc[0], -(Xc[2] - pz),  (Xc[1] - py)))
    wy_fun = as_vector(((Xc[2] - pz), 0.0*Xc[0], -(Xc[0] - px)))
    wz_fun = as_vector((-(Xc[1] - py), (Xc[0] - px), 0.0*Xc[0]))
    ubar_fun = Function(V)
    Vs = V.sub(0)
    Wcoords = VectorFunctionSpace(mesh, Vs.ufl_element())
    X = assemble(interpolate(mesh.coordinates, Wcoords))
    def ubar_points(Xnd):
        x0 = Xnd[:, 0]
        x1 = Xnd[:, 1]
        x2 = Xnd[:, 2]

        rad = np.sqrt((R + x0)**2 + x1**2)
        r = -R + rad
        z = x2

        u_th = U_RBS.ev(r, z)
        u_r  = V_RBS.ev(r, z)
        u_z  = W_RBS.ev(r, z)

        cos_th = (x0 + R) / rad
        sin_th = x1 / rad

        out = np.empty((Xnd.shape[0], 3), dtype=ScalarType)
        out[:, 0] = -u_th * sin_th + u_r * cos_th
        out[:, 1] =  u_th * cos_th + u_r * sin_th
        out[:, 2] =  u_z
        return out
    ubar_fun = Function(V)
    ubar_fun.dat.data[:] = ubar_points(X.dat.data_ro)

    # Locate boundary dofs is not native firedrake

    # No-slip condition on walls and inlet/outlet
    noslip = np.zeros(mesh.geometric_dimension(), dtype=PETSc.ScalarType)  # type: ignore
    bc_walls  = DirichletBC(Z.sub(0), noslip, facet_markers["walls"])
    #bc_inlet  = DirichletBC(V, noslip, facet_markers["inlet"])
    #bc_outlet = DirichletBC(V, noslip, facet_markers["outlet"])

    # First boundary condition on the particle
    bc_particle = DirichletBC(Z.sub(0), ex, facet_markers["particle"])

    # Collect Dirichlet boundary conditions
    bcs = [bc_walls,bc_particle]#,bc_inlet,bc_outlet]

    ############################################################################
    ### Step 4: Setup weak form and ONE matrix (homogeneous BCs) + lifting solves
    ############################################################################

    (u, p) = TrialFunctions(Z)
    (v, q) = TestFunctions(Z)
    a = (-2*inner(sym(grad(u)), sym(grad(v))) * dx
         + inner(p, div(v)) * dx
         + inner(div(u), q) * dx)
    L = (inner(zero_vec, v) * dx
       + inner(zero_const, q) * dx)

    # Define a solver object
    def solve_system(bcs, up=None):
        if up is None:
            up = Function(Z)

        # Use Firedrake's built-in solve which handles mixed spaces and bcs correctly
        solve(a == L, up, bcs=bcs, solver_parameters={
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps'
        })

        u, p = up.subfunctions
        return u, p

    #################################################################################
    ### Step 5: Solve for the various boundary conditions and calculate mass matrix
    #################################################################################

    # Define a convenience function for calculating drag and torque coefficients
    def calculate_drag_torque(up,p=None,string=None):
        if p is None:
            u,p = up.subfunctions
        else:
            u = up
        traction = dot(-p*Identity(3)+nabla_grad(u)+nabla_grad(u).T,-n)
        # Additionally, we take -n to get the outward normal on the particle surface.
        Dx = assemble(traction[0]*ds(particle_marker))
        Dy = assemble(traction[1]*ds(particle_marker))
        Dz = assemble(traction[2]*ds(particle_marker))
        Tx = assemble(cross(Xc-Xp,traction)[0]*ds(particle_marker))
        Ty = assemble(cross(Xc-Xp,traction)[1]*ds(particle_marker))
        Tz = assemble(cross(Xc-Xp,traction)[2]*ds(particle_marker))
        if MPI_size>1:
            Dx,Dy,Dz,Tx,Ty,Tz = comm.allreduce(np.array([Dx,Dy,Dz,Tx,Ty,Tz]),MPI.SUM)
        if MPI_rank==0 and (string is not None):
            print(string,Dx,Dy,Dz,Tx,Ty,Tz)
        return Dx,Dy,Dz,Tx,Ty,Tz

    # Do the first solve and calculate the coefficients
    v_ex,q_ex = solve_system(bcs)
    Dx_ex,Dy_ex,Dz_ex,Tx_ex,Ty_ex,Tz_ex = calculate_drag_torque(v_ex,q_ex,"U_x coefficients:")

    # Adjust the particle bc and solve once more...
    # Note: Because of the identity
    # Theta*cross(e_z,x)+cross(Omega_p,x-x_p)=Theta*cross(e_z,x_p)+cross(Omega_p+Theta*e_z,x-x_p)
    #                                        =Theta*(R+x_p)*e_y   +cross(Omega_p+Theta*e_z,x-x_p)
    #                                        =U_y*e_y             +cross(Omega_p+Theta*e_z,x-x_p)
    # we can carry out our computations with a Cartesian based mass matrix.
    # We just need to keep in mind that the z-component of the spin needs to be interpreted as Omega_p+Theta*e_z
    bc_particle = DirichletBC(Z.sub(0), ey, facet_markers["particle"])
    bcs[1] = bc_particle
    v_ey,q_ey = solve_system(bcs)
    Dx_ey,Dy_ey,Dz_ey,Tx_ey,Ty_ey,Tz_ey = calculate_drag_torque(v_ey,q_ey,"U_y coefficients:")

    # Adjust the particle bc and solve once more...
    bc_particle = DirichletBC(Z.sub(0), ez, facet_markers["particle"])
    bcs[1] = bc_particle
    v_ez,q_ez = solve_system(bcs)
    Dx_ez,Dy_ez,Dz_ez,Tx_ez,Ty_ez,Tz_ez = calculate_drag_torque(v_ez,q_ez,"U_z coefficients:")

    # Adjust the particle bc and solve once more...
    bc_particle = DirichletBC(Z.sub(0), wx_fun, facet_markers["particle"])
    bcs[1] = bc_particle
    v_wx,q_wx = solve_system(bcs)
    Dx_wx,Dy_wx,Dz_wx,Tx_wx,Ty_wx,Tz_wx = calculate_drag_torque(v_wx,q_wx,"Omega_x coefficients:")

    # Adjust the particle bc and solve once more...
    bc_particle = DirichletBC(Z.sub(0), wy_fun, facet_markers["particle"])
    bcs[1] = bc_particle
    v_wy,q_wy = solve_system(bcs)
    Dx_wy,Dy_wy,Dz_wy,Tx_wy,Ty_wy,Tz_wy = calculate_drag_torque(v_wy,q_wy,"Omega_y coefficients:")

    # Adjust the particle bc and solve once more...
    bc_particle = DirichletBC(Z.sub(0), wz_fun, facet_markers["particle"])
    bcs[1] = bc_particle
    v_wz,q_wz = solve_system(bcs)
    Dx_wz,Dy_wz,Dz_wz,Tx_wz,Ty_wz,Tz_wz = calculate_drag_torque(v_wz,q_wz,"Omega_z coefficients:")

    # Adjust the particle bc and solve once more...
    bc_particle = DirichletBC(Z.sub(0), ubar_fun, facet_markers["particle"])
    bcs[1] = bc_particle
    v_ub,q_ub = solve_system(bcs)
    Dx_ub,Dy_ub,Dz_ub,Tx_ub,Ty_ub,Tz_ub = calculate_drag_torque(v_ub,q_ub,"Background flow induced force/torque:")

    # Calculate the particle leading order motion via the mass matrix
    M = np.array([[Dx_ex,Dy_ex,Dz_ex,Tx_ex,Ty_ex,Tz_ex],
                  [Dx_ey,Dy_ey,Dz_ey,Tx_ey,Ty_ey,Tz_ey],
                  [Dx_ez,Dy_ez,Dz_ez,Tx_ez,Ty_ez,Tz_ez],
                  [Dx_wx,Dy_wx,Dz_wx,Tx_wx,Ty_wx,Tz_wx],
                  [Dx_wy,Dy_wy,Dz_wy,Tx_wy,Ty_wy,Tz_wy],
                  [Dx_wz,Dy_wz,Dz_wz,Tx_wz,Ty_wz,Tz_wz]])
    b = np.array([Dx_ub,Dy_ub,Dz_ub,Tx_ub,Ty_ub,Tz_ub])
    res = np.linalg.solve(M.T,b)
    if MPI_rank==0:
        print("Leading order particle vel/spin:",res)

    # Reconstruct v0 and check the result produces no force/torque
    def dummy_exp(x):
        UVW = ubar(x)
        return (res[0]                 +res[4]*(x[2]-pz)-res[5]*(x[1]-py)-UVW[0],
                res[1]-res[3]*(x[2]-pz)                 +res[5]*(x[0]-px)-UVW[1],
                res[2]+res[3]*(x[1]-py)-res[4]*(x[0]-px)                 -UVW[2])
    dummy_fun = Function(V)
    # ---firedrake version of dummy_fun.interpolate(dummy_exp)---
    X = X.dat.data_ro
    UVW = ubar_points(X)
    out = np.empty_like(UVW)
    out[:,0] = res[0] + res[4]*(X[:,2]-pz) - res[5]*(X[:,1]-py) - UVW[:,0]
    out[:,1] = res[1] - res[3]*(X[:,2]-pz) + res[5]*(X[:,0]-px) - UVW[:,1]
    out[:,2] = res[2] + res[3]*(X[:,1]-py) - res[4]*(X[:,0]-px) - UVW[:,2]
    dummy_fun.dat.data[:] = out
    # ------
    bc_particle = DirichletBC(Z.sub(0), dummy_fun, facet_markers["particle"])
    bcs[1] = bc_particle
    v0,q0 = solve_system(bcs)
    Dx_v0,Dy_v0,Dz_v0,Tx_v0,Ty_v0,Tz_v0 = calculate_drag_torque(v0,q0,"Leading order drag/torque:")

    ##################################################################################
    ### Step 6: Calculate the inertial lift force (and centripetal/centrifugal forces)
    ##################################################################################

    # Convenience function for implementing the reciprocal theorem
    def reciprocal_coefficients(inertia_term,string=None):
        Fx = -assemble(dot(inertia_term,v_ex)*dx)
        Fy = -assemble(dot(inertia_term,v_ey)*dx)
        Fz = -assemble(dot(inertia_term,v_ez)*dx)
        Tx = -assemble(dot(inertia_term,v_wx)*dx)
        Ty = -assemble(dot(inertia_term,v_wy)*dx)
        Tz = -assemble(dot(inertia_term,v_wz)*dx)
        if MPI_size>1:
            Fx,Fy,Fz,Tx,Ty,Tz = comm.allreduce(np.array([Fx,Fy,Fz,Tx,Ty,Tz]),MPI.SUM)
        if MPI_rank==0 and (string is not None):
            print(string,Fx,Fy,Fz,Tx,Ty,Tz)
        return np.array([Fx,Fy,Fz,Tx,Ty,Tz])

    # Now calculate the inertial lift force utilising the reciprocal theorem...
    Theta_val = res[1]/(R+px)
    Theta = Constant(ez*Theta_val)
    inertia_term = cross(Theta,v0)\
                   +dot(v0,nabla_grad(ubar_fun))\
                   +dot(v0+ubar_fun-cross(Theta,Xc+R_vec),nabla_grad(v0)) # Note addition of R_vec because of mesh location
    Lx,Ly,Lz,Tx,Ty,Tz = reciprocal_coefficients(inertia_term,"Reciprocal inertial lift and torque:")

    # Calculate the centripetal and centrifugal contributions...
    particle_mass = 4.0/3.0*np.pi*pr**3 # assume unit density
    centrifugal = particle_mass*Theta_val**2*(R+px)
    if MPI_rank==0:
        print("Centrifugal force:",centrifugal)

    # Note: It would be better to create a finer surface mesh of the particle to estimate this more accurately...
    centripetal = assemble(dot(-n,outer(ubar_fun,ubar_fun))[0]*ds(particle_marker))
    centripetal = comm.allreduce(centripetal,MPI.SUM)
    if MPI_rank==0:
        print("Centripetal force:",centripetal)

    print("F_p_x =", Lx)
    print("F_p_z =", Lz)

    return Lx, Lz