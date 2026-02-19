import gmsh

from firedrake import *

from mpi4py import MPI


def GenerateCurvedDuctParticleMesh(W, H, R, px, pz, pr, L, filename=None, comm=None):
    """Use gmsh to generate a curved rectanglular duct with a spherical particle removed.
    It is setup so the curve centre is (-R,0,0) and the central cross-section is
    in the x-z plane centred at the origin (i.e. flow is primarily along the y-axis
    through the central cross-section).
    A particle with radius pr centered in the plane y=0 is removed from the mesh.
    The mesh is finally converted to a dolfinx compatible mesh."""

    # Use provided communicator or default to COMM_WORLD
    if comm is None:
        comm = MPI.COMM_WORLD
    
    MPI_rank = comm.Get_rank()
    MPI_size = comm.Get_size()

    # Set some additional parameters describing the geometry and mesh resolution
    # py = 0.0
    theta = 0.5 * L / R
    meshsize = min(H, W) / 2.5
    # Setup a dictionary for facet markers
    facet_markers = {"inlet": 1,
                     "outlet": 2,
                     "walls": 3,
                     "particle": 4}
    # Generate a mesh using gmsh (this only needs to be done on one process!)
    if MPI_rank == 0:
        # Initialise gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        # meshsize settings
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        # Describe the cross-section
        p1 = gmsh.model.occ.addPoint(-W / 2, 0, -H / 2)
        p2 = gmsh.model.occ.addPoint(+W / 2, 0, -H / 2)
        p3 = gmsh.model.occ.addPoint(+W / 2, 0, +H / 2)
        p4 = gmsh.model.occ.addPoint(-W / 2, 0, +H / 2)
        l1 = gmsh.model.occ.addLine(p1, p2)
        l2 = gmsh.model.occ.addLine(p2, p3)
        l3 = gmsh.model.occ.addLine(p3, p4)
        l4 = gmsh.model.occ.addLine(p4, p1)
        c1 = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
        s1 = gmsh.model.occ.addPlaneSurface([c1])
        # Define the line/wire along which the cross-section will be extruded
        l5 = gmsh.model.occ.addCircle(-R, 0, 0, R, angle1=-theta, angle2=+theta)
        w1 = gmsh.model.occ.addWire([l5])
        # Do the extrusion (via occ.addPipe, could alternatively use occ.revolve)
        # (May need to check "trihedron" argument to addPipe.)
        extrusion = gmsh.model.occ.addPipe([(2, s1)], w1, "Frenet")
        assert len(extrusion) == 1  # only one result
        assert extrusion[0][0] == 3  # which is three dimensional
        duct = extrusion[0][1]
        # Remove the particle
        particle = gmsh.model.occ.addSphere(px, 0.0, pz, pr)
        fluid = gmsh.model.occ.cut([(3, duct)], [(3, particle)])
        gmsh.model.occ.synchronize()
        # Fetch and label the fluid volume
        volumes = gmsh.model.getEntities(dim=3)
        assert len(volumes) == 1  # only one result
        assert volumes == fluid[0]  # matches our fluid domain
        fluid_marker = 11
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
        gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid volume")
        # Fetch and label the surfaces
        surfaces = gmsh.model.occ.getEntities(dim=2)
        walls = []
        particles = []
        inlet, outlet = None, None
        curve_offset = -R + R * np.sin(theta) / theta
        for surface in surfaces:
            CoM = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            area = gmsh.model.occ.getMass(surface[0], surface[1])
            # print(surface,CoM,area)
            if np.allclose(CoM, [R * (-1 + np.cos(-theta)), R * np.sin(-theta), 0]):
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], facet_markers["inlet"])
                inlet = surface[1]
                gmsh.model.setPhysicalName(surface[0], facet_markers["inlet"], "inlet")
            elif np.allclose(CoM, [R * (-1 + np.cos(+theta)), R * np.sin(+theta), 0]):
                gmsh.model.addPhysicalGroup(surface[0], [surface[1]], facet_markers["outlet"])
                outlet = surface[1]
                gmsh.model.setPhysicalName(surface[0], facet_markers["outlet"], "outlet")
            elif (np.allclose(CoM, [px, 0, pz]) and area <= 1.1 * (4.0 * np.pi * pr ** 2)):
                # There is a potential issue where, if pz=0, the curved outer wall
                # could have a CoM close to the particle CoM, hence the area check also.
                particles.append(surface[1])
            elif (np.allclose([abs(CoM[2]), area], [H / 2, L * W]) or
                  (np.allclose(CoM, [curve_offset - W / 2, 0, 0], atol=1.0E-3 * min(W, H)) and np.allclose([area],
                                                                                                           [H * 2 * theta * (
                                                                                                                   R - W / 2)])) or
                  (np.allclose(CoM, [curve_offset + W / 2, 0, 0], atol=1.0E-3 * min(W, H)) and np.allclose([area],
                                                                                                           [H * 2 * theta * (
                                                                                                                   R + W / 2)]))):
                walls.append(surface[1])
            else:
                print("Warning: a surface has not been characterised:", surface, CoM, area)
        # Perform various checks on surface labels...
        assert inlet is not None
        assert outlet is not None
        assert len(walls) == 4  # expect exactly four side walls
        gmsh.model.addPhysicalGroup(2, walls, facet_markers["walls"])
        gmsh.model.setPhysicalName(2, facet_markers["walls"], "walls")
        assert len(particles) == 1  # expect exactly one particle
        gmsh.model.addPhysicalGroup(2, particles, facet_markers["particle"])
        gmsh.model.setPhysicalName(2, facet_markers["particle"], "particle")
        # Setup refinement
        # First create field of distance from particle (surface)
        distance_from_particle = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_from_particle, "SurfacesList", particles)
        # Create a near particle field
        threshold1 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold1, "IField", distance_from_particle)
        gmsh.model.mesh.field.setNumber(threshold1, "LcMin", pr / 4)
        gmsh.model.mesh.field.setNumber(threshold1, "LcMax", meshsize)
        gmsh.model.mesh.field.setNumber(threshold1, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold1, "DistMax", 2.0 * min(W, H))
        # Create an intermediate particle field
        threshold2 = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold2, "IField", distance_from_particle)
        gmsh.model.mesh.field.setNumber(threshold2, "LcMin", meshsize / 4)
        gmsh.model.mesh.field.setNumber(threshold2, "LcMax", meshsize)
        gmsh.model.mesh.field.setNumber(threshold2, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(threshold2, "DistMax", 4.0 * min(W, H))
        # Create a minimum of the two fields
        minfield = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(minfield, "FieldsList", [threshold1, threshold2])
        # Set the desired refinement field
        gmsh.model.mesh.field.setAsBackgroundMesh(minfield)
        # mesh generation
        gmsh.model.mesh.generate(3)
        # optionally run several optimizer sweeps
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize()
        # optionally save
        if MPI_rank == 0:
            if filename is not None:
                gmsh.write(filename)
            else:
                filename = "curved_duct.msh"

            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write(filename)
        gmsh.finalize()
        comm.Barrier()
        mesh = Mesh(filename, comm=comm)
        plex = mesh.topology_dm
        cell_tags = plex.getLabel("Cell Sets")
        facet_tags = plex.getLabel("Face Sets")
        comm.Barrier()
        # Return
        return facet_markers, mesh, cell_tags, facet_tags


if __name__ == "__main__":
    GenerateCurvedDuctParticleMesh(2.0, 2.0, 160.0, 0.0, 0.0, 0.05, 8.0)

