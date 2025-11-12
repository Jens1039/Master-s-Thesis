from netgen.csg import *
from firedrake import Mesh, COMM_WORLD
import netgen
import math, netgen.libngpy
from netgen.occ import Pnt, ArcOfCircle, Wire, WorkPlane, Axes, Pipe, Sphere, OCCGeometry, X, Y, Z



def make_curved_channel_section_with_spherical_hole(R, W, H, L, a, particle_maxh, global_maxh, r_off=0.0, z_off=0.0, order=3):

    theta = L / R

    # creates a point with the coordinates (R, 0, 0) cross-sectional middle point of our curved duct section
    p_0 = Pnt(R*math.cos(0.0),       R*math.sin(0.0),       0.0)
    # creates a point halfway through the curved duct section
    p_m = Pnt(R*math.cos(theta*0.5), R*math.sin(theta*0.5), 0.0)
    # creates a point at the end of the cross-section
    p_1 = Pnt(R*math.cos(theta),     R*math.sin(theta),     0.0)

    # ArcOfCircle constructs the geometry, Wire adds the topology
    spine = Wire([ArcOfCircle(p_0, p_m, p_1)])

    # create a plane with "anchor-point" (p_0.x, p_0.y, p_0.z) normal vector Y and horizontal direction Z
    wp = WorkPlane(Axes((p_0.x, p_0.y, p_0.z), n=Y, h=Z))

    # the 2d shape, which is later extruded along the spine is created here on the workplane
    rect_face = wp.RectangleC(W, H).Face()

    # extrudes the 2d rectangular face along the spine to create a 3d manifold
    channel_section = Pipe(spine, rect_face)

    # Initially, all the facets are identified as walls - we differentiate that later
    channel_section.faces.name = "walls"

    # cylindrical coordinates of the sphere center
    cx = (R + r_off) * math.cos(theta*0.5)
    cy = (R + r_off) * math.sin(theta*0.5)
    cz = z_off

    # creates a sphere with radius a around the previously determined particle center
    sphere_filled = Sphere(Pnt(cx, cy, cz), a)
    sphere_filled.faces.name = "particle"

    # creates the final geometrie curved channel section with spherical hole
    fluid = channel_section - sphere_filled

    # differentiates inlet and outlet from the other walls by proximity to the starting and end point of the channel section
    fluid.faces.Nearest((p_0.x, p_0.y, p_0.z)).name = "inlet"
    fluid.faces.Nearest((p_1.x, p_1.y, p_1.z)).name = "outlet"

    # Locating the sphere surface
    particle_surface = fluid.faces.Nearest((cx + a, cy, cz))
    particle_surface.name = "particle"
    particle_surface.maxh = particle_maxh

    # preparation for clean parallel solving later
    if COMM_WORLD.rank == 0:
        netgenmesh = OCCGeometry(fluid, dim=3).GenerateMesh(maxh=global_maxh)

    else:
        netgenmesh = netgen.libngpy._meshing.Mesh(3)


    if order and order >= 2:
        mesh3d = Mesh(Mesh(netgenmesh, comm=COMM_WORLD).curve_field(order))
    else:
        mesh3d = Mesh(netgenmesh, comm=COMM_WORLD)

    names = netgenmesh.GetRegionNames(codim=1)
    def _id(name):
        return names.index(name) + 1 if name in names else None

    tags = {
        "walls":    _id("walls"),
        "inlet":    _id("inlet"),
        "outlet":   _id("outlet"),
        "particle": _id("particle"),
        "theta":    theta,
        "center":   (cx, cy, cz),
    }
    return mesh3d, tags