from netgen.csg import *
from firedrake import Mesh, COMM_WORLD
import netgen
import math, netgen.libngpy
from netgen.occ import Pnt, ArcOfCircle, Wire, WorkPlane, Axes, Pipe, Sphere, OCCGeometry, X, Y, Z



def make_curved_channel_section_with_spherical_hole(R, W, H, L, a, h, r_off=0.0, z_off=0.0, order=3, global_maxh=None):

    theta = L / R

    p0 = Pnt(R*math.cos(0.0),       R*math.sin(0.0),       0.0)
    pm = Pnt(R*math.cos(theta*0.5), R*math.sin(theta*0.5), 0.0)
    p1 = Pnt(R*math.cos(theta),     R*math.sin(theta),     0.0)
    spine = Wire([ArcOfCircle(p0, pm, p1)])

    wp = WorkPlane(Axes((p0.x, p0.y, p0.z), n=Y, h=Z))
    rect_face = wp.RectangleC(W, H).Face()

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
        global_maxh = 0.3*max(W,H)

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