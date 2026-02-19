import math
from firedrake import Mesh, COMM_SELF
from netgen.occ import *


def make_curved_channel_section_with_spherical_hole(R, H, W, L, a, particle_maxh, global_maxh, scaling="second_nondimensionalisation", r_off=0.0, z_off=0.0, order=3, comm=COMM_SELF):

    # We need to scale down the coordinates, since Netgen struggles with very large coordinates.
    if scaling == "second_nondimensionalisation":
        SECURE_a = 0.025
    elif scaling == "first_nondimensionalisation":
        SECURE_a = 1
    else:
        raise ValueError(f"Unknown scaling {scaling}")

    _R = R * SECURE_a
    _H = H * SECURE_a
    _W = W * SECURE_a
    _L = L * SECURE_a
    _a = a * SECURE_a
    _particle_maxh = particle_maxh * SECURE_a
    _global_maxh = global_maxh * SECURE_a
    _r_off = r_off * SECURE_a
    _z_off = z_off * SECURE_a

    theta = _L / _R

    p_0 = Pnt(_R * math.cos(0.0), _R * math.sin(0.0), 0.0)
    p_m = Pnt(_R * math.cos(theta * 0.5), _R * math.sin(theta * 0.5), 0.0)
    p_1 = Pnt(_R * math.cos(theta), _R * math.sin(theta), 0.0)

    spine = Wire([ArcOfCircle(p_0, p_m, p_1)])
    wp = WorkPlane(Axes((p_0.x, p_0.y, p_0.z), n=Y, h=Z))
    rect_face = wp.RectangleC(_H, _W).Face()
    channel_section = Pipe(spine, rect_face)

    cx = (_R + _r_off) * math.cos(theta * 0.5)
    cy = (_R + _r_off) * math.sin(theta * 0.5)
    cz = _z_off

    sphere_filled = Sphere(Pnt(cx, cy, cz), _a)
    sphere_filled.faces.name = "particle"

    fluid = channel_section - sphere_filled

    fluid.faces.name = "walls"

    fluid.faces.Nearest((p_0.x, p_0.y, p_0.z)).name = "inlet"
    fluid.faces.Nearest((p_1.x, p_1.y, p_1.z)).name = "outlet"

    particle_surface = fluid.faces.Nearest((cx + _a, cy, cz))
    particle_surface.name = "particle"
    particle_surface.maxh = _particle_maxh

    netgenmesh = OCCGeometry(fluid, dim=3).GenerateMesh(maxh=_global_maxh)
    netgenmesh.Curve(order)

    netgenmesh.Scale(1.0 / SECURE_a)
    cx *= 1.0 / SECURE_a
    cy *= 1.0 / SECURE_a
    cz *= 1.0 / SECURE_a

    mesh3d = Mesh(netgenmesh, comm=comm)

    names = netgenmesh.GetRegionNames(codim=1)

    def _ids(name):
        ids = tuple(i + 1 for i, nm in enumerate(names) if nm == name)
        if not ids:
            raise ValueError(f"{name} not found")
        return ids

    tags = {
        "walls": _ids("walls"),
        "inlet": _ids("inlet")[0],
        "outlet": _ids("outlet")[0],
        "particle": _ids("particle")[0],
        "theta": theta,
        "particle_center": (cx, cy, cz),
    }

    return mesh3d, tags