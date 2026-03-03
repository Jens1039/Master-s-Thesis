import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
from firedrake import *
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


if __name__ == "__main__":

    TEST_R = 220.0
    TEST_a = 0.1
    TEST_H = 2.0
    TEST_W = 2.0
    TEST_L = 10.0
    TEST_r_off = 0.5
    TEST_z_off = 0.0

    print("Generating mesh for sanity checks...")
    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R=TEST_R, H=TEST_H, W=TEST_W, L=TEST_L, a=TEST_a,
        particle_maxh=TEST_a / 3, global_maxh=1.0,
        scaling="second_nondimensionalisation",
        r_off=TEST_r_off, z_off=TEST_z_off, order=6
    )

    x, y, z = SpatialCoordinate(mesh3d)

    print("\n" + "=" * 40)
    print("=== STARTING SANITY CHECKS ===")
    print("=" * 40)

    # 1. Bounding Box & Aspect Ratio Check
    # Verifies if H and W are swapped (common RectangleC orientation issue)
    coords = mesh3d.coordinates.dat.data_ro
    min_x, max_x = coords[:, 0].min(), coords[:, 0].max()
    min_z, max_z = coords[:, 2].min(), coords[:, 2].max()

    width_x = max_x - min_x
    height_z = max_z - min_z

    print("\n1. CHANNEL DIMENSIONS AND ORIENTATION")
    print(f"   Radial extent (X-axis, expected ~W): {width_x:.4f}")
    print(f"   Vertical extent (Z-axis, expected ~H): {height_z:.4f}")

    if abs(width_x - TEST_H) < 1e-2 and abs(height_z - TEST_W) < 1e-2 and TEST_W != TEST_H:
        print("   -> WARNING: X maps to H and Z maps to W! Your channel is likely rotated 90 degrees.")
        print("   -> Suggestion: Change RectangleC(_H, _W) to RectangleC(_W, _H).")
    elif abs(width_x - TEST_W) < 1e-2 and abs(height_z - TEST_H) < 1e-2:
        print("   -> OK: Orientation appears correct.")

    # 2. Boundary Surface Areas
    print("\n2. SURFACE AREAS (Inlet / Outlet / Particle)")
    expected_cross_section = TEST_W * TEST_H
    inlet_area = assemble(1.0 * ds(tags["inlet"], domain=mesh3d))
    outlet_area = assemble(1.0 * ds(tags["outlet"], domain=mesh3d))

    print(f"   Inlet Area:  Expected {expected_cross_section:.4f}, Measured {inlet_area:.4f}")
    print(f"   Outlet Area: Expected {expected_cross_section:.4f}, Measured {outlet_area:.4f}")

    expected_particle_area = 4.0 * math.pi * TEST_a ** 2
    actual_particle_area = assemble(1.0 * ds(tags["particle"], domain=mesh3d))
    err_area = abs(expected_particle_area - actual_particle_area) / expected_particle_area
    print(
        f"   Particle Surface Area: Expected {expected_particle_area:.4f}, Measured {actual_particle_area:.4f} (Relative Error: {err_area:.2%})")

    # 3. Total Mesh Volume
    print("\n3. TOTAL VOLUME")
    # Volume = (Cross-section * Arc Length L) - Sphere Volume
    expected_vol = (TEST_W * TEST_H * TEST_L) - ((4.0 / 3.0) * math.pi * TEST_a ** 3)
    actual_vol = assemble(1.0 * dx(domain=mesh3d))
    err_vol = abs(expected_vol - actual_vol) / expected_vol
    print(f"   Volume: Expected {expected_vol:.4f}, Measured {actual_vol:.4f} (Relative Error: {err_vol:.2%})")

    # 4. Particle Centroid Position
    print("\n4. PARTICLE CENTROID")
    cx_mesh = assemble(x * ds(tags["particle"], domain=mesh3d)) / actual_particle_area
    cy_mesh = assemble(y * ds(tags["particle"], domain=mesh3d)) / actual_particle_area
    cz_mesh = assemble(z * ds(tags["particle"], domain=mesh3d)) / actual_particle_area
    cx_exp, cy_exp, cz_exp = tags["particle_center"]

    print(f"   Expected Center: ({cx_exp:.5f}, {cy_exp:.5f}, {cz_exp:.5f})")
    print(f"   Measured Center: ({cx_mesh:.5f}, {cy_mesh:.5f}, {cz_mesh:.5f})")
    err_center = math.sqrt((cx_exp - cx_mesh) ** 2 + (cy_exp - cy_mesh) ** 2 + (cz_exp - cz_mesh) ** 2)
    print(f"   -> Absolute L2 distance error: {err_center:.2e}")

    # 5. Geometric Boundary Integrity
    print("\n5. BOUNDARY ALIGNMENT")
    # Inlet should lie strictly on the plane where y = 0 (relative to local start)
    y_inlet_mean = assemble(abs(y) * ds(tags["inlet"], domain=mesh3d)) / inlet_area
    print(f"   Mean Y-deviation at Inlet (should be ~0): {y_inlet_mean:.2e}")

    # Outlet should lie strictly on the plane defined by the arc angle theta
    theta = tags["theta"]
    # Plane equation for the outlet: x * sin(theta) - y * cos(theta) = 0
    outlet_plane_dev = assemble(
        abs(x * math.sin(theta) - y * math.cos(theta)) * ds(tags["outlet"], domain=mesh3d)) / outlet_area
    print(f"   Mean plane-deviation at Outlet (should be ~0): {outlet_plane_dev:.2e}")

    print("\n" + "=" * 40)
    print("=== SANITY CHECKS COMPLETED ===")
    print("=" * 40)