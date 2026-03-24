import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import tempfile
import numpy as np
import gmsh
from firedrake import *


def make_curved_channel_section_with_spherical_hole(R, H, W, L, a, particle_maxh, global_maxh,
                                                     r_off=0.0, z_off=0.0, order=2, comm=COMM_SELF):
    """
    Gmsh-based drop-in replacement for the Netgen version in build_3d_geometry_netgen.py.

    Coordinate system (identical to the Netgen version):
      - Arc center at the origin
      - Channel sweeps from angle 0 to theta = L/R around the z-axis
      - Cross-section: width W (radial / x), height H (vertical / z)
      - Particle at ((R+r_off)*cos(theta/2), (R+r_off)*sin(theta/2), z_off)

    The 'scaling' parameter is accepted for API compatibility but not needed by gmsh.
    """

    theta = L / R

    cx = (R + r_off) * math.cos(theta * 0.5)
    cy = (R + r_off) * math.sin(theta * 0.5)
    cz = z_off

    INLET, OUTLET, WALLS, PARTICLE, FLUID_VOL = 1, 2, 3, 4, 11

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", global_maxh)

    # Cross-section at angle 0 in the x-z plane (y = 0), centred at (R, 0, 0)
    p1 = gmsh.model.occ.addPoint(R - W / 2, 0, -H / 2)
    p2 = gmsh.model.occ.addPoint(R + W / 2, 0, -H / 2)
    p3 = gmsh.model.occ.addPoint(R + W / 2, 0, +H / 2)
    p4 = gmsh.model.occ.addPoint(R - W / 2, 0, +H / 2)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    face = gmsh.model.occ.addPlaneSurface([cl])

    # Revolve the cross-section around the z-axis by angle theta
    rev = gmsh.model.occ.revolve([(2, face)], 0, 0, 0, 0, 0, 1, theta)

    duct_tag = None
    for dim, tag in rev:
        if dim == 3:
            duct_tag = tag
            break
    assert duct_tag is not None, "Revolve did not produce a volume"

    # Boolean-subtract the particle sphere
    particle_sphere = gmsh.model.occ.addSphere(cx, cy, cz, a)
    gmsh.model.occ.cut([(3, duct_tag)], [(3, particle_sphere)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=3)
    assert len(volumes) == 1, f"Expected 1 volume, found {len(volumes)}"
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], FLUID_VOL)
    gmsh.model.setPhysicalName(3, FLUID_VOL, "Fluid volume")

    surfaces = gmsh.model.occ.getEntities(dim=2)
    wall_tags = []
    particle_tags = []
    inlet_tag = None
    outlet_tag = None

    inlet_com_exp = np.array([R, 0.0, 0.0])
    outlet_com_exp = np.array([R * np.cos(theta), R * np.sin(theta), 0.0])
    particle_com_exp = np.array([cx, cy, cz])
    cross_area = W * H
    tol = 1.0e-3 * min(W, H)

    for surface in surfaces:
        com = np.array(gmsh.model.occ.getCenterOfMass(surface[0], surface[1]))
        area = gmsh.model.occ.getMass(surface[0], surface[1])

        if inlet_tag is None and np.allclose(com, inlet_com_exp, atol=tol) and np.isclose(area, cross_area, rtol=0.15):
            inlet_tag = surface[1]
        elif outlet_tag is None and np.allclose(com, outlet_com_exp, atol=tol) and np.isclose(area, cross_area, rtol=0.15):
            outlet_tag = surface[1]
        elif np.allclose(com, particle_com_exp, atol=2 * a) and area <= 1.2 * 4.0 * np.pi * a ** 2:
            particle_tags.append(surface[1])
        else:
            wall_tags.append(surface[1])

    def _debug_surfaces(surfaces, inlet_com, outlet_com, particle_com, cross_area):
        """Print diagnostic information when surface identification fails."""
        print("\n=== SURFACE IDENTIFICATION DEBUG ===")
        print(f"Expected inlet  CoM: {inlet_com},   area: {cross_area:.6f}")
        print(f"Expected outlet CoM: {outlet_com},  area: {cross_area:.6f}")
        print(f"Expected particle CoM: {particle_com}")
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            area = gmsh.model.occ.getMass(surface[0], surface[1])
            print(f"  Surface {surface}: CoM={np.array(com)}, area={area:.6f}")
        print("====================================\n")

    if inlet_tag is None or outlet_tag is None or len(particle_tags) == 0:
        _debug_surfaces(surfaces, inlet_com_exp, outlet_com_exp, particle_com_exp, cross_area)
    assert inlet_tag is not None, "Could not identify inlet surface"
    assert outlet_tag is not None, "Could not identify outlet surface"
    assert len(particle_tags) >= 1, "Particle surface not found"
    assert len(wall_tags) == 4, f"Expected 4 wall surfaces, found {len(wall_tags)}"

    gmsh.model.addPhysicalGroup(2, [inlet_tag], INLET)
    gmsh.model.setPhysicalName(2, INLET, "inlet")
    gmsh.model.addPhysicalGroup(2, [outlet_tag], OUTLET)
    gmsh.model.setPhysicalName(2, OUTLET, "outlet")
    gmsh.model.addPhysicalGroup(2, wall_tags, WALLS)
    gmsh.model.setPhysicalName(2, WALLS, "walls")
    gmsh.model.addPhysicalGroup(2, particle_tags, PARTICLE)
    gmsh.model.setPhysicalName(2, PARTICLE, "particle")

    # --- Mesh refinement near particle (Brendan's strategy) ---
    # Brendan uses an internal meshsize derived from the channel geometry, not the
    # user-provided global_maxh.  The user's global_maxh still acts as a hard cap
    # via MeshSizeMax (set above), so the far-field never exceeds it.
    meshsize = min(W, H) / 2.5

    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "SurfacesList", particle_tags)

    thresh_fine = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_fine, "IField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_fine, "LcMin", particle_maxh)
    gmsh.model.mesh.field.setNumber(thresh_fine, "LcMax", meshsize)
    gmsh.model.mesh.field.setNumber(thresh_fine, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_fine, "DistMax", 2.0 * min(W, H))

    thresh_mid = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh_mid, "IField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh_mid, "LcMin", meshsize / 4)
    gmsh.model.mesh.field.setNumber(thresh_mid, "LcMax", meshsize)
    gmsh.model.mesh.field.setNumber(thresh_mid, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(thresh_mid, "DistMax", 4.0 * min(W, H))

    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thresh_fine, thresh_mid])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)

    gmsh.model.mesh.generate(3)

    if order > 1:
        gmsh.model.mesh.setOrder(order)

    try:
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize()
        gmsh.model.mesh.optimize("Netgen")
        gmsh.model.mesh.optimize()
    except Exception:
        gmsh.model.mesh.optimize()

    fd, tmp_path = tempfile.mkstemp(suffix=".msh")
    os.close(fd)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(tmp_path)
    gmsh.finalize()

    mesh3d = Mesh(tmp_path, comm=comm)

    try:
        os.unlink(tmp_path)
    except OSError:
        pass

    tags = {
        "walls":           (WALLS,),
        "inlet":           INLET,
        "outlet":          OUTLET,
        "particle":        PARTICLE,
        "theta":           theta,
        "particle_center": (cx, cy, cz),
    }

    return mesh3d, tags


if __name__ == "__main__":

    TEST_R = 160.0
    TEST_a = 0.05
    TEST_H = 2.0
    TEST_W = 2.0
    TEST_L = 10.0
    TEST_r_off = 0.0
    TEST_z_off = 0.0

    print("Generating gmsh mesh for sanity checks...")
    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R=TEST_R, H=TEST_H, W=TEST_W, L=TEST_L, a=TEST_a,
        particle_maxh=TEST_a / 3, global_maxh=1.0,
        r_off=TEST_r_off, z_off=TEST_z_off, order=2
    )

    V = FunctionSpace(mesh3d, "CG", 1)
    dummy = Function(V, name="Mesh_Visualization").assign(1.0)

    VTKFile("correct_channel.pvd").write(dummy)
    exit()
    x, y, z = SpatialCoordinate(mesh3d)

    print("\n" + "=" * 40)
    print("=== STARTING SANITY CHECKS ===")
    print("=" * 40)

    coords = mesh3d.coordinates.dat.data_ro
    min_z, max_z = coords[:, 2].min(), coords[:, 2].max()
    height_z = max_z - min_z
    print(f"\n1. CHANNEL VERTICAL EXTENT (expected ~H={TEST_H}): {height_z:.4f}")

    print("\n2. SURFACE AREAS (Inlet / Outlet / Particle)")
    expected_cross_section = TEST_W * TEST_H
    inlet_area = assemble(1.0 * ds(tags["inlet"], domain=mesh3d))
    outlet_area = assemble(1.0 * ds(tags["outlet"], domain=mesh3d))
    print(f"   Inlet Area:  Expected {expected_cross_section:.4f}, Measured {inlet_area:.4f}")
    print(f"   Outlet Area: Expected {expected_cross_section:.4f}, Measured {outlet_area:.4f}")

    expected_particle_area = 4.0 * math.pi * TEST_a ** 2
    actual_particle_area = assemble(1.0 * ds(tags["particle"], domain=mesh3d))
    err_area = abs(expected_particle_area - actual_particle_area) / expected_particle_area
    print(f"   Particle Surface Area: Expected {expected_particle_area:.4f}, Measured {actual_particle_area:.4f} (Rel Err: {err_area:.2%})")

    print("\n3. TOTAL VOLUME")
    expected_vol = (TEST_W * TEST_H * TEST_L) - ((4.0 / 3.0) * math.pi * TEST_a ** 3)
    actual_vol = assemble(1.0 * dx(domain=mesh3d))
    err_vol = abs(expected_vol - actual_vol) / expected_vol
    print(f"   Volume: Expected {expected_vol:.4f}, Measured {actual_vol:.4f} (Rel Err: {err_vol:.2%})")

    print("\n4. PARTICLE CENTROID")
    cx_mesh = assemble(x * ds(tags["particle"], domain=mesh3d)) / actual_particle_area
    cy_mesh = assemble(y * ds(tags["particle"], domain=mesh3d)) / actual_particle_area
    cz_mesh = assemble(z * ds(tags["particle"], domain=mesh3d)) / actual_particle_area
    cx_exp, cy_exp, cz_exp = tags["particle_center"]
    print(f"   Expected Center: ({cx_exp:.5f}, {cy_exp:.5f}, {cz_exp:.5f})")
    print(f"   Measured Center: ({cx_mesh:.5f}, {cy_mesh:.5f}, {cz_mesh:.5f})")
    err_center = math.sqrt((cx_exp - cx_mesh) ** 2 + (cy_exp - cy_mesh) ** 2 + (cz_exp - cz_mesh) ** 2)
    print(f"   -> Absolute L2 distance error: {err_center:.2e}")

    print("\n" + "=" * 40)
    print("=== SANITY CHECKS COMPLETED ===")
    print("=" * 40)
