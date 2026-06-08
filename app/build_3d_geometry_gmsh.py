import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import tempfile
import numpy as np
import gmsh
from firedrake import *


def make_curved_channel_section_with_spherical_hole(R, H, W, L, a, particle_maxh, global_maxh, x_off=0.0, z_off=0.0, order=2, comm=COMM_SELF):

    theta = L / R

    cx = (R + x_off) * math.cos(theta * 0.5)
    cy = (R + x_off) * math.sin(theta * 0.5)
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


def _strip_periodic_section(msh_path):
    """Remove the ``$Periodic`` ... ``$EndPeriodic`` block from an MSH 2.2 file.

    ``gmsh.model.mesh.setPeriodic`` does its job at mesh-generation time: it
    copies the master surface mesh onto the slave through the affine map, so the
    written nodes/elements are already mirror-paired. But gmsh ALSO records the
    master/slave node correspondence in a ``$Periodic`` section, and
    Firedrake/DMPlex reads that section as a genuine periodic identification --
    it topologically glues the two hemisphere cavity facets together, so the
    particle stops being an exterior boundary and ``ds(particle)`` collapses to
    an empty subdomain.

    We only want the mesh-copy side effect, not the identification. Deleting the
    section after writing leaves a standalone mesh whose particle surface is
    still mirror-symmetric (verified to ~1e-15) but whose facets are read as a
    normal exterior boundary again.
    """
    with open(msh_path) as fh:
        lines = fh.readlines()
    out, skip = [], False
    for ln in lines:
        if ln.startswith("$Periodic"):
            skip = True
            continue
        if ln.startswith("$EndPeriodic"):
            skip = False
            continue
        if not skip:
            out.append(ln)
    with open(msh_path, "w") as fh:
        fh.writelines(out)


def make_curved_channel_section_with_spherical_hole_symmetric(
        R, H, W, L, a, particle_maxh, global_maxh,
        x_off=0.0, z_off=0.0, order=2, comm=COMM_SELF,
        apply_periodic=True, debug=False):
    """Stufe-1 variant: particle SURFACE mesh mirror-symmetric about z = z_off.

    Independent copy of ``make_curved_channel_section_with_spherical_hole``.
    The ONLY differences are in the particle construction + meshing:

      1. The particle ball is split at the plane z = cz (= z_off) into an
         upper and a lower hemisphere (two coincident balls each intersected
         with a half-space box, then ``fragment``-glued so they share the
         equatorial circle conformally).
      2. After the boolean cut, the two hemisphere cavity surfaces are tagged
         separately, and the LOWER one is made a periodic *reflection* of the
         UPPER via ``setPeriodic`` with the affine map  z -> 2*cz - z.

    Consequence: the surface mesh (facets, normals, quadrature points) on the
    particle is exactly mirror-paired about z = cz. The spurious antisymmetric
    part of the traction integral  ∫ σ·n·e_z ds(particle)  cancels to machine
    precision, so the F_z noise floor (~1e-7 with the unstructured sphere)
    drops. At z_off = 0 the whole geometry is symmetric and F_z ≡ 0 to ~eps;
    off-axis the genuine physical F_z survives while the mesh noise is removed
    (the Stokes disturbance decays fast, so the asymmetric far-field is
    irrelevant to the surface integral).

    Stufe 1 only makes the SURFACE symmetric; the near-field VOLUME (boundary
    layer) is still grown by the unstructured mesher, so a small solution-side
    residual may remain. If the F_z self-check is not low enough, Stufe 2 adds
    a symmetric boundary layer.

    Returns ``(mesh3d, tags)`` exactly like the original. ``tags`` additionally
    carries ``particle_upper`` / ``particle_lower`` physical ids for diagnostics.
    """
    theta = L / R

    cx = (R + x_off) * math.cos(theta * 0.5)
    cy = (R + x_off) * math.sin(theta * 0.5)
    cz = z_off

    INLET, OUTLET, WALLS, PARTICLE, FLUID_VOL = 1, 2, 3, 4, 11
    PARTICLE_UP, PARTICLE_LO = 5, 6

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

    # ── Particle as two hemispheres split at z = cz ────────────────────────
    # Two coincident balls; each intersected with a half-space box so we get
    # the upper (z >= cz) and lower (z <= cz) half-ball as separate solids.
    pad = 4.0 * a
    ball_u = gmsh.model.occ.addSphere(cx, cy, cz, a)
    ball_l = gmsh.model.occ.addSphere(cx, cy, cz, a)
    box_u  = gmsh.model.occ.addBox(cx - pad, cy - pad, cz,       2*pad, 2*pad, pad)
    box_l  = gmsh.model.occ.addBox(cx - pad, cy - pad, cz - pad, 2*pad, 2*pad, pad)

    upper = gmsh.model.occ.intersect([(3, ball_u)], [(3, box_u)],
                                     removeObject=True, removeTool=True)[0]
    lower = gmsh.model.occ.intersect([(3, ball_l)], [(3, box_l)],
                                     removeObject=True, removeTool=True)[0]

    # Glue the two half-balls so they share the equatorial disk conformally,
    # then subtract the (now two-piece) ball from the duct.
    glued, _ = gmsh.model.occ.fragment(upper, lower)
    tool = [dt for dt in glued if dt[0] == 3]
    gmsh.model.occ.cut([(3, duct_tag)], tool)
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=3)
    assert len(volumes) == 1, f"Expected 1 volume, found {len(volumes)}"
    gmsh.model.addPhysicalGroup(3, [volumes[0][1]], FLUID_VOL)
    gmsh.model.setPhysicalName(3, FLUID_VOL, "Fluid volume")

    surfaces = gmsh.model.occ.getEntities(dim=2)
    wall_tags = []
    particle_up_tags = []
    particle_lo_tags = []
    inlet_tag = None
    outlet_tag = None

    inlet_com_exp = np.array([R, 0.0, 0.0])
    outlet_com_exp = np.array([R * np.cos(theta), R * np.sin(theta), 0.0])
    particle_com_exp = np.array([cx, cy, cz])
    cross_area = W * H
    tol = 1.0e-3 * min(W, H)
    hemi_area_cap = 0.7 * 4.0 * np.pi * a ** 2   # each hemisphere ~ 2πa²

    for surface in surfaces:
        com = np.array(gmsh.model.occ.getCenterOfMass(surface[0], surface[1]))
        area = gmsh.model.occ.getMass(surface[0], surface[1])

        if inlet_tag is None and np.allclose(com, inlet_com_exp, atol=tol) and np.isclose(area, cross_area, rtol=0.15):
            inlet_tag = surface[1]
        elif outlet_tag is None and np.allclose(com, outlet_com_exp, atol=tol) and np.isclose(area, cross_area, rtol=0.15):
            outlet_tag = surface[1]
        elif (np.sqrt((com[0] - cx) ** 2 + (com[1] - cy) ** 2) <= 2 * a
              and area <= hemi_area_cap):
            # A hemisphere cap of the particle. Classify by its z-centroid
            # relative to the split plane cz.
            if com[2] >= cz:
                particle_up_tags.append(surface[1])
            else:
                particle_lo_tags.append(surface[1])
        else:
            wall_tags.append(surface[1])

    particle_tags = particle_up_tags + particle_lo_tags

    def _debug_surfaces(surfaces, inlet_com, outlet_com, particle_com, cross_area):
        print("\n=== SURFACE IDENTIFICATION DEBUG (symmetric) ===")
        print(f"Expected inlet  CoM: {inlet_com},   area: {cross_area:.6f}")
        print(f"Expected outlet CoM: {outlet_com},  area: {cross_area:.6f}")
        print(f"Expected particle CoM: {particle_com}  (split plane cz={cz})")
        for surface in surfaces:
            com = gmsh.model.occ.getCenterOfMass(surface[0], surface[1])
            area = gmsh.model.occ.getMass(surface[0], surface[1])
            print(f"  Surface {surface}: CoM={np.array(com)}, area={area:.6f}")
        print("================================================\n")

    if debug or (inlet_tag is None or outlet_tag is None
                 or len(particle_up_tags) == 0 or len(particle_lo_tags) == 0):
        _debug_surfaces(surfaces, inlet_com_exp, outlet_com_exp, particle_com_exp, cross_area)
    if debug:
        print(f"  [build-sym] classified: inlet={inlet_tag} outlet={outlet_tag} "
              f"walls={wall_tags}")
        print(f"  [build-sym] particle_upper tags={particle_up_tags}  "
              f"particle_lower tags={particle_lo_tags}  "
              f"apply_periodic={apply_periodic}")
    assert inlet_tag is not None, "Could not identify inlet surface"
    assert outlet_tag is not None, "Could not identify outlet surface"
    assert len(particle_up_tags) >= 1, "Upper particle hemisphere not found"
    assert len(particle_lo_tags) >= 1, "Lower particle hemisphere not found"
    assert len(wall_tags) == 4, f"Expected 4 wall surfaces, found {len(wall_tags)}"

    gmsh.model.addPhysicalGroup(2, [inlet_tag], INLET)
    gmsh.model.setPhysicalName(2, INLET, "inlet")
    gmsh.model.addPhysicalGroup(2, [outlet_tag], OUTLET)
    gmsh.model.setPhysicalName(2, OUTLET, "outlet")
    gmsh.model.addPhysicalGroup(2, wall_tags, WALLS)
    gmsh.model.setPhysicalName(2, WALLS, "walls")
    # NOTE: put the cavity caps in EXACTLY ONE physical group (PARTICLE=4).
    # MSH 2.2 writes each element under only one physical entity, so adding the
    # same surfaces to PARTICLE *and* separate upper/lower groups left ds(4)
    # empty. setPeriodic below uses the GEOMETRIC surface tags directly, so it
    # does not need its own physical groups.
    gmsh.model.addPhysicalGroup(2, particle_tags, PARTICLE)
    gmsh.model.setPhysicalName(2, PARTICLE, "particle")

    # ── Reflection periodicity: lower hemisphere mesh = mirror of upper ────
    # Affine map (row-major 4x4) taking the MASTER (upper) surface points to
    # the SLAVE (lower) surface points:  (x, y, z) -> (x, y, 2*cz - z).
    refl = [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 2.0 * cz,
            0.0, 0.0, 0.0, 1.0]
    if apply_periodic:
        gmsh.model.mesh.setPeriodic(2, particle_lo_tags, particle_up_tags, refl)

    # --- Mesh refinement near particle (same as the base function) ---
    # Apply the distance field to BOTH hemispheres so the refinement that
    # feeds the periodic master is itself symmetric.
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

    # The setPeriodic reflection has already mirror-paired the particle surface
    # mesh in the written file. Strip the $Periodic record so Firedrake does not
    # glue the two hemisphere facets and empty ds(particle); see
    # _strip_periodic_section for the full rationale.
    if apply_periodic:
        _strip_periodic_section(tmp_path)

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


def check_particle_surface_symmetry(mesh3d, tags, cz, atol=1e-9):
    """Verify the particle-surface mesh is mirror-symmetric about z = cz.

    Extracts the nodes on the particle physical surface, mirrors them
    (z -> 2*cz - z), and reports the max nearest-neighbour distance between
    the mirrored set and the original. ~0 means the ``setPeriodic`` reflection
    produced an exactly mirror-paired surface mesh (Stufe-1 goal). Use this as
    a cheap, flow-free self-check before measuring the F_z noise floor.
    """
    from scipy.spatial import cKDTree
    V = VectorFunctionSpace(mesh3d, "CG", 1)
    coords = Function(V).interpolate(SpatialCoordinate(mesh3d))
    bc = DirichletBC(V, 0, tags["particle"])
    node_ids = bc.nodes
    pts = np.asarray(coords.dat.data_ro)[node_ids]
    if pts.shape[0] == 0:
        print("[SymCheck] no particle-surface nodes found.")
        return float("nan")
    mirror = pts.copy()
    mirror[:, 2] = 2.0 * cz - mirror[:, 2]
    d, _ = cKDTree(pts).query(mirror)
    max_d = float(np.max(d))
    print(f"[SymCheck] particle-surface nodes: {pts.shape[0]}  "
          f"max mirror-pair distance about z={cz:.4f}: {max_d:.3e}  "
          f"({'SYMMETRIC' if max_d < atol else 'NOT symmetric'})")
    return max_d


def plot_mesh_3d(mesh3d, filename="mesh_3d_wireframe.png", figsize=(9, 7), linewidth=0.15,
                 elev=25, azim=-60, show_axes=True, dpi=200,
                 origin=(0.0, 0.0, 0.0), basis=None,
                 tick_offset=(0.0, 0.0, 0.0),
                 xticks=None, yticks=None, zticks=None):
    # Render the tetrahedral mesh as a pure wireframe: every unique edge of
    # every tet is drawn as an opaque black line on a white background.  No
    # face polygons are drawn, so the interior structure (particle hemisphere,
    # refinement around it) is visible straight through the outer hull.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    W_p1 = FunctionSpace(mesh3d, "CG", 1)
    V_p1 = VectorFunctionSpace(mesh3d, "CG", 1)
    cell_node_map = W_p1.cell_node_map().values
    pts = np.array(Function(V_p1).interpolate(SpatialCoordinate(mesh3d)).dat.data_ro)

    # Optional rigid transform into a local frame: p_local = basis^T @ (p - origin).
    # Columns of `basis` are the local axis directions expressed in the global frame.
    o = np.asarray(origin, dtype=float)
    if basis is not None:
        pts = (pts - o) @ np.asarray(basis, dtype=float)
    elif np.any(o):
        pts = pts - o

    # 6 edges per tet, deduplicate across the whole mesh.
    local_edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    all_edges = np.sort(cell_node_map[:, local_edges].reshape(-1, 2), axis=1)
    unique_edges = np.unique(all_edges, axis=0)
    segments = pts[unique_edges]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    lc = Line3DCollection(segments, colors="black", linewidths=linewidth)
    ax.add_collection3d(lc)

    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    try:
        ax.set_box_aspect((mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]))
    except AttributeError:
        pass

    ax.view_init(elev=elev, azim=azim)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor("white")
        pane.set_edgecolor("white")
    ax.grid(False)

    if show_axes:
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")

        from matplotlib.ticker import FuncFormatter, MultipleLocator
        x0, y0, z0 = tick_offset

        # Explicit tick positions are given in shifted (displayed) coordinates,
        # so we add the offset back to place them on the real axis.
        if xticks is not None:
            ax.set_xticks([t + x0 for t in xticks])
        else:
            ax.xaxis.set_major_locator(MultipleLocator(1))
        if yticks is not None:
            ax.set_yticks([t + y0 for t in yticks])
        if zticks is not None:
            ax.set_zticks([t + z0 for t in zticks])

        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - x0:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - y0:g}"))
        ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v - z0:g}"))
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":

    R = 40.0
    a = 0.1
    H = 2.0
    W = 2.0
    L = 4 * max(H, W)
    x_off = 0.0
    z_off = 0.0

    # ── Stufe-1 symmetric-surface self-check (flow-free) ───────────────────
    # Build the symmetric variant at z_off = 0 (bifurcation case) AND off-axis,
    # then verify the particle SURFACE mesh is mirror-paired about z = cz. A
    # max mirror-pair distance ~0 means setPeriodic worked and the F_z surface
    # quadrature noise should collapse. Run me on mu44: `python3 build_3d_geometry_gmsh.py`.
    print("\n" + "#" * 60)
    print("#  STUFE-1 SYMMETRIC PARTICLE-SURFACE SELF-CHECK")
    print("#" * 60)
    # Isolate whether setPeriodic is what empties ds(particle): build z=0
    # WITHOUT then WITH the reflection periodicity. If ds(4) is fine without
    # and empty with, Firedrake is reading the $Periodic section as topology.
    for _periodic in (False, True):
        print(f"\n--- z_off=0.0, apply_periodic={_periodic} ---")
        try:
            _m, _t = make_curved_channel_section_with_spherical_hole_symmetric(
                R=R, H=H, W=W, L=L, a=a,
                particle_maxh=1 / 3 * a, global_maxh=1 / 3 * H,
                x_off=0.0, z_off=0.0, order=1,
                apply_periodic=_periodic, debug=True)
            _area = assemble(1.0 * ds(_t["particle"], domain=_m))
            print(f"  particle area = {_area:.5f} "
                  f"(expected {4.0*math.pi*a**2:.5f})")
            check_particle_surface_symmetry(_m, _t, cz=0.0)
        except Exception as _e:
            print(f"  FAILED: {_e}")
    print("#" * 60 + "\n")

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R=R, H=H, W=W, L=L, a=a,
                                                                    particle_maxh=1 / 3 * a, global_maxh=1 / 3 * H,
                                                                    x_off=x_off, z_off=z_off, order=2)

    V = FunctionSpace(mesh3d, "CG", 1)
    dummy = Function(V, name="Mesh_Visualization").assign(1.0)

    VTKFile("correct_channel.pvd").write(dummy)

    th_half = 0.5 * L / R
    # Local channel frame at the particle:
    #   x_loc = radial outward, y_loc = along channel, z_loc = vertical.
    channel_basis = np.array([
        [math.cos(th_half), -math.sin(th_half), 0.0],
        [math.sin(th_half), math.cos(th_half), 0.0],
        [0.0, 0.0, 1.0],
    ])
    particle_origin = (R * math.cos(th_half),
                       R * math.sin(th_half), 0.0)

    # plot_mesh_3d(mesh3d, filename="../images/MASTER'S THESIS/mesh_3d_wireframe.png", origin=particle_origin, basis=channel_basis,
    #     xticks=[-1, 0, 1], yticks=[-4, -3, -2, -1, 0, 1, 2, 3, 4], zticks=[-1, 0, 1])

    x, y, z = SpatialCoordinate(mesh3d)

    print("\n" + "=" * 40)
    print("=== STARTING SANITY CHECKS ===")
    print("=" * 40)

    coords = mesh3d.coordinates.dat.data_ro
    min_z, max_z = coords[:, 2].min(), coords[:, 2].max()
    height_z = max_z - min_z
    print(f"\n1. CHANNEL VERTICAL EXTENT (expected ~H={H}): {height_z:.4f}")

    print("\n2. SURFACE AREAS (Inlet / Outlet / Particle)")
    expected_cross_section = W * H
    inlet_area = assemble(1.0 * ds(tags["inlet"], domain=mesh3d))
    outlet_area = assemble(1.0 * ds(tags["outlet"], domain=mesh3d))
    print(f"   Inlet Area:  Expected {expected_cross_section:.4f}, Measured {inlet_area:.4f}")
    print(f"   Outlet Area: Expected {expected_cross_section:.4f}, Measured {outlet_area:.4f}")

    expected_particle_area = 4.0 * math.pi * a ** 2
    actual_particle_area = assemble(1.0 * ds(tags["particle"], domain=mesh3d))
    err_area = abs(expected_particle_area - actual_particle_area) / expected_particle_area
    print(
        f"   Particle Surface Area: Expected {expected_particle_area:.4f}, Measured {actual_particle_area:.4f} (Rel Err: {err_area:.2%})")

    print("\n3. TOTAL VOLUME")
    expected_vol = (W * H * L) - ((4.0 / 3.0) * math.pi * a ** 3)
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