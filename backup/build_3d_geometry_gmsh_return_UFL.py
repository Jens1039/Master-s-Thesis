import os
os.environ["OMP_NUM_THREADS"] = "1"

import os
import math
import tempfile
import numpy as np
import gmsh
from firedrake import *
from ufl import atan


def make_2d_base_mesh(R, H, W, particle_maxh, global_maxh, r_off=0.2, z_off=0.2, comm=COMM_SELF) -> Mesh:
    """
    Erzeugt das 2D-Querschnittsnetz mit lokaler Verfeinerung am Partikelort.
    x-Achse entspricht der radialen Richtung, y-Achse der vertikalen Richtung (z_off).
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", global_maxh)

    # Partikelzentrum im Querschnitt
    px = R + r_off
    py = z_off

    # Eckpunkte des Querschnitts
    p1 = gmsh.model.occ.addPoint(R - W / 2, -H / 2, 0)
    p2 = gmsh.model.occ.addPoint(R + W / 2, -H / 2, 0)
    p3 = gmsh.model.occ.addPoint(R + W / 2, +H / 2, 0)
    p4 = gmsh.model.occ.addPoint(R - W / 2, +H / 2, 0)

    # Der Dummy-Punkt für das Partikel (wird nicht in die Geometrie geschnitten,
    # dient nur als Anker für das Mesh-Refinement)
    p_particle = gmsh.model.occ.addPoint(px, py, 0)

    l1 = gmsh.model.occ.addLine(p1, p2)
    l2 = gmsh.model.occ.addLine(p2, p3)
    l3 = gmsh.model.occ.addLine(p3, p4)
    l4 = gmsh.model.occ.addLine(p4, p1)

    cl = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
    face = gmsh.model.occ.addPlaneSurface([cl])

    # Partikel-Punkt in die Fläche einbetten, damit Gmsh dort mesht
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, [p_particle], 2, face)

    # Physikalische Gruppen für die Wände definieren
    # Diese IDs werden beim Extrudieren in Firedrake als "Seitenwände" übernommen
    WALL_ID = 3
    gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], WALL_ID)
    gmsh.model.setPhysicalName(1, WALL_ID, "walls")

    FLUID_2D = 11
    gmsh.model.addPhysicalGroup(2, [face], FLUID_2D)
    gmsh.model.setPhysicalName(2, FLUID_2D, "fluid_2d")

    # --- Refinement Field um das Partikel ---
    dist_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(dist_field, "NodesList", [p_particle])

    thresh = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(thresh, "IField", dist_field)
    gmsh.model.mesh.field.setNumber(thresh, "LcMin", particle_maxh)
    gmsh.model.mesh.field.setNumber(thresh, "LcMax", global_maxh)
    gmsh.model.mesh.field.setNumber(thresh, "DistMin", min(W, H) * 0.2)
    gmsh.model.mesh.field.setNumber(thresh, "DistMax", min(W, H) * 0.8)

    gmsh.model.mesh.field.setAsBackgroundMesh(thresh)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    fd, tmp_path = tempfile.mkstemp(suffix=".msh")
    os.close(fd)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(tmp_path)
    gmsh.finalize()

    mesh2d = Mesh(tmp_path, comm=comm)
    os.unlink(tmp_path)

    return mesh2d


def extrude_and_curve_mesh(mesh2d, R, L, layers=50, cluster_factor=3.0):

    mesh3d = ExtrudedMesh(mesh2d, layers, layer_height=1.0 / layers)

    V_coord = mesh3d.coordinates.function_space()
    x_b, z_b, zeta = SpatialCoordinate(mesh3d)

    theta_max = L / R

    theta = theta_max * (0.5 + atan(cluster_factor * (zeta - 0.5)) / (2.0 * atan(cluster_factor / 2.0)))

    x_new = x_b * cos(theta)
    y_new = x_b * sin(theta)
    z_new = z_b

    mesh3d.coordinates.interpolate(as_vector([x_new, y_new, z_new]))

    tags = {
        "walls": 3,
        "inlet": "bottom",
        "outlet": "top",
        "theta_max": theta_max
    }

    return mesh3d, tags


if __name__ == "__main__":

    R_hat = 160.0
    H_hat = 2.0
    TEST_W = 2.0
    a = 0.05


    TEST_L = 10.0
    TEST_r_off = 0.0
    TEST_z_off = 0.0

    mesh2d = make_2d_base_mesh(TEST_R, TEST_H, TEST_W, 0.2 * TEST_a, 0.2 * TEST_H)

    mesh3d, tags = extrude_and_curve_mesh(mesh2d, TEST_R, TEST_L)

    V = FunctionSpace(mesh3d, "CG", 1)
    dummy = Function(V, name="Mesh_Visualization").assign(1.0)

    VTKFile("channel.pvd").write(dummy)