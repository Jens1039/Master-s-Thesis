import os
os.environ["OMP_NUM_THREADS"] = "1"

import math
import tempfile
import numpy as np
import gmsh
from firedrake import *

from build_3d_geometry_gmsh import *
from nondimensionalization import *


class FullNavierStokesSolver:

    def __init__(self, R, H, W, L, a, Re, mesh3d, tags):
        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.L = L
        self.Re = Re

        self.mesh3d = mesh3d
        self.tags = tags

        self.x = SpatialCoordinate(self.mesh3d)
        self.x_p = as_vector(self.tags["particle_center"])

        self.e_x_prime = Constant([cos(self.L / self.R * 0.5), sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-sin(self.L / self.R * 0.5), cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        self.e_r_prime = as_vector(
            [self.x[0] / sqrt(self.x[0] ** 2 + self.x[1] ** 2), self.x[1] / sqrt(self.x[0] ** 2 + self.x[1] ** 2), 0])
        self.e_theta_prime = as_vector(
            [- self.x[1] / sqrt(self.x[0] ** 2 + self.x[1] ** 2), self.x[0] / sqrt(self.x[0] ** 2 + self.x[1] ** 2), 0])

        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.R_space = FunctionSpace(self.mesh3d, "R", 0)

        # Alle 7 Felder (inklusive G) bleiben drin für das lückenlose AD-Tape!
        self.W = self.V * self.Q * self.R_space * self.R_space * self.R_space * self.R_space * self.R_space


    def solve_flow(self):

        w = Function(self.W)
        u, p, Theta, Omega_p_x, Omega_p_y, Omega_p_z, G = split(w)
        v, q, test_Theta, test_Omega_p_x, test_Omega_p_y, test_Omega_p_z, g = TestFunctions(self.W)

        Omega_p = as_vector([Omega_p_x, Omega_p_y, Omega_p_z])
        test_Omega_p = as_vector([test_Omega_p_x, test_Omega_p_y, test_Omega_p_z])

        a_form = (
                (2.0 / self.Re) * inner(sym(grad(u)), sym(grad(v))) * dx(domain=self.mesh3d)
                - p * div(v) * dx(domain=self.mesh3d)
                + inner(dot(u, grad(u)), v) * dx(domain=self.mesh3d)
                + 2.0 * Theta * inner(cross(self.e_z_prime, u), v) * dx(domain=self.mesh3d)
                + (Theta ** 2) * inner(cross(self.e_z_prime, cross(self.e_z_prime, self.x)), v) * dx(domain=self.mesh3d)
        )

        continuity_equation = - q * div(u) * dx(domain=self.mesh3d)

        n = FacetNormal(self.mesh3d)
        h_cell = CellDiameter(self.mesh3d)
        gamma = Constant(50.0)

        def sigma(u, p): return (2.0 / self.Re) * sym(grad(u)) - p * Identity(3)

        nitsche_walls_bcs = (
                - inner(sigma(u, p) * n, v) * ds(self.tags["walls"])
                - inner(sigma(v, q) * n, u + Theta * cross(self.e_z_prime, self.x)) * ds(self.tags["walls"])
                + (gamma / h_cell) * inner(u + Theta * cross(self.e_z_prime, self.x), v) * ds(self.tags["walls"])
        )

        u_part = cross(Omega_p, self.x - self.x_p)
        nitsche_particle_bcs = (
                - inner(sigma(u, p) * n, v) * ds(self.tags["particle"])
                - inner(sigma(v, q) * n, u - u_part) * ds(self.tags["particle"])
                + (gamma / h_cell) * inner(u - u_part, v) * ds(self.tags["particle"])
        )

        force_and_torque_eqs = (
                test_Theta * inner(sigma(u, p) * n, self.e_y_prime) * ds(self.tags["particle"])
                + inner(cross(self.x - self.x_p, sigma(u, p) * n), test_Omega_p) * ds(self.tags["particle"])
        )

        body_force = - G * inner(v, self.e_theta_prime) * dx(domain=self.mesh3d)

        u_lab = u + Theta * cross(self.e_z_prime, self.x)
        constraint_eq = g * (inner(u_lab, self.e_theta_prime) - Constant(1.0)) * dx(domain=self.mesh3d)

        a_form_total = (a_form + continuity_equation + nitsche_walls_bcs + nitsche_particle_bcs + force_and_torque_eqs
                        + body_force + constraint_eq)

        solver_params = {
            "snes_type": "newtonls",
            "snes_monitor": None,
            "snes_linesearch_type": "l2",
            "mat_type": "matfree",
            "ksp_type": "fgmres",
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",

            "pc_fieldsplit_0_fields": "0,1",
            "pc_fieldsplit_1_fields": "2,3,4,5,6",

            "fieldsplit_0": {
                "ksp_type": "preonly",
                "pc_type": "python",
                "pc_python_type": "firedrake.AssembledPC",
                "assembled_pc_type": "lu",
                "assembled_pc_factor_mat_solver_type": "mumps"
            },

            "fieldsplit_1": {
                "ksp_type": "preonly",
                "pc_type": "none"
            },
        }
        '''
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "python",
            "fieldsplit_0_pc_python_type": "lu",
            "fieldsplit_0_assembled_pc_type": "lu",
            "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
            "fieldsplit_0_assembled_mat_mumps_icntl_14": 400,
            "fieldsplit_0_assembled_mat_mumps_icntl_24": 1,
            "fieldsplit_1_ksp_type": "gmres",
            "fieldsplit_1_ksp_max_it": 10,
            "fieldsplit_1_pc_type": "none",
        }
        '''

        solve(a_form_total == 0, w, solver_parameters=solver_params)
        self.w_sol = w
        return w


    def compute_particle_force(self):

        u_sol, p_sol, Theta_sol, Omega_p_x_sol, Omega_p_y_sol, Omega_p_z_sol, G_sol = self.w_sol.subfunctions
        n = FacetNormal(self.mesh3d)

        def sigma(u, p): return (2.0 / self.Re) * sym(grad(u)) - p * Identity(3)

        e_r_local = as_vector([cos(self.L / self.R * 0.5), sin(self.L / self.R * 0.5), 0])
        e_theta_local = as_vector([-sin(self.L / self.R * 0.5), cos(self.L / self.R * 0.5), 0])
        e_z_local = as_vector([0, 0, 1])

        F_p_x_prime_expr = -inner(sigma(u_sol, p_sol) * n, e_r_local) * ds(self.tags["particle"])
        F_p_y_prime_expr = -inner(sigma(u_sol, p_sol) * n, e_theta_local) * ds(self.tags["particle"])
        F_p_z_prime_expr = -inner(sigma(u_sol, p_sol) * n, e_z_local) * ds(self.tags["particle"])

        F_p_x_prime = assemble(F_p_x_prime_expr)
        F_p_y_prime = assemble(F_p_y_prime_expr)
        F_p_z_prime = assemble(F_p_z_prime_expr)

        m_p = (4.0 / 3.0) * pi * (self.a ** 3)
        r_p_mag = sqrt(self.tags["particle_center"][0] ** 2 + self.tags["particle_center"][1] ** 2)

        F_centri_mag_expr = m_p * (Theta_sol ** 2) * r_p_mag

        vol = assemble(Constant(1.0) * dx(domain=self.mesh3d))
        F_centri_mag = assemble(F_centri_mag_expr * dx(domain=self.mesh3d)) / vol

        F_net_r = F_p_x_prime + F_centri_mag
        F_net_theta = F_p_y_prime
        F_net_z = F_p_z_prime

        return F_net_r, F_net_theta, F_net_z


def make_curved_channel_section_with_spherical_hole_periodic(R, H, W, L, a, particle_maxh, global_maxh,
                                                    r_off=0.0, z_off=0.0, order=1, comm=COMM_SELF):
    theta = L / R

    cx = (R + r_off) * math.cos(theta * 0.5)
    cy = (R + r_off) * math.sin(theta * 0.5)
    cz = z_off

    WALLS, PARTICLE, FLUID_VOL = 3, 4, 11 # INLET und OUTLET entfernt

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMax", global_maxh)

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

    rev = gmsh.model.occ.revolve([(2, face)], 0, 0, 0, 0, 0, 1, theta)

    duct_tag = None
    for dim, tag in rev:
        if dim == 3:
            duct_tag = tag
            break
    assert duct_tag is not None, "Revolve did not produce a volume"

    particle_sphere = gmsh.model.occ.addSphere(cx, cy, cz, a)
    gmsh.model.occ.cut([(3, duct_tag)], [(3, particle_sphere)])
    gmsh.model.occ.synchronize()

    volumes = gmsh.model.getEntities(dim=3)
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

    assert inlet_tag is not None, "Could not identify inlet surface"
    assert outlet_tag is not None, "Could not identify outlet surface"

    # =====================================================================
    # WICHTIG: Keine Physical Groups für Inlet und Outlet definieren!
    # =====================================================================
    gmsh.model.addPhysicalGroup(2, wall_tags, WALLS)
    gmsh.model.setPhysicalName(2, WALLS, "walls")
    gmsh.model.addPhysicalGroup(2, particle_tags, PARTICLE)
    gmsh.model.setPhysicalName(2, PARTICLE, "particle")

    transform_matrix = [
        math.cos(theta), -math.sin(theta), 0.0, 0.0,
        math.sin(theta),  math.cos(theta), 0.0, 0.0,
        0.0,              0.0,             1.0, 0.0,
        0.0,              0.0,             0.0, 1.0
    ]

    gmsh.model.mesh.setPeriodic(2, [outlet_tag], [inlet_tag], transform_matrix)

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

    # Inlet und Outlet aus den Tags entfernt
    tags = {
        "walls": (WALLS,),
        "particle": PARTICLE,
        "theta": theta,
        "particle_center": (cx, cy, cz),
    }

    return mesh3d, tags

if __name__ == "__main__":

    R = 160.0
    H = 2.0
    W = 2.0
    L = 30 * max(H, W)
    a = 0.05
    Re = 1.0

    mesh3d, tags = make_curved_channel_section_with_spherical_hole_periodic(R, H, W, L, a, 0.2*a, 0.2 * min(H, W), r_off=3*a)
    NS = FullNavierStokesSolver(R, H, W, L, a, Re, mesh3d, tags)
    NS.solve_flow()
    F_net_r, F_net_theta, F_net_z = NS.compute_particle_force()

    print("NS:", F_net_r, F_net_theta, F_net_z)

    from background_flow import *
    from perturbed_flow import *

    bg = background_flow(R, H, W, Re)
    G_val, U_m_hat, u_bar_2d_hat, p_bar_2d_hat = bg.solve_2D_background_flow()

    from nondimensionalization import *

    R, H, W, a, G, _, _, u_bar, p_bar, Re_p = second_nondimensionalisation(R, H, W, a, H / 2, 0.008366733466944444,
                                                                           G_val, Re, u_bar_2d_hat, p_bar_2d_hat,
                                                                           U_m_hat)

    L = 30 * max(H, W)

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, H, W, L, a, 0.2 * a, 0.2 * min(H, W), r_off=3*a)

    u_bar_3d, p_bar_3d = build_3d_background_flow(R, H, W, G, mesh3d, tags, u_bar, p_bar)
    pf = perturbed_flow(R, H, W, L, a, Re_p, mesh3d, tags, u_bar_3d, p_bar_3d)
    F_p_x, F_p_z = pf.F_p()

    print("6 Stokes", F_p_x, F_p_z)

    print("Stokes ratio", F_p_x / F_p_z)
    print("NS ratio", F_net_r / F_net_z)

# r_off = z_off = 0.0


# r_off = z_off = 0.0

