import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np
from math import atan2, hypot, cos, sin
import matplotlib.pyplot as plt


class background_flow:

    def __init__(self, R, H, W, L_c, Re, comm=None):
        self.R = R
        self.H = H
        self.W = W
        self.L_c = L_c
        self.Re = Re

        from firedrake import COMM_WORLD
        actual_comm = comm if comm is not None else COMM_WORLD

        self.mesh2d = RectangleMesh(120, 120, self.W, self.H, quadrilateral=False, comm=actual_comm)


    def solve_2D_background_flow(self):

        # Velocity space (3 components: u_r, u_z, u_theta), pressure space, and scalar space for G
        V = VectorFunctionSpace(self.mesh2d, "CG", 2, dim=3)
        Q = FunctionSpace(self.mesh2d, "CG", 1)
        G_space = FunctionSpace(self.mesh2d, "R", 0)

        # Just the Cartesian product of V, Q and R
        W_mixed = V * Q * G_space

        # extracts a UFL expression for the coordinates of the mesh (non-dimensional!)
        x_mesh, z_mesh = SpatialCoordinate(self.mesh2d)

        # Initialise r and z so that (r,z) = (0,0) is at the center of the cross-section
        r = x_mesh - 0.5 * self.W
        z = z_mesh - 0.5 * self.H

        w = Function(W_mixed)
        u, p, G = split(w)
        v, q, g = TestFunctions(W_mixed)

        # Transformation of the velocity components to our local coordinate system
        # u = (u_x, u_z, u_theta) in the mesh coordinates, we interpret them as:
        u_r = u[0]
        u_z = u[1]
        u_theta = u[2]

        v_r = v[0]
        v_z = v[1]
        v_theta = v[2]

        # Shortcuts for readability
        def del_r(f):  return Dx(f, 0)
        def del_z(f):  return Dx(f, 1)

        # weak form of the radial-component
        F_r = ((u_r * del_r(u_r) + u_z * del_z(u_r) - (u_theta**2) / (self.R + r)) * v_r
                + del_r(p) * v_r
                + 1/self.Re * dot(grad(u_r), grad(v_r))
                + 1/self.Re * ((1.0/(self.R + r)) * del_r(u_r) - (u_r / (self.R + r)**2)) * v_r) * (self.R + r) * dx

        # weak form of the azimuthal component
        F_theta = ((u_r * del_r(u_theta) + u_z * del_z(u_theta) + (u_theta * u_r) / (self.R +r)) * v_theta
                - ((G*self.R) / (self.R + r)) * v_theta
                + 1/self.Re * dot(grad(u_theta), grad(v_theta))
                + 1/self.Re * ((1.0/(self.R + r)) * del_r(u_theta) - (u_theta / (self.R + r)**2)) * v_theta) * (self.R + r) * dx

        # weak form of the z-component
        F_z = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
            + del_z(p) * v_z
            + 1/self.Re * dot(grad(u_z), grad(v_z))
            + 1/self.Re * ( (1.0/(self.R + r)) * del_r(u_z) ) * v_z) * (self.R + r) * dx

        # weak form of the continuity equation
        F_cont = q * (del_r(u_r) + del_z(u_z) + u_r / (self.R + r)) * (self.R + r) * dx

        # weak form of the flow rate constraint
        F_G = (u_theta  - 1.0) * g * dx

        # Total residual
        F = F_r + F_theta + F_z + F_cont + F_G

        # When you define the problem the boundary conditions must apply to components of W_mixed, not to the stand-alone V,
        # thats why we're using W_mixed.sub(0) = "First component of W_mixed"
        no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")

        # In Navier-Stokes, the pressure just appears in a differentiated form and is therefore unique except for a constant.
        # A constant pressure is (because of a lack of boundary conditions for p) part of the kernel of the linearised operator.
        # If we give the solver the information where nullspaces are, it can ignore this direction while inverting the matrix
        # in our Newton method.
        nullspace = MixedVectorSpaceBasis(
            W_mixed,
            [
                W_mixed.sub(0),
                VectorSpaceBasis(constant=True, comm=W_mixed.comm),
                W_mixed.sub(2),
            ],
        )

        problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

        # Firedrake calculates the Gateaux-Derivative and transfers it into the Jacobian matrix on the given FEM mesh.
        # The resulting LSE is solved by using LU - decomposition.
        solver = NonlinearVariationalSolver(
            problem,
            nullspace=nullspace,
            solver_parameters={
                # SNES = Scalable Nonlinear Equation Solver  here: Newton with Line Search as step range criteria
                "snes_type": "newtonls",
                # bt: test the descent criteria ||F(w_k + a_k*d_k)|| < (1 - ca_k)||F(w_k)|| with decreasing a_k until it is valid
                # l2: solves min ||F(x_k + alpha*d_k||_(L^2) over alpha
                "snes_linesearch_type": "l2",
                # Tests for ||F(w_k)||/||F(w_0)|| and stops algorithm once ratio has fallen below
                "snes_rtol": 1e-8,
                # Tests for ||F(w_k)|| and stops algorithm once absolute value has fallen below
                "snes_atol": 1e-10,
                # Maximal Newton steps
                "snes_max_it": 50,
                # Tell firedrake not to assemble the global matrix to avoid the "Monolithic matrix assembly ..." error
                "mat_type": "matfree",
                # Use FieldSplit to separate (u, p) from (G)
                "ksp_type": "fgmres",
                "pc_type": "fieldsplit",
                "pc_fieldsplit_type": "schur",
                "pc_fieldsplit_schur_fact_type": "full",
                # Manually group fields:
                #    Split '0' = Fields 0, 1 (Velocity, Pressure)
                #    Split '1' = Field 2 (G)
                "pc_fieldsplit_0_fields": "0,1",
                "pc_fieldsplit_1_fields": "2",
                # Solver for Split '0' (Navier-Stokes block)
                #    Since the global operator is matfree, we must use "AssembledPC"
                #    to force assembly of this block so we can use MUMPS LU.
                "fieldsplit_0": {
                    "ksp_type": "preonly",
                    "pc_type": "python",
                    "pc_python_type": "firedrake.AssembledPC",
                    "assembled_pc_type": "lu",
                    "assembled_pc_factor_mat_solver_type": "mumps"
                },
                # Solver for Split '1' (The scalar G)
                #    This is just a 1x1 block, so 'none' or 'jacobi' is fine.
                "fieldsplit_1": {
                    "ksp_type": "preonly",
                    "pc_type": "none"
                },
            },
        )

        solver.solve()
        u_bar, p_bar, G_hat = w.subfunctions
        self.u_bar = u_bar
        self.p_bar = p_bar

        # necessary for rescale Re in the same way as in the paper
        u_data = self.u_bar.dat.data_ro
        U_max = np.max(np.abs(u_data[:, 2]))
        self.U_m = U_max

        return u_bar, p_bar, U_max


    def plot_2D_background_flow(self):
        coords = self.mesh2d.coordinates.dat.data_ro
        xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
        zmin, zmax = float(coords[:, 1].min()), float(coords[:, 1].max())

        nxp, nzp = 160, 160
        xi = np.linspace(xmin, xmax, nxp)
        zi = np.linspace(zmin, zmax, nzp)
        Xi, Zi = np.meshgrid(xi, zi)

        pts = np.column_stack([Xi.ravel(), Zi.ravel()])
        U_at_list = self.u_bar.at(pts)

        try:
            U_at = np.asarray(U_at_list, dtype=float)
            if U_at.ndim != 2 or U_at.shape[1] != 3:
                raise ValueError
        except Exception:
            U_at = np.vstack([np.asarray(v, dtype=float).ravel() for v in U_at_list])

        Ur = U_at[:, 0].reshape(nzp, nxp)
        Uz = U_at[:, 1].reshape(nzp, nxp)
        Uth = U_at[:, 2].reshape(nzp, nxp)

        Speed = np.sqrt(Ur ** 2 + Uz ** 2)
        Speed[~np.isfinite(Speed)] = 0.0
        Ur[~np.isfinite(Ur)] = 0.0
        Uz[~np.isfinite(Uz)] = 0.0

        eps = 1e-14
        lw = 0.8 + 2.0 * (Speed / (Speed.max() + eps))

        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("r")
        ax.set_ylabel("z")

        cf = ax.contourf(Xi, Zi, Uth, levels=40, cmap="coolwarm")
        cbar1 = fig.colorbar(cf, ax=ax, shrink=0.9, pad=0.02)
        cbar1.set_label(r"$u_\theta$")

        strm = ax.streamplot(
            xi, zi, Ur, Uz,
            density=1.4,
            color=Speed,
            linewidth=lw,
            cmap="viridis",
            arrowsize=1.2,
            minlength=0.1
        )
        cbar2 = fig.colorbar(strm.lines, ax=ax, shrink=0.9, pad=0.02)
        cbar2.set_label(r"$|u_{\mathrm{sec}}| = \sqrt{u_r^2 + u_z^2}$")

        plt.tight_layout()
        plt.show()

'''
def build_3d_background_flow(R, H, W, mesh3d, u_bar_2d, p_bar_2d):
    # ufl_element() gets the information for the family (e.g. "Lagrange"), the cell-type (e.g. "quadrilateral") and
    # the polynomial degree out of a function space like VectorFunctionSpace(mesh, "CG", 2)
    u_el = u_bar_2d.function_space().ufl_element()
    p_el = p_bar_2d.function_space().ufl_element()

    # We now create 3d function spaces with the same family and degree as our 2d function spaces
    V = VectorFunctionSpace(mesh3d, u_el.family(), u_el.degree())
    u_bar_3d = Function(V)
    Q = FunctionSpace(mesh3d, p_el.family(), p_el.degree())
    p_bar_3d = Function(Q)

    x = SpatialCoordinate(mesh3d)

    # creates an empty function in the V Vector Space
    coord_vec = Function(V)
    # fills coord_vec with the true mesh coordinates meaning coord_vec = id_x
    coord_vec.interpolate(as_vector((x[0], x[1], x[2])))
    # .dat grants access to the container, which holds the PETCs-vectors containing the actual data
    # .data_ro grants acces to theese data with ro = read_only
    coords_u = coord_vec.dat.data_ro

    eps = 1e-10
    u_points = []
    u_angles = []
    # iterates over all N grid points of our mesh3d and assigns the associated 2d-coordinate
    for j in range(coords_u.shape[0]):
        # Reads out the spatial coordinates of the current grid point
        X, Y, Z = coords_u[j, :]
        # We transform from global cartesian coordinates to global cylindric coordinates
        r_glob = hypot(X, Y)
        theta = atan2(Y, X)
        z = Z
        # numerical correction to ensure the functionality of PointEvaluator
        r_2d = min(max(W / 2 + (r_glob - R), 0.0 + eps), W - eps)
        z_2d = min(max(H / 2 + z, 0.0 + eps), H - eps)

        u_points.append((r_2d, z_2d))
        u_angles.append(theta)

    ur_uz_utheta_all = PointEvaluator(mesh=u_bar_2d.function_space().mesh(), points=u_points,
                                      tolerance=1e-10).evaluate(u_bar_2d)

    for j, vals in enumerate(ur_uz_utheta_all):
        u_r_val, u_z_val, u_theta_val = [float(x) for x in np.asarray(vals).reshape(-1)[:3]]
        phi = u_angles[j]
        Ux = u_r_val * cos(phi) - u_theta_val * sin(phi)
        Uy = u_r_val * sin(phi) + u_theta_val * cos(phi)
        Uz = u_z_val

        u_bar_3d.dat.data[j, 0] = Ux
        u_bar_3d.dat.data[j, 1] = Uy
        u_bar_3d.dat.data[j, 2] = Uz

    VQ = VectorFunctionSpace(mesh3d, p_el.family(), p_el.degree(), dim=3)
    coord_sca = Function(VQ)
    coord_sca.interpolate(as_vector((x[0], x[1], x[2])))
    coords_p = coord_sca.dat.data_ro

    p_points = []
    for j in range(coords_p.shape[0]):
        X, Y, Z = coords_p[j, :]
        R_glob = hypot(X, Y)
        x2d = min(max(W / 2.0 + (R_glob - R), 0.0 + eps), W - eps)
        z2d = min(max(H / 2.0 + Z, 0.0 + eps), H - eps)
        p_points.append((x2d, z2d))

    p_vals_raw = PointEvaluator(mesh=p_bar_2d.function_space().mesh(), points=p_points, tolerance=1e-10).evaluate(
        p_bar_2d)

    p_data = p_bar_3d.dat.data
    for j, v in enumerate(p_vals_raw):
        p_data[j] = float(np.asarray(v).reshape(-1)[0])

    return u_bar_3d, p_bar_3d
'''

def build_3d_background_flow(R_s2, H_s2, W_s2, mesh3d, u_bar_2d, p_bar_2d):
    """
    Extrudiert die 2D Background-Flow Lösung auf das 3D Mesh.
    Transformiert dabei die zylindrischen Geschwindigkeitskomponenten (u_r, u_z, u_theta)
    in kartesische Komponenten (u_x, u_y, u_z).

    Args:
        R_s2, H_s2, W_s2: Geometrie-Parameter auf der Skala des 3D Meshes (Scale 2).
        mesh3d: Das 3D Mesh (Firedrake Mesh).
        u_bar_2d: Die 2D Geschwindigkeitslösung (auf dem 2D Mesh Scale 2).
                  Erwartete Komponenten: [0]=u_r, [1]=u_z, [2]=u_theta.
        p_bar_2d: Die 2D Drucklösung (auf dem 2D Mesh Scale 2).

    Returns:
        u_3d, p_3d: Firedrake Funktionen auf dem 3D Mesh.
    """

    # 1. Funktionsräume auf dem 3D Mesh erstellen
    # Geschwindigkeit ist CG2, Druck ist CG1 (Taylor-Hood)
    V_3d = VectorFunctionSpace(mesh3d, "CG", 2)
    Q_3d = FunctionSpace(mesh3d, "CG", 1)

    u_3d = Function(V_3d)
    p_3d = Function(Q_3d)

    # --- GESCHWINDIGKEIT (Vektoriell, CG2) ---

    # Wir benötigen die Koordinaten aller Freiheitsgrade (DoFs) von V_3d.
    # Dazu interpolieren wir die Mesh-Koordinaten auf einen Vector-CG2 Raum.
    V_coords = VectorFunctionSpace(mesh3d, "CG", 2)
    coords_func_u = Function(V_coords).interpolate(SpatialCoordinate(mesh3d))

    # Zugriff auf die Koordinaten-Daten (N_nodes x 3)
    xyz_nodes_u = coords_func_u.dat.data_ro

    # Berechnung der lokalen Koordinaten für jeden Knoten im 3D Mesh
    # r_3d: Abstand vom Krümmungsmittelpunkt (0,0,0)
    # theta_3d: Winkel in der x-y Ebene
    r_3d = np.sqrt(xyz_nodes_u[:, 0] ** 2 + xyz_nodes_u[:, 1] ** 2)
    theta_3d = np.arctan2(xyz_nodes_u[:, 1], xyz_nodes_u[:, 0])
    z_3d = xyz_nodes_u[:, 2]

    # Mapping auf das 2D Mesh Koordinatensystem:
    # Das 2D Mesh in background_flow wird als [0, W] x [0, H] erzeugt.
    # In der 2D-Rechnung wurde definiert: r_sim = x_mesh - 0.5*W, z_sim = y_mesh - 0.5*H.
    # Wir haben r_local = r_3d - R.
    # Also: x_query = r_local + 0.5*W = (r_3d - R) + 0.5*W
    #       y_query = z_local + 0.5*H = z_3d + 0.5*H

    x_query_u = (r_3d - R_s2) + 0.5 * W_s2
    y_query_u = z_3d + 0.5 * H_s2

    # Um numerische Rundungsfehler am Rand abzufangen (damit .at() nicht fehlschlägt),
    # clippen wir die Koordinaten leicht auf die Domain des 2D Meshes.
    epsilon = 1e-12
    x_query_u = np.clip(x_query_u, 0.0, W_s2)
    y_query_u = np.clip(y_query_u, 0.0, H_s2)

    query_points_u = np.column_stack((x_query_u, y_query_u))

    # Auswertung der 2D-Lösung an den entsprechenden Punkten
    # u_vals Shape: (N_nodes, 3) -> [u_r, u_z, u_theta]
    u_vals = np.array(u_bar_2d.at(query_points_u))

    u_r = u_vals[:, 0]
    u_z = u_vals[:, 1]
    u_th = u_vals[:, 2]

    # Rücktransformation in kartesische 3D Vektoren (x, y, z)
    # Rotation um die z-Achse basierend auf theta_3d
    cos_th = np.cos(theta_3d)
    sin_th = np.sin(theta_3d)

    # u_x = u_r * cos(theta) - u_theta * sin(theta)
    # u_y = u_r * sin(theta) + u_theta * cos(theta)
    u_x_3d = u_r * cos_th - u_th * sin_th
    u_y_3d = u_r * sin_th + u_th * cos_th
    u_z_3d = u_z

    # Zuweisung in die 3D Funktion
    u_3d.dat.data[:] = np.column_stack((u_x_3d, u_y_3d, u_z_3d))

    # --- DRUCK (Skalar, CG1) ---

    # Gleiches Verfahren für den Druck, aber auf CG1 Knoten (weniger Punkte)
    Q_coords = VectorFunctionSpace(mesh3d, "CG", 1)
    coords_func_p = Function(Q_coords).interpolate(SpatialCoordinate(mesh3d))
    xyz_nodes_p = coords_func_p.dat.data_ro

    r_3d_p = np.sqrt(xyz_nodes_p[:, 0] ** 2 + xyz_nodes_p[:, 1] ** 2)
    z_3d_p = xyz_nodes_p[:, 2]

    x_query_p = (r_3d_p - R_s2) + 0.5 * W_s2
    y_query_p = z_3d_p + 0.5 * H_s2

    x_query_p = np.clip(x_query_p, 0.0, W_s2)
    y_query_p = np.clip(y_query_p, 0.0, H_s2)

    query_points_p = np.column_stack((x_query_p, y_query_p))

    # Auswertung des Drucks
    p_vals = np.array(p_bar_2d.at(query_points_p))

    # Zuweisung in die 3D Funktion
    p_3d.dat.data[:] = p_vals

    return u_3d, p_3d
