from firedrake import *
import numpy as np


W = 2.0
H = 2.0
R = 160
Re = 1



def ComputeBackgroundFlow(W, H, R, Re = 1.0, nW = 32, nH = 32, scaling = "mean"):

    assert scaling == "mean"

    mesh = RectangleMesh(
        nW, nH,
        W, H,
        quadrilateral=False,  # False -> Triangles
    )

    x = mesh.coordinates
    x.dat.data[:, 0] -= W / 2
    x.dat.data[:, 1] -= H / 2

    R_c = Constant(float(R))    # UFL-compatible expressions
    Re_c = Constant(float(Re))  # UFL-compatible expressions

    r, z = SpatialCoordinate(mesh)

    VS1 = FunctionSpace(mesh, "Lagrange", 2)
    VS2 = FunctionSpace(mesh, "Lagrange", 2)
    VS3 = FunctionSpace(mesh, "Lagrange", 2)
    PS = FunctionSpace(mesh, "Lagrange", 1)

    GS = FunctionSpace(mesh, "R", 0)
    WS = VS1 * VS2 * VS3 * PS * GS

    w = Function(WS)

    un, vn, wn, pn, Gn = split(w)
    tu, tv, tw, tp, tG = TestFunctions(WS)

    F_p = tp * (vn.dx(0) + wn.dx(1) + vn / (R_c + r)) * (R_c + r) * dx
    F_u = (Re_c * (vn * un.dx(0) + wn * un.dx(1) + (vn * un) / (R_c + r)) * tu
           - ((Gn * R_c) / (R_c + r)) * tu
           + dot(grad(un), grad(tu))
           + (un / (R_c + r) ** 2) * tu
           ) * (R_c + r) * dx
    F_v = (Re_c * (vn * vn.dx(0) + wn * vn.dx(1) - (un * un) / (R_c + r)) * tv
           + pn.dx(0) * tv
           + dot(grad(vn), grad(tv))
           + (vn / (R_c + r) ** 2) * tv
           ) * (R_c + r) * dx
    F_w = (Re_c * (vn * wn.dx(0) + wn * wn.dx(1)) * tw
           + pn.dx(1) * tw
           + dot(grad(wn), grad(tw))
           ) * (R_c + r) * dx
    # F = F_v + F_u + F_w + F_p
    F_G = (un - 1.0) * tG * dx
    # F = F_v + F_u + F_w + F_p + F_G
    zero = Constant(0.0)
    bcs = [
        DirichletBC(WS.sub(0), zero, "on_boundary"),  # u
        DirichletBC(WS.sub(1), zero, "on_boundary"),  # v
        DirichletBC(WS.sub(2), zero, "on_boundary"),  # w
    ]
    # Setup and solve the blocked non-linear problem
    W_n = [un, vn, wn, pn]
    Residual = [F_u, F_v, F_w, F_p]
    W_n.append(Gn)
    Residual.append(F_G)
    J = derivative(sum(Residual), w)
    petsc_options = {
                        "snes_type": "newtonls",
                        "snes_linesearch_type": "none",
                        "snes_atol": 1e-6,
                        "snes_rtol": 1e-6,
                        "snes_monitor": None,

                        "mat_type": "matfree",
                        "ksp_type": "fgmres",
                        "pc_type": "fieldsplit",
                        "pc_fieldsplit_type": "schur",
                        "pc_fieldsplit_schur_fact_type": "full",

                        "pc_fieldsplit_0_fields": "0,1,2,3",
                        "pc_fieldsplit_1_fields": "4",

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
    problem = NonlinearVariationalProblem(sum(Residual), w, bcs=bcs, J=J)
    solver = NonlinearVariationalSolver(problem, solver_parameters=petsc_options)
    solver.solve()
    # Extract velocity values at dof coordinates and re-structure as a grid
    uh, vh, wh, ph, Gh = w.subfunctions
    r, z = SpatialCoordinate(mesh)
    Rs_unsrt = Function(VS1).interpolate(r).dat.data_ro.copy()
    Zs_unsrt = Function(VS1).interpolate(z).dat.data_ro.copy()
    assert len(Rs_unsrt) == (2 * nW + 1) * (2 * nH + 1)
    sorting = np.argsort(Rs_unsrt + 4 * nW * (Zs_unsrt))
    Rs = Rs_unsrt[sorting].reshape((2 * nH + 1, 2 * nW + 1))
    Zs = Zs_unsrt[sorting].reshape((2 * nH + 1, 2 * nW + 1))
    Us = uh.dat.data_ro[sorting].reshape((2 * nH + 1, 2 * nW + 1))
    Vs = vh.dat.data_ro[sorting].reshape((2 * nH + 1, 2 * nW + 1))
    Ws = wh.dat.data_ro[sorting].reshape((2 * nH + 1, 2 * nW + 1))
    return [Rs, Zs], [Us, Vs, Ws]





'''
RZ, UVW = ComputeBackgroundFlow(W, H, R)
np.set_printoptions(
    threshold=np.inf,
    linewidth=200,
    suppress=True,
    precision=6
)

print(RZ)
print(UVW)
'''