import os
os.environ["OMP_NUM_THREADS"] = "1"
from firedrake import *
import numpy as np

def Stokes_solver_3d(mesh3d, bcs_on_particle, walls_id, particle_id, mu):
    """
    Solves
        -grad(q) + mu * grad(div(v)) = 0
         div(v) = 0
         v = 0      on wall (no-slip)
         bcs_on_particle.
    """
    V = VectorFunctionSpace(mesh3d, "CG", 2)
    Q = FunctionSpace(mesh3d, "CG", 1)
    W = V * Q

    v, q = TrialFunctions(W)
    f, g = TestFunctions(W)

    # weak form of -grad(q) + mu * grad(div(v))
    a = mu * inner(grad(v), grad(f)) * dx - q * div(f) * dx + g * div(v) * dx

    # weak form of div(v) = 0
    b = g * div(v) * dx

    w = Function(W, name="stokes_solution")

    wall_ids = walls_id if isinstance(walls_id, (list, tuple, set)) else [walls_id]
    bcs = [DirichletBC(W.sub(0), Constant((0.0, 0.0, 0.0)), wid) for wid in wall_ids]
    bcs.append(DirichletBC(W.sub(0), bcs_on_particle, particle_id))

    one = Function(Q); one.assign(1.0)

    nullspace = MixedVectorSpaceBasis(W, [W.sub(0), VectorSpaceBasis([one])])

    problem = LinearVariationalProblem(a, 0, w, bcs=bcs)
    solver = LinearVariationalSolver(
        problem,
        nullspace=nullspace,
        transpose_nullspace=nullspace,
        solver_parameters={
            "mat_type": "aij",
            "ksp_type": "gmres",
            "ksp_rtol": 1.0e-8,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_schur_fact_type": "full",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_0_pc_type": "hypre",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "jacobi",
        }
    )

    solver.solve()

    v_0, q_0 = w.subfunctions

    return v_0, q_0

def F_minus_1_a(v_0_a, q_0_a, mesh3d, particle_id):

    n = FacetNormal(mesh3d)
    sigma = -q_0_a*Identity(3) + grad(v_0_a) + grad(v_0_a).T
    traction = -dot(n, sigma)                # Vektor
    dS_particle = ds(particle_id)

    comps = [assemble(traction[i] * dS_particle) for i in range(3)]
    return np.array([float(c) for c in comps])

def T_minus_1_a(v_0_a, q_0_a, mesh3d, particle_id, x_p):

    n = FacetNormal(mesh3d)
    x = SpatialCoordinate(mesh3d)

    if not isinstance(x_p, ufl.core.expr.Expr):
        x_p = Constant(tuple(x_p))
    rvec = as_vector((x[0]-x_p[0], x[1]-x_p[1], x[2]-x_p[2]))

    sigma = -q_0_a*Identity(3) + grad(v_0_a) + grad(v_0_a).T

    traction = -dot(n, sigma)

    moment_density = cross(rvec, traction)   # Vektor

    dS_part = ds(particle_id)

    return np.array([float(assemble(moment_density[i] * dS_part)) for i in range(3)])

def numerically_stable_solver_2x2(A, b, tol):
    """
    Löse A x = b robust, auch wenn A fast singulär ist.
    """
    detA = np.linalg.det(A)
    if abs(detA) > tol:
        # normale exakte Lösung
        x = np.linalg.solve(A, b)
    else:
        # Matrix ist (nahezu) singulär → numerisch stabilere Lösung per Pseudoinverse
        x = np.linalg.pinv(A) @ b
    return x
