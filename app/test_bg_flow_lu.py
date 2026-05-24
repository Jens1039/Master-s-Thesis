"""Smoke test: solve the 2D BG-flow with a direct LU solver and report
DOF count, convergence, wall-clock time, peak memory. Does NOT touch the
production solver. Run from app/:  python3 test_bg_flow_lu.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
import resource
from firedrake import *

# Same physical parameters as the current shape-opt run (log_current.txt:7-13)
R_hat = 500.0
H_hat = 2.0
W_hat = 2.0
Re_float = 1.0

mesh2d = RectangleMesh(128, 128, W_hat, H_hat,
                       quadrilateral=False, diagonal="crossed")

V       = VectorFunctionSpace(mesh2d, "CG", 3, dim=3)
Q       = FunctionSpace(mesh2d, "CG", 2)
G_space = FunctionSpace(mesh2d, "R", 0)
W_mixed = V * Q * G_space

dofs_V = V.dof_dset.layout_vec.getSize()
dofs_Q = Q.dof_dset.layout_vec.getSize()
dofs_W = W_mixed.dof_dset.layout_vec.getSize()
print(f"[Info] V (CG3 vec3): {dofs_V} DOFs")
print(f"[Info] Q (CG2):      {dofs_Q} DOFs")
print(f"[Info] W_mixed:      {dofs_W} DOFs (≈ what LU has to factorise)")

w  = Function(W_mixed)
Re = Function(G_space).assign(Re_float)
u, p, G = split(w)
v, q, g = TestFunctions(W_mixed)

u_r, u_theta, u_z = u[0], u[2], u[1]
v_r, v_theta, v_z = v[0], v[2], v[1]

x = SpatialCoordinate(mesh2d)
r = x[0] - 0.5 * W_hat

def del_r(f): return Dx(f, 0)
def del_z(f): return Dx(f, 1)

F_cont = (q * (del_r(u_r) + del_z(u_z) + u_r / (R_hat + r))
          * (R_hat + r) * dx)
F_r    = ((u_r * del_r(u_r) + u_z * del_z(u_r)
           - (u_theta**2) / (R_hat + r)) * v_r
          + del_r(p) * v_r
          + (Constant(1.0) / Re) * dot(grad(u_r), grad(v_r))
          + (1 / Re) * (u_r / (R_hat + r)**2) * v_r
          ) * (R_hat + r) * dx
F_theta = ((u_r * del_r(u_theta) + u_z * del_z(u_theta)
            + (u_r * u_theta) / (R_hat + r)) * v_theta
           - ((G * R_hat) / (R_hat + r)) * v_theta
           + 1 / Re * dot(grad(u_theta), grad(v_theta))
           + 1 / Re * (u_theta / (R_hat + r)**2) * v_theta
           ) * (R_hat + r) * dx
F_z     = ((u_r * del_r(u_z) + u_z * del_z(u_z)) * v_z
           + del_z(p) * v_z
           + 1 / Re * dot(grad(u_z), grad(v_z))
           ) * (R_hat + r) * dx
F_G     = (u_theta - 1.0) * g * dx

F = F_r + F_theta + F_z + F_cont + F_G

no_slip = DirichletBC(W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), "on_boundary")
nullspace = MixedVectorSpaceBasis(
    W_mixed,
    [W_mixed.sub(0),
     VectorSpaceBasis(constant=True, comm=W_mixed.comm),
     W_mixed.sub(2)])

problem = NonlinearVariationalProblem(F, w, bcs=[no_slip])

# === Direct LU ===
solver = NonlinearVariationalSolver(
    problem, nullspace=nullspace,
    solver_parameters={
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_monitor": None,
        "snes_converged_reason": None,
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
    },
)

print("\n[Solve] starting direct-LU Newton solve …")
t0 = time.time()
try:
    solver.solve()
    ok = True
    err = None
except Exception as e:
    ok = False
    err = e
elapsed = time.time() - t0

rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# macOS reports ru_maxrss in bytes; Linux in kB. Show both.
print(f"\n[Result] converged: {ok}")
if not ok:
    print(f"[Result] error: {type(err).__name__}: {err}")
print(f"[Result] wall-clock: {elapsed:.2f} s")
print(f"[Result] peak RSS:   {rss_kb/1024:.1f} MB  (assuming kB units)")
print(f"[Result] peak RSS:   {rss_kb/1024**2:.1f} MB  (assuming B units, macOS)")

if ok:
    u_sol, p_sol, G_sol = w.subfunctions
    u_max = float(max(abs(u_sol.dat.data_ro.ravel())))
    print(f"[Result] |u|_max = {u_max:.4e}")
    print(f"[Result] G       = {float(G_sol.dat.data_ro[0]):.4e}")
