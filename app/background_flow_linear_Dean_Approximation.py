from firedrake import *
import matplotlib.pyplot as plt
from main import Re_flow, R, H, De_opt

# -----------------------------
# Dimensionlose Parameter
# -----------------------------
Re = Re_flow
Rc_ratio = R/H
De = 5
C = Constant(1.0)

# -----------------------------
# Diskretisierung
# -----------------------------
nx, ny = 64, 64
mesh = UnitSquareMesh(nx, ny)

V = FunctionSpace(mesh, "CG", 2)
W = FunctionSpace(mesh, "CG", 2)
Z = V * W  # mixed space for (phi, psi)

u = Function(V, name="u*")
v = TestFunction(V)

# -----------------------------
# Hauptströmung: ∇² u = -C
# -----------------------------
F_u = inner(grad(u), grad(v)) * dx - C * v * dx
bcs_u = DirichletBC(V, Constant(0.0), "on_boundary")

solve(F_u == 0, u, bcs=bcs_u, solver_parameters={'ksp_type': 'cg'})

# -----------------------------
# Dean-Strömung (ψ, φ)
# -----------------------------
phi, psi = TrialFunctions(Z)
p, q = TestFunctions(Z)

F_dean = (inner(grad(psi), grad(p)) * dx - phi * p * dx
          + inner(grad(phi), grad(q)) * dx
          + Constant(De**2) * dot(as_vector((0, 1)), grad(u)) * q * dx)

a, L = lhs(F_dean), rhs(F_dean)
z = Function(Z)
bcs = [DirichletBC(Z.sub(0), Constant(0.0), "on_boundary"),
       DirichletBC(Z.sub(1), Constant(0.0), "on_boundary")]

solve(a == L, z, bcs=bcs, solver_parameters={'ksp_type': 'cg'})
phi_sol, psi_sol = z.subfunctions



# ---------------
Vv = VectorFunctionSpace(mesh, "CG", 2)
vfield = Function(Vv, name="Dean_vortices")

# Sekundärströmung: u_y = dψ/dz, u_z = -dψ/dy
vfield.project(as_vector((psi_sol.dx(1), -psi_sol.dx(0))))


# -----------------------------
# Visualisierung
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

# Gitter zum Abtasten (z. B. 40×40 Punkte)
n = 80
Y = np.linspace(0, 1, n)
Z = np.linspace(0, 1, n)
YY, ZZ = np.meshgrid(Y, Z)

UY = np.zeros_like(YY)
UZ = np.zeros_like(ZZ)

# Werte an den Rasterpunkten aus dem FEM-Feld auslesen
for i in range(n):
    for j in range(n):
        point = np.array([Y[i], Z[j]])
        val = vfield.at(point)
        UY[j, i] = val[0]
        UZ[j, i] = val[1]

# Normierung (optional)
speed = np.sqrt(UY**2 + UZ**2)

# -----------------------------
# Streamplot
# -----------------------------
plt.figure(figsize=(6,6))
plt.streamplot(Y, Z, UY, UZ, color=speed, cmap='plasma', density=1.4)
plt.title("Dean vortices — streamlines of secondary flow")
plt.xlabel("y*")
plt.ylabel("z*")
plt.axis("equal")
plt.show()


