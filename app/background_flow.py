from firedrake import *
from firedrake import triplot
import matplotlib.pyplot as plt
import math

R = 500*1e-6
H = 500*1e-6
mu = 1e-3       # dynamic viscosity [Pa·s]
rho = 1000.0    # density [kg/m^3]
Q_vol = 500e-9 / 60  # volumetric flow [m³/s]
U_mean = Q_vol / (H**2)

#---GEOMETRIE-------------------------------------------
theta_max = math.pi/2

nx, ny, nz = 16, 16, 32

mesh0 = BoxMesh(nx, ny, nz, 1, 1, 1)

x = SpatialCoordinate(mesh0)

# Transformation to the cylindric lab reference frame
r = sqrt(x[0]**2 + x[1]**2) - R
theta = atan2(x[1], x[0])
z = x[2]

xi, eta, zeta = x

r_local = H * xi
z_local = H * eta
theta = theta_max * zeta

X = as_vector([(R + r_local) * cos(theta),
               (R + r_local) * sin(theta),
               z_local])

mesh0.coordinates.interpolate(X)
# triplot(mesh0)
# plt.show()
# ------------------------------------------------------

# ---FUNCTION SPACE-------------------------------------
U = VectorFunctionSpace(mesh0, "CG", 2)
Q = FunctionSpace(mesh0, "CG", 1)
W = U * Q

w = Function(W)
(u, q) = split(w)
(f, g) = TestFunctions(W)
# ------------------------------------------------------

# ---BOUNDARY CONDITIONS-------------------------------
# For a BoxMesh, faces are labeled 1–6.
# Convention (usually for BoxMesh):
# 1,2 -> x=0,1   (streamwise direction)
# 3,4 -> y=0,1
# 5,6 -> z=0,1   (azimuthal direction before mapping)

# After mapping:
# - face 1 (x=0) = inner bend surface
# - face 2 (x=1) = outer bend surface
# - face 3 (y=0) = bottom wall
# - face 4 (y=1) = top wall
# - face 5 (z=0) = inlet
# - face 6 (z=1) = outlet

inlet_vel = Constant((U_mean, 0.0, 0.0))

bcs = [
    DirichletBC(W.sub(0), inlet_vel, 5),                # inlet velocity
    DirichletBC(W.sub(0), Constant((0, 0, 0)), (1, 2, 3, 4)),  # walls
    DirichletBC(W.sub(1), Constant(0.0), 6)             # outlet pressure
]
# ------------------------------------------------------

# ---VARIATIONAL FORM OF THE NAVIER STOKES EQN----------
F = (rho * inner(dot(u, nabla_grad(u)), f) * dx
     + 2*mu * inner(sym(grad(u)), sym(grad(f))) * dx
     - div(f) * q * dx
     + g * div(u) * dx)
# ------------------------------------------------------

# ---SOLVE----------------------------------------------
solve(F == 0, w, bcs=bcs)
# ------------------------------------------------------

# Option A (tuple unpack)
u_sol, p_sol = w.subfunctions

# Option B (index explicitly)
u_sol = w.subfunctions[0]
p_sol = w.subfunctions[1]
u_sol.rename("Velocity")
p_sol.rename("Pressure")

U_vis = VectorFunctionSpace(mesh0, "CG", 1)
u_vis = Function(U_vis)
u_vis.interpolate(u_sol)

coords = mesh0.coordinates.dat.data_ro
vel = u_vis.dat.data_ro

Vmag = FunctionSpace(mesh0, "CG", 1)
u_mag = Function(Vmag, name="VelocityMagnitude")
u_mag.interpolate(sqrt(dot(u_sol, u_sol)))

mid_z = 0.5
slice_func = Function(FunctionSpace(mesh0, "CG", 1))
slice_func.interpolate(u_mag)

triplot(mesh0)
plt.show()

# Angenommen, u_sol ist deine Lösung:
coords = mesh0.coordinates.dat.data_ro
vel = u_sol.dat.data_ro

# Nehmen wir nur jeden n-ten Punkt, sonst wird's zu viel
import matplotlib.pyplot as plt
import numpy as np

step = 100  # Stichprobe anpassen
x = coords[::step, 0]*1e6
y = coords[::step, 1]*1e6
z = coords[::step, 2]*1e6
u = vel[::step, 0]
v = vel[::step, 1]
w = vel[::step, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.quiver(x, y, z, u, v, w, length=20, normalize=True, color='blue')

ax.set_xlabel("x [µm]")
ax.set_ylabel("y [µm]")
ax.set_zlabel("z [µm]")
ax.set_title("3D Velocity field (sampled arrows)")
plt.show()


print("Simulation finished")