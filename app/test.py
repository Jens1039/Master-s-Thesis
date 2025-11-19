import numpy as np

data = np.load("coarse_data.npz", allow_pickle=False)

r_vals = np.array(data["r_vals"], dtype=float).flatten()
z_vals = np.array(data["z_vals"], dtype=float).flatten()

phi = np.array(data["phi"], dtype=float)
Fr_grid = np.array(data["Fr_grid"], dtype=float)
Fz_grid = np.array(data["Fz_grid"], dtype=float)

candidates = data["candidates"]

print("r_vals:", type(r_vals), r_vals.shape, r_vals)
print("z_vals:", type(z_vals), z_vals.shape, z_vals)
print("phi:", type(phi), phi.shape)
print("Fr_grid:", type(Fr_grid), Fr_grid.shape)
print("Fz_grid:", type(Fz_grid), Fz_grid.shape)
