from firedrake import *

from perturbed_flow import F_p_r_z

def sample_grid(background_flow, R, H, W, L, a, h, Re, N_r=60, N_z=60, eps=1e-10):

    r_min = -W / 2 + a + eps
    r_max = W / 2 - a - eps
    z_min = -H / 2 + a + eps
    z_max = H / 2 - a - eps

    r_line = np.linspace(r_min, r_max, N_r)
    z_line = np.linspace(z_min, z_max, N_z)

    r, z = np.meshgrid(r_line, z_line, indexing='ij')  # (N_r, N_z)

    Fr = np.zeros_like(r)
    Fz = np.zeros_like(z)

    for i in range(N_r):
        for j in range(N_z):
            Fr[i, j], Fz[i, j] = F_p_r_z(background_flow, R, W, H, L, a, h, r[i, j], z[i, j], Re)
            print(i,j)
    points = np.stack([r.flatten(), z.flatten()], axis=1)
    F = np.stack([Fr.flatten(), Fz.flatten()], axis=1)

    return {
        "r": r, "z": z,
        "Fr": Fr, "Fz": Fz,
        "points": points,
        "F": F,
        "r_line": r_line, "z_line": z_line,
        "N_r": N_r, "N_z": N_z
    }







