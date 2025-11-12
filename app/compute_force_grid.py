import multiprocessing as mp
import numpy as np
from tqdm import tqdm

from background_flow import background_flow
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow

_BG = None

def init_bg_worker(R, H, W, Q, Re):
    global _BG
    _BG = background_flow(R, H, W, Q, Re)
    _BG.solve_2D_background_flow()


def _compute_single_force(arg):

    i, j, r_loc, z_loc, R, H, W, L, a, particle_maxh, global_maxh, Re = arg

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, W, H, L, a, particle_maxh, global_maxh, r_loc, z_loc)

    perturbed_flow_object = perturbed_flow(mesh3d, tags, a, _BG)

    F_vec = perturbed_flow_object.F_p()

    x0, y0, z0 = tags["center"]

    r0 = float(np.hypot(x0, y0))

    if r0 == 0.0:
        ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ex0 = np.array([x0 / r0, y0 / r0, 0.0], dtype=float)

    ez0 = np.array([0.0, 0.0, 1.0], dtype=float)
    Fr_val = float(ex0 @ F_vec)
    Fz_val = float(ez0 @ F_vec)
    return (i, j, Fr_val, Fz_val)


def build_grid(W, H, a, N_r, N_z, eps=1e-6):
    r_min = -W / 2 + a + eps
    r_max =  W / 2 - a - eps
    z_min = -H / 2 + a + eps
    z_max =  H / 2 - a - eps

    r_line = np.linspace(r_min, r_max, N_r)
    z_line = np.linspace(z_min, z_max, N_z)
    R, Z = np.meshgrid(r_line, z_line, indexing="ij")
    return R, Z


def sample_grid(R, H, W, Q, L, a, particle_maxh, global_maxh, Re, N_r, N_z, eps=1e-10, nproc=None):
    # eps absolutely necessary !!! without it process fails immediately
    r_min = -W / 2 + a + eps
    r_max =  W / 2 - a - eps
    z_min = -H / 2 + a + eps
    z_max =  H / 2 - a - eps

    r_line = np.linspace(r_min, r_max, N_r)
    z_line = np.linspace(z_min, z_max, N_z)
    r, z = np.meshgrid(r_line, z_line, indexing='ij')

    Fr = np.zeros_like(r, dtype=float)
    Fz = np.zeros_like(z, dtype=float)

    tasks = [(i, j, r[i, j], z[i, j], R, H, W, L, a, particle_maxh, global_maxh, Re)
             for i in range(N_r) for j in range(N_z)]

    if nproc is None:
        nproc = mp.cpu_count()

    print(f"Start parallelisation with {nproc} processes ...")

    with mp.Pool(processes=nproc,
                 initializer=init_bg_worker,
                 initargs=(R, H, W, Q, Re)) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(_compute_single_force, tasks),
                        total=len(tasks),
                        desc="Calculate Grid-points", ncols=80):
            results.append(res)

    for (i, j, Fr_val, Fz_val) in results:
        Fr[i, j] = Fr_val
        Fz[i, j] = Fz_val

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


