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

    D_h = (2*H*W)/(H + W)
    Re_p = Re * (a/D_h)**2

    perturbed_flow_object = perturbed_flow(mesh3d, tags, a, Re_p, _BG)

    F_vec = perturbed_flow_object.F_p()

    x0, y0, z0 = tags["particle_center"]

    r0 = float(np.hypot(x0, y0))

    if r0 == 0.0:
        ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ex0 = np.array([x0 / r0, y0 / r0, 0.0], dtype=float)

    ez0 = np.array([0.0, 0.0, 1.0], dtype=float)
    Fr_val = float(ex0 @ F_vec)
    Fz_val = float(ez0 @ F_vec)
    return (i, j, Fr_val, Fz_val)


def sample_grid(R, H, W, Q, L, a, particle_maxh, global_maxh, Re, N_r, N_z, eps=None, nproc=None):

    if eps is None: eps = 3*particle_maxh

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

    if nproc is None: nproc = mp.cpu_count()

    print(f"Start parallelisation with {nproc} processes ...")

    with mp.Pool(processes=nproc, initializer=init_bg_worker, initargs=(R, H, W, Q, Re)) as pool:
        results = []
        for res in tqdm(pool.imap_unordered(_compute_single_force, tasks), total=len(tasks), desc="Calculate Grid-points", ncols=80):
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


import vtk
from vtk.util.numpy_support import numpy_to_vtk

def write_force_grid_vts(data, filename):
    r = data["r"]   # shape (N_r, N_z)
    z = data["z"]   # shape (N_r, N_z)
    Fr = data["Fr"] # shape (N_r, N_z)
    Fz = data["Fz"] # shape (N_r, N_z)
    N_r = data["N_r"]
    N_z = data["N_z"]

    # --- Punkte anlegen (2D rz-Ebene → y = 0) -----------------------
    points = vtk.vtkPoints()
    for i in range(N_r):
        for j in range(N_z):
            points.InsertNextPoint(r[i, j], 0.0, z[i, j])

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(N_r, 1, N_z)   # (Nx, Ny=1, Nz)
    grid.SetPoints(points)

    # --- Skalarfelder Fr, Fz ----------------------------------------
    Fr_vtk = numpy_to_vtk(Fr.flatten(order="C"), deep=True)
    Fr_vtk.SetName("Fr")
    Fz_vtk = numpy_to_vtk(Fz.flatten(order="C"), deep=True)
    Fz_vtk.SetName("Fz")

    grid.GetPointData().AddArray(Fr_vtk)
    grid.GetPointData().AddArray(Fz_vtk)

    # --- Optional: Vektorfeld F = (Fr, 0, Fz) -----------------------
    F_vec = np.zeros((N_r * N_z, 3), dtype=float)
    F_vec[:, 0] = Fr.flatten(order="C")   # x-Komponente: Fr
    F_vec[:, 2] = Fz.flatten(order="C")   # z-Komponente: Fz

    F_vtk = numpy_to_vtk(F_vec, deep=True)
    F_vtk.SetName("F")
    F_vtk.SetNumberOfComponents(3)
    grid.GetPointData().AddArray(F_vtk)
    grid.GetPointData().SetActiveVectors("F")

    # --- Schreiben --------------------------------------------------
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(grid)
    writer.Write()

    print(f"Geschrieben: {filename}")
