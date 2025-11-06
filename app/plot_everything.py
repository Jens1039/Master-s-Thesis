import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import matplotlib.pyplot as plt
import pyvista as pv
import numpy as np
from firedrake import Function, FunctionSpace


def plot_2d_background_flow(mesh2d, u):
    coords = mesh2d.coordinates.dat.data_ro
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    zmin, zmax = float(coords[:, 1].min()), float(coords[:, 1].max())

    nxp, nzp = 160, 160
    xi = np.linspace(xmin, xmax, nxp)
    zi = np.linspace(zmin, zmax, nzp)
    Xi, Zi = np.meshgrid(xi, zi)


    pts = np.column_stack([Xi.ravel(), Zi.ravel()])
    U_at_list = u.at(pts)

    try:
        U_at = np.asarray(U_at_list, dtype=float)
        if U_at.ndim != 2 or U_at.shape[1] != 3:
            raise ValueError
    except Exception:
        U_at = np.vstack([np.asarray(v, dtype=float).ravel() for v in U_at_list])

    Ur  = U_at[:, 0].reshape(nzp, nxp)
    Uz  = U_at[:, 1].reshape(nzp, nxp)
    Uth = U_at[:, 2].reshape(nzp, nxp)

    Speed = np.sqrt(Ur**2 + Uz**2)
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


def plot_curved_channel_section_with_spherical_hole(mesh, style="wireframe", opacity=0.4, show_axes=True):


    Vcoord = mesh.coordinates.function_space()
    degree = Vcoord.ufl_element().degree()
    if degree > 1:
        print(f"Mesh is curved (degree={degree}) – linearised CG1-coordinates are used for the plot")
    V1 = VectorFunctionSpace(mesh, "CG", 1)
    coords_lin = Function(V1).interpolate(mesh.coordinates)

    coords = coords_lin.dat.data_ro.copy()

    cells = np.array(V1.cell_node_list, dtype=np.int64)
    num_cells = cells.shape[0]
    cells_pv = np.hstack([np.column_stack([np.full(num_cells, 4), cells]).ravel()])
    celltypes = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells_pv, celltypes, coords)

    plotter = pv.Plotter()
    plotter.add_mesh(grid, style=style, opacity=opacity, show_edges=True)
    if show_axes:
        plotter.add_axes()
    plotter.show()

    return grid


def plot_3d_background_flow(mesh3d, u_bg, p_bg, *, mode="surface", normal=(1, 0, 0), origin=None, opacity=0.15,
    show_vectors=True, vector_stride=6, vector_scale=None, streamline=True, n_seeds=300, seed_radius_factor=0.35,
    cmap="viridis", show_axes=True):

    coords = mesh3d.coordinates.dat.data_ro.copy()
    Vcoord = mesh3d.coordinates.function_space()
    cells_fd = np.asarray(Vcoord.cell_node_list, dtype=np.int64)
    n_cells, nverts = cells_fd.shape
    cells_pv = np.column_stack([np.full(n_cells, nverts, dtype=np.int64), cells_fd]).ravel()
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells_pv, celltypes, coords)

    if p_bg is not None:
        Q1 = FunctionSpace(mesh3d, "CG", 1)
        p1 = Function(Q1, name="p_vis")
        p1.interpolate(p_bg)
        grid.point_data["pressure"] = p1.dat.data_ro.copy()

    if u_bg is not None:
        V1 = VectorFunctionSpace(mesh3d, "CG", 1, dim=3)
        u1 = Function(V1, name="u_vis")
        u1.interpolate(u_bg)
        U = u1.dat.data_ro.copy()  # (n_verts, 3)
        grid.point_data["u_bg"] = U
        grid.point_data["u_mag"] = np.linalg.norm(U, axis=1)

    surf = grid.extract_surface()
    if origin is None:
        origin = tuple(grid.center)
    if mode == "clip":
        focus = grid.clip(normal=normal, origin=origin)
    elif mode == "slice":
        focus = grid.slice(normal=normal, origin=origin)
    elif mode == "surface":
        focus = surf
    else:
        raise ValueError("mode must be 'clip', 'slice' or 'surface'")

    glyphs = None
    if show_vectors and ("u_bg" in focus.point_data) and focus.n_points > 0:
        ids = np.arange(focus.n_points)[::max(1, int(vector_stride))]
        pts = focus.points[ids]
        seeds = pv.PolyData(pts)
        v = focus.point_data["u_bg"][ids]
        vmag = np.linalg.norm(v, axis=1)
        seeds["vectors"] = v
        seeds["u_mag"] = vmag
        if vector_scale is None:
            L = grid.length
            vmax = float(vmag.max()) if vmag.size else 1.0
            vector_scale = (0.08 * L / vmax) if vmax > 0 else 0.0
        glyphs = seeds.glyph(orient="vectors", scale="u_mag", factor=vector_scale)

    stream = None
    if streamline and ("u_bg" in grid.point_data) and grid.n_points > 0:
        L = grid.length
        r = seed_radius_factor * L
        phi = np.random.rand(n_seeds) * 2*np.pi
        costh = 2*np.random.rand(n_seeds) - 1.0
        sinth = np.sqrt(1 - costh**2)
        seeds = np.c_[r*sinth*np.cos(phi) + origin[0],
                      r*sinth*np.sin(phi) + origin[1],
                      r*costh             + origin[2]]
        seed_src = pv.PolyData(seeds)
        stream = grid.streamlines_from_source(
            seed_src,
            vectors="u_bg",
            max_time=L,
            initial_step_length=0.01*L,
            terminal_speed=1e-8,
            integrator_type=45,
        )

    p = pv.Plotter()
    p.add_mesh(surf, color="white", opacity=opacity, show_edges=False)
    scalars = "pressure" if ("pressure" in focus.point_data) else None
    p.add_mesh(focus, scalars=scalars, cmap=cmap, nan_opacity=0.0, show_edges=False)
    if glyphs is not None:
        p.add_mesh(glyphs, scalar_bar_args={"title": "|u|"})
    if stream is not None:
        p.add_mesh(stream.tube(radius=0.003*grid.length), color="w", opacity=0.85)
    if show_axes:
        p.add_axes()
    p.show()

    return grid


def plot_force_grid(data):

    r = np.array(data["r"])
    z = np.array(data["z"])
    Fr = np.array(data["Fr"])
    Fz = np.array(data["Fz"])
    Fmag = np.sqrt(Fr**2 + Fz**2)


    Nr, Nz = data.get("N_r", int(np.sqrt(r.size))), data.get("N_z", int(np.sqrt(z.size)))
    R = r.reshape((Nr, Nz))
    Z = z.reshape((Nr, Nz))
    Fr2 = Fr.reshape((Nr, Nz))
    Fz2 = Fz.reshape((Nr, Nz))
    Fmag2 = Fmag.reshape((Nr, Nz))


    fig, ax = plt.subplots(figsize=(6, 5))


    im = ax.contourf(R, Z, Fmag2, levels=40, cmap="viridis")


    c1 = ax.contour(R, Z, Fr2, levels=[0], colors="black", linewidths=1.2)
    c2 = ax.contour(R, Z, Fz2, levels=[0], colors="white", linewidths=1.2)


    step = max(1, Nr // 20)
    ax.quiver(R[::step, ::step], Z[::step, ::step], Fr2[::step, ::step], Fz2[::step, ::step], color="k", scale=40, width=0.004, alpha=0.8)


    ax.set_xlabel("r / a")
    ax.set_ylabel("z / a")
    ax.set_title("Cross-sectional Lift Force Field")
    fig.colorbar(im, ax=ax, label="|F′ₚ| / (ρU²a²)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()













