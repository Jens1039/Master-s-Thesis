import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import os
import matplotlib.patches as patches

from config_paper_parameters import *
from background_flow import background_flow
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow


_BG = None

def _init_bg_worker_defl(R, H, W, Q, Re):

    global _BG
    _BG = background_flow(R, H, W, Q, Re)
    _BG.solve_2D_background_flow()


def _compute_single_F_defl(task):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NETGEN_NUM_THREADS"] = "1"

    (i, j, r_loc, z_loc,
     R, H, W, Q, L, a,
     particle_maxh, global_maxh, Re_p_correct) = task

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R, W, H, L, a, particle_maxh, global_maxh, r_off=r_loc, z_off=z_loc
    )

    pf = perturbed_flow(mesh3d, tags, a, Re_p_correct, _BG)

    F_vec = pf.F_p()

    cx, cy, cz = tags["particle_center"]
    r0 = float(np.hypot(cx, cy))

    if r0 < 1e-14:
        ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ex0 = np.array([cx / r0, cy / r0, 0.0], dtype=float)

    ez0 = np.array([0.0, 0.0, 1.0], dtype=float)

    Fr = float(ex0 @ F_vec)
    Fz = float(ez0 @ F_vec)

    return (i, j, Fr, Fz)


def coarse_candidates_parallel_deflated(fp_eval, n_r=7, n_z=7, verbose=True, nproc=None,
                                        Re_bg_input=None):

    R = fp_eval.R
    H = fp_eval.H
    W = fp_eval.W
    Q = fp_eval.Q
    L = fp_eval.L
    a = fp_eval.a
    particle_maxh = fp_eval.particle_maxh
    global_maxh = fp_eval.global_maxh


    Re_p_correct = fp_eval.Re_p

    if Re_bg_input is None:
        raise ValueError("Re_bg_input (Input Reynolds number) must be provided for correct background flow physics!")

    eps = particle_maxh
    r_min = -W / 2 + a + eps
    r_max = W / 2 - a - eps
    z_min = -H / 2 + a + eps
    z_max = H / 2 - a - eps

    r_vals = np.linspace(r_min, r_max, n_r)
    z_vals = np.linspace(z_min, z_max, n_z)

    Fr_grid = np.zeros((n_r, n_z))
    Fz_grid = np.zeros((n_r, n_z))


    tasks = [
        (i, j, r_vals[i], z_vals[j],
         R, H, W, Q, L, a, particle_maxh, global_maxh, Re_p_correct)
        for i in range(n_r)
        for j in range(n_z)
    ]

    if nproc is None:
        nproc = mp.cpu_count()

    if verbose:
        print(f"Start parallel coarse grid with {nproc} processes.")
        print(f"  Background Re (Worker): {Re_bg_input:.2f}")
        print(f"  Particle Re_p (Worker): {Re_p_correct:.4f}")

    with mp.Pool(
            processes=nproc,
            initializer=_init_bg_worker_defl,
            initargs=(R, H, W, Q, Re_bg_input)
    ) as pool:

        results = []
        for res in tqdm(pool.imap_unordered(_compute_single_F_defl, tasks),
                        total=len(tasks), ncols=80):
            results.append(res)

    for (i, j, Fr, Fz) in results:
        Fr_grid[i, j] = Fr
        Fz_grid[i, j] = Fz

    phi = np.sqrt(Fr_grid ** 2 + Fz_grid ** 2)

    candidates = []
    for i in range(n_r):
        for j in range(n_z):
            val = phi[i, j]
            neigh = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < n_r and 0 <= jj < n_z:
                    neigh.append(phi[ii, jj])
            if neigh and all(val <= nb for nb in neigh):
                candidates.append(np.array([r_vals[i], z_vals[j]]))

    return candidates, r_vals, z_vals, phi, Fr_grid, Fz_grid


def plot_paper_reproduction(fp_eval, r_vals, z_vals, phi, Fr_grid, Fz_grid,
                            title="Figure 2 Reproduction", invert_xaxis=False):
    # Gitter für den Plot
    R_grid, Z_grid = np.meshgrid(r_vals, z_vals, indexing='ij')

    # Physikalische Parameter holen
    W = fp_eval.W
    H = fp_eval.H
    a = fp_eval.a

    # Dein Sicherheitsabstand (aus dem Code rekonstruiert)
    # In deiner Berechnung: eps = particle_maxh = 0.2 * a
    eps = fp_eval.particle_maxh

    # Berechnete "verbotene Zone" für den Partikel-MITTELPUNKT
    # Das ist der Bereich: Radius (a) + Sicherheitsabstand (eps)
    exclusion_dist = a + eps

    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. Kontur-Plot der Daten (nur dort, wo wir gerechnet haben)
    # Mehr Levels (40) für weichere Übergänge
    levels = np.linspace(phi.min(), phi.max(), 40)
    cs = ax.contourf(R_grid, Z_grid, phi, levels=levels, cmap="viridis", alpha=0.9)
    cbar = plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"Force Magnitude $\|\mathbf{F}\|$")

    # 2. Null-Linien (Gleichgewichtslagen)
    # Fr = 0 (Radiales Gleichgewicht)
    ax.contour(R_grid, Z_grid, Fr_grid, levels=[0], colors="cyan", linestyles="--", linewidths=2)
    # Fz = 0 (Axiales Gleichgewicht)
    ax.contour(R_grid, Z_grid, Fz_grid, levels=[0], colors="magenta", linestyles="-", linewidths=2)

    # 3. VISUALISIERUNG DES ECHTEN KANALS

    # A) Die echte Kanalwand (schwarzer Rahmen)
    wall_rect = patches.Rectangle((-W / 2, -H / 2), W, H,
                                  linewidth=3, edgecolor='black', facecolor='none', zorder=10)
    ax.add_patch(wall_rect)

    # B) Der "verbotene Bereich" (Grau)
    # Das ist der Bereich, in dem der Partikelmittelpunkt nicht sein darf.
    # Wir füllen den Bereich zwischen der echten Wand und deinen Daten grau.

    # Links
    ax.add_patch(patches.Rectangle((-W / 2, -H / 2), exclusion_dist, H,
                                   facecolor='gray', alpha=0.3, hatch='///'))
    # Rechts
    ax.add_patch(patches.Rectangle((W / 2 - exclusion_dist, -H / 2), exclusion_dist, H,
                                   facecolor='gray', alpha=0.3, hatch='///'))
    # Unten
    ax.add_patch(patches.Rectangle((-W / 2, -H / 2), W, exclusion_dist,
                                   facecolor='gray', alpha=0.3, hatch='///'))
    # Oben
    ax.add_patch(patches.Rectangle((-W / 2, H / 2 - exclusion_dist), W, exclusion_dist,
                                   facecolor='gray', alpha=0.3, hatch='///'))

    # 4. Achsen hart auf die physikalischen Maße setzen (+ etwas Rand)
    margin = 0.1
    ax.set_xlim(-W / 2 - margin, W / 2 + margin)
    ax.set_ylim(-H / 2 - margin, H / 2 + margin)
    ax.set_aspect('equal')  # Wichtig! Sonst verzerrt

    ax.set_xlabel("r (Radial coordinate)")
    ax.set_ylabel("z (Axial coordinate)")

    # Spiegelung korrigieren, falls gewünscht
    if invert_xaxis:
        ax.invert_xaxis()
        title += " (Inverted X)"

    ax.set_title(title)

    # Manuelle Legende erstellen
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='cyan', lw=2, ls='--', label='$F_r=0$'),
        Line2D([0], [0], color='magenta', lw=2, ls='-', label='$F_z=0$'),
        patches.Patch(facecolor='gray', alpha=0.3, hatch='///', label='Wall exclusion ($a+\epsilon$)'),
        patches.Patch(facecolor='none', edgecolor='black', lw=2, label='Physical Wall')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(f"force_contour_a={a}.png", dpi=300)









def newton_monitor(iter, x, Fx, delta):
    print(
        f"[Newton iter {iter:02d}] "
        f"x = ({x[0]:.5f}, {x[1]:.5f}) | "
        f"|F| = {np.linalg.norm(Fx):.3e} | "
        f"|dx| = {np.linalg.norm(delta):.3e}"
    )


class FpEvaluator:

    def __init__(self,
                 R, W, H, L, a,
                 particle_maxh, global_maxh,
                 Re, Re_p,
                 bg_flow=None,
                 Q=1.0):
        self.R = float(R)
        self.W = float(W)
        self.H = float(H)
        self.L = float(L)
        self.a = float(a)
        self.particle_maxh = float(particle_maxh)
        self.global_maxh = float(global_maxh)
        self.Re = float(Re)
        self.Re_p = float(Re_p)
        self.Q = float(Q)

        self.r_min = -0.5 * self.W + self.a
        self.r_max =  0.5 * self.W - self.a
        self.z_min = -0.5 * self.H + self.a
        self.z_max =  0.5 * self.H - self.a

        if bg_flow is None:
            self.bg_flow = background_flow(self.R, self.H, self.W, self.Q, self.Re)
            self.bg_flow.solve_2D_background_flow()
        else:
            self.bg_flow = bg_flow

    def _check_inside_box(self, r, z):
        return (self.r_min <= r <= self.r_max) and (self.z_min <= z <= self.z_max)

    def evaluate_F(self, x):

        r, z = float(x[0]), float(x[1])

        if not self._check_inside_box(r, z):
            raise ValueError(f"(r,z)=({r},{z}) außerhalb des zulässigen Bereichs.")

        mesh3d, tags = make_curved_channel_section_with_spherical_hole(
            self.R, self.H, self.W, self.L, self.a,
            self.particle_maxh, self.global_maxh,
            r_off=r, z_off=z
        )

        pf = perturbed_flow(mesh3d, tags, self.a, self.Re_p, self.bg_flow)
        F_cart = pf.F_p()

        cx, cy, cz = tags["particle_center"]
        r0 = np.hypot(cx, cy)
        if r0 < 1e-14:
            ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            ex0 = np.array([cx / r0, cy / r0, 0.0], dtype=float)
        ez0 = np.array([0.0, 0.0, 1.0], dtype=float)

        Fr = float(ex0 @ F_cart)
        Fz = float(ez0 @ F_cart)

        return np.array([Fr, Fz], dtype=float)


def approx_jacobian(F, x, Fx=None, eps_rel=1e-3, eps_abs=1e-4,
                    r_min=None, r_max=None, z_min=None, z_max=None):


    x = np.asarray(x, float)

    if Fx is None:
        Fx = F(x)

    clamp = not (r_min is None or r_max is None or z_min is None or z_max is None)

    J = np.zeros((2, 2), float)

    for i in range(2):
        h = max(eps_rel * (1.0 + abs(x[i])), eps_abs)

        dx = np.zeros(2)
        dx[i] = h

        xp = x + dx
        xm = x - dx

        if clamp:
            xp[0] = np.clip(xp[0], r_min, r_max)
            xp[1] = np.clip(xp[1], z_min, z_max)
            xm[0] = np.clip(xm[0], r_min, r_max)
            xm[1] = np.clip(xm[1], z_min, z_max)

        fp = F(xp)
        fm = F(xm)

        denom = xp[i] - xm[i]
        if abs(denom) < 1e-14:
            denom = h
            fp = F(x + dx)
            J[:, i] = (fp - Fx) / denom
        else:
            J[:, i] = (fp - fm) / denom

    return J



def make_deflated_F(F, known_roots, alpha=1e-2, p=2.0):

    roots = [np.asarray(rz, dtype=float) for rz in known_roots]

    def F_defl(x):
        x = np.asarray(x, dtype=float)
        val = F(x)
        if not roots:
            return val
        factor = 1.0
        for rstar in roots:
            dist = np.linalg.norm(x - rstar)
            if dist < 1e-12:
                dist = 1e-12
            factor *= (1.0 / dist**p + alpha)
        return factor * val

    return F_defl



def newton_deflated_2d(fp_eval, x0, known_roots, alpha=1e-2, p=2.0, tol_F=1e-3, tol_x=1e-6, max_iter=15, monitor=None,
                       ls_max_steps=8, ls_reduction=0.5):

    x = np.asarray(x0, dtype=float)

    def F_orig(x_vec):
        return np.asarray(fp_eval.evaluate_F(x_vec), dtype=float)

    roots = [np.asarray(rz, dtype=float) for rz in known_roots]

    def defl_factor(x_vec):
        if not roots:
            return 1.0
        fac = 1.0
        for rstar in roots:
            dist = np.linalg.norm(x_vec - rstar)
            if dist < 1e-12:
                dist = 1e-12
            fac *= (1.0 / dist**p + alpha)
        return fac

    def F_defl(x_vec): return defl_factor(x_vec) * F_orig(x_vec)


    def compute_alpha_max(x_vec, delta):
        alpha_max = 1.0
        r_min, r_max = fp_eval.r_min, fp_eval.r_max
        z_min, z_max = fp_eval.z_min, fp_eval.z_max

        for (xi, di, xmin, xmax) in [
            (x_vec[0], delta[0], r_min, r_max),
            (x_vec[1], delta[1], z_min, z_max),
        ]:
            if abs(di) < 1e-14:
                continue
            if di > 0:
                a_k = (xmax - xi) / di
            else:
                a_k = (xmin - xi) / di
            alpha_max = min(alpha_max, a_k)

        return max(min(alpha_max, 1.0), 0.0)

    F0 = F_orig(x)
    F0_norm = np.linalg.norm(F0)
    F0_defl = defl_factor(x) * F0

    if monitor is not None:
        monitor(0, x, F0, np.zeros_like(x))

    if F0_norm < tol_F:
        return x, True

    for k in range(1, max_iter + 1):

        J = approx_jacobian(
                F_defl, x, Fx=F0_defl,
                r_min=fp_eval.r_min, r_max=fp_eval.r_max,
                z_min=fp_eval.z_min, z_max=fp_eval.z_max
            )

        try:
            delta = np.linalg.solve(J, -F0_defl)
        except np.linalg.LinAlgError:
            print(f"[Abort] LinAlgError (singuläre Jacobi-Matrix) bei Iteration {k}, x={x}")
            return x, False

        alpha_max = compute_alpha_max(x, delta)
        if alpha_max <= 0:
            print(f"[Abort] alpha_max <= 0 (Randschnitt) bei Iteration {k}, x={x}, delta={delta}")
            return x, False

        alpha_ls = alpha_max
        F_trial = None

        for _ in range(ls_max_steps):
            x_trial = x + alpha_ls * delta

            if not (fp_eval.r_min <= x_trial[0] <= fp_eval.r_max and
                    fp_eval.z_min <= x_trial[1] <= fp_eval.z_max):
                alpha_ls *= ls_reduction
                continue

            F_trial = F_orig(x_trial)
            F_trial_norm = np.linalg.norm(F_trial)

            if F0_norm > 1e-1:
                accept = (F_trial_norm < F0_norm * (1 - 1e-2))
            else:
                accept = (F_trial_norm < F0_norm)

            if accept:
                break

            alpha_ls *= ls_reduction


        if F_trial is None or F_trial_norm >= F0_norm:
            print(f"[Abort] Linesearch-Stagnation bei Iteration {k}, x={x}, |F|={F0_norm:.3e}")
            return x, (F0_norm < tol_F)

        step = alpha_ls * delta
        x = x + step

        F0 = F_trial
        F0_norm = F_trial_norm
        F0_defl = defl_factor(x) * F0

        if monitor is not None:
            monitor(k, x, F0, step)

        if F0_norm < tol_F:
            return x, True
        if np.linalg.norm(step) < tol_x:
            return x, (F0_norm < tol_F)

    return x, (F0_norm < tol_F)





def find_equilibria_with_deflation(fp_eval, n_r=10, n_z=10, max_roots=20, skip_radius=0.02, newton_kwargs=None, verbose=True,
                                   coarse_data=None, max_candidates=None, refine_factor=4, boundary_tol=5e-3):

    if newton_kwargs is None:
        newton_kwargs = {}

    if coarse_data is None:
        if verbose:
            print("=== Coarse Grid ===")
        _, r_vals, z_vals, phi, _, _ = coarse_candidates_parallel_deflated(fp_eval, n_r=n_r, n_z=n_z, verbose=verbose)
    else:
        _, r_vals, z_vals, phi, _, _ = coarse_data

    if verbose:
        print("\n=== Kandidaten aus interpoliertem coarse grid ===")
    refined_candidates = refine_candidates_by_interpolation(
        r_vals, z_vals, phi,
        refine_factor=refine_factor,
        max_candidates=max_candidates
    )

    filtered_candidates = []
    for x0 in refined_candidates:
        r0, z0 = x0

        if (abs(r0 - fp_eval.r_min) < boundary_tol or
            abs(r0 - fp_eval.r_max) < boundary_tol or
            abs(z0 - fp_eval.z_min) < boundary_tol or
            abs(z0 - fp_eval.z_max) < boundary_tol):

            if verbose:
                print(f"[Skip] x0={x0} liegt zu nahe am Rand → verworfen.")
            continue

        filtered_candidates.append(x0)

    if verbose:
        for x in filtered_candidates:
            print(f"  Startkandidat x = ({x[0]: .4f}, {x[1]: .4f})")

    roots = []
    if verbose:
        print("\n=== Newton + Deflation ===")

    for x0 in filtered_candidates:

        if len(roots) >= max_roots:
            break

        if any(np.linalg.norm(x0 - r) < skip_radius for r in roots):
            if verbose:
                print(f"[Skip] x0={x0} (zu nah an existierender Wurzel).")
            continue

        if verbose:
            print(f"[OK] Starte Newton bei x0={x0}")

        x_root, ok_newton = newton_deflated_2d(
            fp_eval,
            x0,
            known_roots=roots,
            alpha=newton_kwargs.get("alpha", 1e-2),
            p=newton_kwargs.get("p", 2.0),
            tol_F=newton_kwargs.get("tol_F", 1e-3),
            tol_x=newton_kwargs.get("tol_x", 1e-6),
            max_iter=newton_kwargs.get("max_iter", 20),
            monitor=newton_kwargs.get("monitor", None),
            ls_max_steps=newton_kwargs.get("ls_max_steps", 8),
            ls_reduction=newton_kwargs.get("ls_reduction", 0.5)
        )

        if not ok_newton:
            if verbose:
                print(f"[Fail] Newton konvergiert nicht für x0={x0}")
            continue

        if any(np.linalg.norm(x_root - r) < skip_radius for r in roots):
            if verbose:
                print(f"[Dup] x_root={x_root} ist Duplikat.")
            continue

        roots.append(x_root)

        if verbose:
            Fvec = fp_eval.evaluate_F(x_root)
            print(f"  Neue Gleichgewichtslage #{len(roots)}:")
            print(f"    r = {x_root[0]:.5f}, z = {x_root[1]:.5f}, "
                  f"|F| = {np.linalg.norm(Fvec):.3e}")

    return np.array(roots)




def plot_equilibria_contour_with_zero_sets(fp_eval, r_vals, z_vals, phi,
                                           equilibria=None,
                                           stability_info=None,
                                           title="Force field and equilibrium positions",
                                           cmap="viridis",
                                           levels=20,
                                           figsize=(7, 5)):

    R, Z = np.meshgrid(r_vals, z_vals, indexing="ij")

    F_r = np.zeros_like(R)
    F_z = np.zeros_like(Z)

    for i in range(len(r_vals)):
        for j in range(len(z_vals)):
            Fr, Fz = fp_eval.evaluate_F([r_vals[i], z_vals[j]])
            F_r[i, j] = Fr
            F_z[i, j] = Fz

    plt.figure(figsize=figsize)

    cs = plt.contourf(R, Z, phi, levels=levels, cmap=cmap)
    plt.colorbar(cs, label=r"$\|\mathbf{F}_p(r,z)\|$")

    c1 = plt.contour(
        R, Z, F_r,
        levels=[0],
        colors="cyan",
        linestyles="--",
        linewidths=2
    )
    c1.collections[0].set_label("F_r = 0")

    c2 = plt.contour(
        R, Z, F_z,
        levels=[0],
        colors="magenta",
        linestyles="-",
        linewidths=2
    )
    c2.collections[0].set_label("F_z = 0")

    color_map = {
        "stabil": "green",
        "Sattel": "yellow",
        "instabil": "red",
        "grenzstabil / unklar": "gray",
        "unklar (NaN in Eigenwerten)": "gray",
    }

    if equilibria is not None and len(equilibria) > 0:
        for i, x in enumerate(equilibria):
            if stability_info is not None:
                eq_type = stability_info[i]["type"]
                color = color_map.get(eq_type, "gray")
                label = f"{eq_type} (EQ {i+1})"
            else:
                color = "red"
                label = f"EQ {i+1}"

            plt.scatter(
                x[0], x[1],
                s=120,
                color=color,
                edgecolors="black",
                linewidths=1.0,
                label=label
            )

    plt.xlabel("r")
    plt.ylabel("z")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()




def local_patch_test(x0, r_vals, z_vals, phi,
                     r_patch=3, z_patch=3, factor=1.0):

    r0, z0 = x0

    dr = np.abs(r_vals[1] - r_vals[0])
    dz = np.abs(z_vals[1] - z_vals[0])

    r_line = r0 + factor * dr * np.linspace(-1, 1, r_patch)
    z_line = z0 + factor * dz * np.linspace(-1, 1, z_patch)

    from scipy.interpolate import griddata
    RR, ZZ = np.meshgrid(r_vals, z_vals, indexing='ij')
    pts = np.stack([RR.flatten(), ZZ.flatten()], axis=1)
    vals = phi.flatten()

    R_patch, Z_patch = np.meshgrid(r_line, z_line, indexing='ij')
    patch_vals = griddata(pts, vals, (R_patch, Z_patch), method='cubic')

    if patch_vals is None:
        return False

    center_val = griddata(pts, vals, np.array([[r0, z0]]), method='cubic')[0]

    if not np.all(center_val <= patch_vals + 1e-12):
        return False

    edge_vals = np.concatenate([
        patch_vals[0, :], patch_vals[-1, :],
        patch_vals[:, 0], patch_vals[:, -1]
    ])
    if np.any(np.abs(center_val - edge_vals) < 1e-10):
        return False

    return True



def refine_candidates_by_interpolation(r_vals,
                                       z_vals,
                                       phi,
                                       refine_factor=4,
                                       max_candidates=None,
                                       min_dist=None):

    n_r, n_z = phi.shape

    n_r_fine = (n_r - 1) * refine_factor + 1
    n_z_fine = (n_z - 1) * refine_factor + 1

    r_fine = np.linspace(r_vals[0], r_vals[-1], n_r_fine)
    z_fine = np.linspace(z_vals[0], z_vals[-1], n_z_fine)

    phi_r_interp = np.zeros((n_r_fine, n_z), dtype=float)
    for j in range(n_z):
        phi_r_interp[:, j] = np.interp(r_fine, r_vals, phi[:, j])

    phi_fine = np.zeros((n_r_fine, n_z_fine), dtype=float)
    for i in range(n_r_fine):
        phi_fine[i, :] = np.interp(z_fine, z_vals, phi_r_interp[i, :])

    minima = []
    for i in range(n_r_fine):
        for j in range(n_z_fine):
            val = phi_fine[i, j]
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < n_r_fine and 0 <= jj < n_z_fine:
                    neighbors.append(phi_fine[ii, jj])
            if neighbors and all(val <= nb for nb in neighbors):
                minima.append((i, j, val))

    minima.sort(key=lambda t: t[2])

    candidates = []
    if min_dist is None:
        dr_coarse = abs(r_vals[1] - r_vals[0])
        dz_coarse = abs(z_vals[1] - z_vals[0])
        min_dist = 0.5 * np.sqrt(dr_coarse**2 + dz_coarse**2)

    for (i, j, val) in minima:
        r0 = r_fine[i]
        z0 = z_fine[j]
        x0 = np.array([r0, z0], float)

        if any(np.linalg.norm(x0 - c) < min_dist for c in candidates):
            continue

        candidates.append(x0)

        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    return candidates



def classify_single_equilibrium(fp_eval, x_eq, eps_rel=1e-4, ode_sign=-1.0, tol_eig=1e-6):

    x_eq = np.asarray(x_eq, dtype=float)

    def G(x):
        F = np.asarray(fp_eval.evaluate_F(x), dtype=float)
        return ode_sign * F

    Gx = G(x_eq)
    J = approx_jacobian(G, x_eq, Fx=Gx, eps_rel=eps_rel)

    eigvals, eigvecs = np.linalg.eig(J)
    real_parts = eigvals.real

    tr = np.trace(J)
    det = np.linalg.det(J)

    if np.any(np.isnan(real_parts)):
        eq_type = "unklar (NaN in Eigenwerten)"
    else:

        if real_parts[0] * real_parts[1] < -tol_eig**2:
            eq_type = "Sattel"

        elif np.all(real_parts < -tol_eig):
            eq_type = "stabil"

        elif np.all(real_parts > tol_eig):
            eq_type = "instabil"
        else:
            eq_type = "grenzstabil / unklar"

    return {
        "x_eq": x_eq,
        "J": J,
        "eigvals": eigvals,
        "trace": tr,
        "det": det,
        "type": eq_type,
    }


def classify_equilibria(fp_eval, equilibria, eps_rel=1e-4, ode_sign=-1.0, tol_eig=1e-6, verbose=True):

    equilibria = np.asarray(equilibria, dtype=float)
    if equilibria.ndim == 1:
        equilibria = equilibria[None, :]

    results = []

    for k, x_eq in enumerate(equilibria, start=1):
        info = classify_single_equilibrium(fp_eval,
                                           x_eq,
                                           eps_rel=eps_rel,
                                           ode_sign=ode_sign,
                                           tol_eig=tol_eig)
        results.append(info)

        if verbose:
            ev = info["eigvals"]
            print(f"EQ #{k}: x_eq = ({x_eq[0]:.6f}, {x_eq[1]:.6f})")
            print(f"        Eigenwerte(J_dyn): "
                  f"{ev[0].real:+.3e}{ev[0].imag:+.3e}i, "
                  f"{ev[1].real:+.3e}{ev[1].imag:+.3e}i")
            print(f"        Typ: {info['type']}\n")

    return results




