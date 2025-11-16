from background_flow import background_flow
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import multiprocessing as mp
from tqdm import tqdm
import numpy as np



###############################################################################
#   DROP-IN Ersatz für coarse_candidates_parallel, kompatibel mit Deflation   #
###############################################################################

_BG = None   # globaler Background-Flow-Holder wie in deiner Vorlage

def _init_bg_worker_defl(R, H, W, Q, Re):
    """
    Initialisierung pro Worker:
    Jeder Prozess bekommt seine eigene Hintergrundströmung.
    """
    global _BG
    _BG = background_flow(R, H, W, Q, Re)
    _BG.solve_2D_background_flow()


def _compute_single_F_defl(task):
    """
    Berechnet EINEN Gitterpunkt (Fr,Fz) für das Deflations-Coarse-Grid.
    Gibt (i, j, ||F||) zurück.
    """
    (i, j, r_loc, z_loc,
     R, H, W, Q, L, a,
     particle_maxh, global_maxh, Re) = task

    # 3D Geometrie + Flow
    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R, W, H, L, a,
        particle_maxh, global_maxh,
        r_off=r_loc, z_off=z_loc
    )

    # Re_p neu berechnen wie in deiner Vorlage
    D_h = (2 * H * W) / (H + W)
    Re_p = Re * (a / D_h)**2

    pf = perturbed_flow(mesh3d, tags, a, Re_p, _BG)
    F_vec = pf.F_p()

    # lokale (r,z)-Richtungen
    x0, y0, z0 = tags["particle_center"]
    r0 = float(np.hypot(x0, y0))

    if r0 == 0.0:
        ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        ex0 = np.array([x0 / r0, y0 / r0, 0.0], dtype=float)

    ez0 = np.array([0.0, 0.0, 1.0], dtype=float)

    # Projektion
    Fr = float(ex0 @ F_vec)
    Fz = float(ez0 @ F_vec)

    return (i, j, np.sqrt(Fr**2 + Fz**2))


def coarse_candidates_parallel_deflated(fp_eval,
                                        n_r=7,
                                        n_z=7,
                                        verbose=True,
                                        nproc=None):
    """
    Drop-in Replacement für coarse_candidates_parallel.

    Gibt zurück:
      candidates, r_vals, z_vals, phi

    candidates: Liste von Koordinaten [r,z] lokaler Minima
    phi[i,j]: ||F(r_i,z_j)||

    Kompatibel mit deinem Deflationsverfahren.
    """

    R = fp_eval.R
    H = fp_eval.H
    W = fp_eval.W
    Q = fp_eval.Q
    L = fp_eval.L
    a = fp_eval.a
    particle_maxh = fp_eval.particle_maxh
    global_maxh = fp_eval.global_maxh
    Re = fp_eval.Re

    # EPS-Rand, super wichtig für Netgen – wie in deiner Vorlage:
    eps = 3 * particle_maxh

    r_min = -W/2 + a + eps
    r_max =  W/2 - a - eps
    z_min = -H/2 + a + eps
    z_max =  H/2 - a - eps

    r_vals = np.linspace(r_min, r_max, n_r)
    z_vals = np.linspace(z_min, z_max, n_z)

    phi = np.zeros((n_r, n_z))

    # Tasks generieren
    tasks = [
        (i, j, r_vals[i], z_vals[j],
         R, H, W, Q, L, a, particle_maxh, global_maxh, Re)
        for i in range(n_r)
        for j in range(n_z)
    ]

    if nproc is None:
        nproc = mp.cpu_count()

    if verbose:
        print(f"Starte paralleles coarse grid mit {nproc} Prozessen...")

    # Stabiler mp.Pool wie in deiner Vorlage
    with mp.Pool(
        processes=nproc,
        initializer=_init_bg_worker_defl,
        initargs=(R, H, W, Q, Re)
    ) as pool:

        results = []
        for res in tqdm(
                pool.imap_unordered(_compute_single_F_defl, tasks),
                total=len(tasks),
                desc="Compute coarse grid",
                ncols=80):
            results.append(res)

    # Resultate einsortieren
    for (i, j, normF) in results:
        phi[i, j] = normF

    # → lokale Minima sammeln
    candidates = []
    for i in range(n_r):
        for j in range(n_z):
            val = phi[i, j]
            neigh = []
            for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                ii, jj = i+di, j+dj
                if 0 <= ii < n_r and 0 <= jj < n_z:
                    neigh.append(phi[ii,jj])
            if neigh and all(val <= nb for nb in neigh):
                candidates.append(np.array([r_vals[i], z_vals[j]]))

    if verbose:
        print("\nLokale Minima:")
        for c in candidates:
            print(f"  {c}")

    return candidates, r_vals, z_vals, phi



# fp_evaluator.py (oder ans Ende deines Skripts)

import numpy as np
from background_flow import background_flow
from build_3d_geometry import make_curved_channel_section_with_spherical_hole
from perturbed_flow import perturbed_flow


class FpEvaluator:
    """
    Kapselt:
      - Hintergrundströmung (2D + 3D)
      - Geometrie
      - perturbed_flow
    und gibt für einen gegebenen Partikel-Offset (r,z) die Kraftkomponenten
    (F_r, F_z) zurück.
    """

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

        # Bereich für (r,z) im Kanal (Partikel muss reinpassen)
        self.r_min = -0.5 * self.W + self.a
        self.r_max =  0.5 * self.W - self.a
        self.z_min = -0.5 * self.H + self.a
        self.z_max =  0.5 * self.H - self.a

        # Hintergrundströmung nur einmal berechnen
        if bg_flow is None:
            self.bg_flow = background_flow(self.R, self.H, self.W, self.Q, self.Re)
            self.bg_flow.solve_2D_background_flow()
        else:
            self.bg_flow = bg_flow

    def _check_inside_box(self, r, z):
        return (self.r_min <= r <= self.r_max) and (self.z_min <= z <= self.z_max)

    def evaluate_F(self, x):
        """
        Eingabe: x = (r,z) in lokalen Kanal-Koordinaten.
        Ausgabe: np.array([F_r, F_z])
        """

        r, z = float(x[0]), float(x[1])

        if not self._check_inside_box(r, z):
            # Bei Bedarf kannst du hier z.B.  große Kraft zurückgeben,
            # statt zu raisen.
            raise ValueError(f"(r,z)=({r},{z}) außerhalb des zulässigen Bereichs.")

        # 3D-Geometrie mit Partikel an Position (r_off=r, z_off=z)
        mesh3d, tags = make_curved_channel_section_with_spherical_hole(
            self.R, self.H, self.W, self.L, self.a,
            self.particle_maxh, self.global_maxh,
            r_off=r, z_off=z
        )

        # Perturbed flow + Kraftberechnung
        pf = perturbed_flow(mesh3d, tags, self.a, self.Re_p, self.bg_flow)
        F_cart = pf.F_p()   # 3D-Vektor in globalen kartesischen Koordinaten

        # Radial-/Axial-Richtung am Partikel-Zentrum konstruieren
        cx, cy, cz = tags["particle_center"]
        r0 = np.hypot(cx, cy)
        if r0 < 1e-14:
            ex0 = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            ex0 = np.array([cx / r0, cy / r0, 0.0], dtype=float)
        ez0 = np.array([0.0, 0.0, 1.0], dtype=float)

        # Da pf.F_p bereits auf ex0, ez0 projiziert ist, reicht einfaches Skalarprodukt
        Fr = float(ex0 @ F_cart)
        Fz = float(ez0 @ F_cart)

        return np.array([Fr, Fz], dtype=float)


import numpy as np

def approx_jacobian(F, x, Fx=None, eps_rel=1e-6):
    """
    Numerische Approximation der 2x2-Jacobi-Matrix von F an x
    mit Vorwärtsdifferenzen.

    F: R^2 -> R^2, x: shape (2,)
    Fx: optional F(x), um eine Auswertung zu sparen.
    """
    x = np.asarray(x, dtype=float)
    if Fx is None:
        Fx = F(x)

    J = np.zeros((2, 2), dtype=float)

    for i in range(2):
        h = eps_rel * (1.0 + abs(x[i]))
        dx = np.zeros(2, dtype=float)
        dx[i] = h
        fp = F(x + dx)
        J[:, i] = (fp - Fx) / h

    return J



def make_deflated_F(F, known_roots, alpha=1e-2, p=2.0):
    """
    Deflation:
        F_defl(x) = (Produkt_k (1/||x-x_k||^p + alpha)) * F(x)
    Nullstellen von F bleiben Nullstellen von F_defl.
    """
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


def newton_deflated_2d(fp_eval,
                        x0,
                        known_roots,
                        alpha=1e-2,      # Deflation-Parameter
                        p=2.0,           # Deflations-Exponent
                        tol_F=1e-3,      # sinnvoller: 1e-3 statt 1e-6
                        tol_x=1e-6,
                        max_iter=15,
                        monitor=None,
                        ls_max_steps=8,
                        ls_reduction=0.5):
    """
    Newton-Verfahren auf F_p(r,z)=0 mit Deflation + Backtracking-Linesearch.

    - F_orig(x) = echte Kraft F_p(r,z)
    - F_defl(x) = defl_factor(x) * F_orig(x)
    - Newton-Richtung aus J(F_defl)
    - Linesearch minimiert ||F_orig(x + alpha * delta)|| entlang delta
      (monotone Abnahme von ||F|| angestrebt)

    known_roots: Liste schon gefundener Wurzeln (für Deflation).
    """

    x = np.asarray(x0, dtype=float)

    # --- echte Kraft ---
    def F_orig(x_vec):
        return np.asarray(fp_eval.evaluate_F(x_vec), dtype=float)

    # --- Deflationsfaktor ---
    roots = [np.asarray(rz, dtype=float) for rz in known_roots]

    def defl_factor(x_vec):
        if not roots:
            return 1.0
        x_vec = np.asarray(x_vec, dtype=float)
        fac = 1.0
        for rstar in roots:
            dist = np.linalg.norm(x_vec - rstar)
            if dist < 1e-12:
                dist = 1e-12
            fac *= (1.0 / dist**p + alpha)
        return fac

    def F_defl(x_vec):
        return defl_factor(x_vec) * F_orig(x_vec)

    # helper: Domain-Check / alpha_max
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
            if di > 0.0:
                a_k = (xmax - xi) / di
            else:
                a_k = (xmin - xi) / di
            if a_k < alpha_max:
                alpha_max = a_k

        if alpha_max < 0.0:
            alpha_max = 0.0
        if alpha_max > 1.0:
            alpha_max = 1.0
        return alpha_max

    # --- initial ---
    F0 = F_orig(x)
    F0_norm = np.linalg.norm(F0)
    F0_defl = defl_factor(x) * F0

    if monitor is not None:
        monitor(0, x, F0, np.zeros_like(x))

    # Falls wir zufällig schon sehr nah an einer Wurzel sind:
    if F0_norm < tol_F:
        return x, True

    for k in range(1, max_iter + 1):

        # Jacobian der deflatierten Funktion
        J = approx_jacobian(F_defl, x, Fx=F0_defl)

        try:
            delta = np.linalg.solve(J, -F0_defl)
        except np.linalg.LinAlgError:
            return x, False

        # maximaler Schritt, um im zulässigen (r,z)-Bereich zu bleiben
        alpha_max = compute_alpha_max(x, delta)
        if alpha_max <= 0.0:
            return x, False

        # Backtracking-Linesearch auf ||F_orig||
        alpha_ls = alpha_max
        F_trial = None
        F_trial_norm = None

        for ls_it in range(ls_max_steps):
            x_trial = x + alpha_ls * delta

            # Domain-Check (nochmal zur Sicherheit)
            if not (fp_eval.r_min <= x_trial[0] <= fp_eval.r_max and
                    fp_eval.z_min <= x_trial[1] <= fp_eval.z_max):
                alpha_ls *= ls_reduction
                continue

            F_trial = F_orig(x_trial)
            F_trial_norm = np.linalg.norm(F_trial)

            if F_trial_norm < F0_norm:
                # Verbesserungs-Schritt gefunden
                break

            alpha_ls *= ls_reduction

        # Linesearch hat nix Besseres gefunden → Abbruch (stagnation)
        if F_trial is None or F_trial_norm >= F0_norm:
            # akzeptiere aktuellen Punkt, falls schon klein genug
            if F0_norm < tol_F:
                return x, True
            else:
                return x, False

        # Update
        step = alpha_ls * delta
        x = x + step
        F0 = F_trial
        F0_norm = F_trial_norm
        F0_defl = defl_factor(x) * F0

        if monitor is not None:
            monitor(k, x, F0, step)

        # Konvergenzkriterium
        if F0_norm < tol_F:
            return x, True
        if np.linalg.norm(step) < tol_x:
            # Schritt sehr klein, aber Residuum noch nicht: als stagnation werten
            return x, (F0_norm < tol_F)

    # nach max_iter: letzter Check auf ||F||
    if F0_norm < tol_F:
        return x, True
    return x, False




def newton_monitor(iter, x, F_orig, delta):
    print(
        f"[Newton iter {iter:02d}] "
        f"x = ({x[0]: .5f}, {x[1]: .5f}) | "
        f"|F| = {np.linalg.norm(F_orig):.3e} | "
        f"|dx| = {np.linalg.norm(delta):.3e}"
    )




def find_equilibria_with_deflation(fp_eval,
                                   n_r=9,
                                   n_z=9,
                                   max_roots=20,
                                   skip_radius=0.02,
                                   newton_kwargs=None,
                                   verbose=True,
                                   coarse_data=None,
                                   max_candidates=None,
                                   refine_factor=4):
    """
    Findet mehrere Gleichgewichtslagen F_p(r,z)=0.

    Strategie:
      1) coarse grid (echter PDE-Aufruf) -> phi(r,z) = ||F_p||
      2) Interpolation von phi auf feineres Gitter
      3) lokale Minima dieses interpolierten Feldes als Startpunkte
      4) Newton + Deflation mit exakter Liniensuche auf echter Kraft
    """

    if newton_kwargs is None:
        newton_kwargs = {}

    # --- 1) coarse grid holen ---
    if coarse_data is None:
        if verbose:
            print("=== Coarse Grid (nur ein Run!) ===")
        _, r_vals, z_vals, phi = coarse_candidates_parallel_deflated(
            fp_eval,
            n_r=n_r,
            n_z=n_z,
            verbose=verbose
        )
    else:
        _, r_vals, z_vals, phi = coarse_data

    # --- 2) verfeinerte Kandidaten aus Interpolation ---
    if verbose:
        print("\n=== Kandidaten aus interpoliertem coarse grid ===")
    refined_candidates = refine_candidates_by_interpolation(
        r_vals,
        z_vals,
        phi,
        refine_factor=refine_factor,
        max_candidates=max_candidates
    )

    if verbose:
        for x in refined_candidates:
            print(f"  Startkandidat x = ({x[0]: .4f}, {x[1]: .4f})")

    roots = []

    if verbose:
        print("\n=== Newton + Deflation ===")

    # --- 3) Newton + Deflation über verfeinerte Startpunkte ---
    for x0 in refined_candidates:

        if len(roots) >= max_roots:
            break

        # skip, falls nahe an bereits gefundener Wurzel
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
            max_iter=newton_kwargs.get("max_iter", 15),
            monitor=newton_kwargs.get("monitor", None),
            ls_max_steps=newton_kwargs.get("ls_max_steps", 8),
            ls_reduction=newton_kwargs.get("ls_reduction", 0.5)
        )

        if not ok_newton:
            if verbose:
                print(f"[Fail] Newton konvergiert nicht für x0={x0}")
            continue

        # Duplikate vermeiden
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




import numpy as np
import matplotlib.pyplot as plt


def plot_equilibria_contour(fp_eval, r_vals, z_vals, phi,
                            equilibria=None,
                            stability_info=None,
                            title="Force field and equilibrium positions",
                            cmap="viridis",
                            levels=20,
                            figsize=(7, 5)):

    R, Z = np.meshgrid(r_vals, z_vals, indexing="ij")

    plt.figure(figsize=figsize)
    cs = plt.contourf(R, Z, phi, levels=levels, cmap=cmap)
    plt.colorbar(cs, label=r"$\|\mathbf{F}_p(r,z)\|$")


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

            plt.scatter(x[0], x[1],
                        s=120,
                        color=color,
                        edgecolors='black',
                        linewidths=1.0,
                        label=label)

    plt.xlabel("r")
    plt.ylabel("z")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()



def local_patch_test(x0, r_vals, z_vals, phi,
                     r_patch=3, z_patch=3, factor=1.0):
    """
    Teste, ob (r0,z0) ein echtes lokales Minimum der FELD-Landschaft ist,
    nicht nur des coarse grids.

    factor:
        wie groß das lokale Patch relativ zur coarse-grid-Auflösung sein soll.
    """

    r0, z0 = x0

    # coarse-grid Abstände schätzen:
    dr = np.abs(r_vals[1] - r_vals[0])
    dz = np.abs(z_vals[1] - z_vals[0])

    # Patch definieren
    r_line = r0 + factor * dr * np.linspace(-1, 1, r_patch)
    z_line = z0 + factor * dz * np.linspace(-1, 1, z_patch)

    # Interpolation der phi-Werte
    # griddata ist perfekt dafür
    from scipy.interpolate import griddata
    RR, ZZ = np.meshgrid(r_vals, z_vals, indexing='ij')
    pts = np.stack([RR.flatten(), ZZ.flatten()], axis=1)
    vals = phi.flatten()

    R_patch, Z_patch = np.meshgrid(r_line, z_line, indexing='ij')
    patch_vals = griddata(pts, vals, (R_patch, Z_patch), method='cubic')

    if patch_vals is None:
        return False

    center_val = griddata(pts, vals, np.array([[r0, z0]]), method='cubic')[0]

    # Ist Zentrum minimal?
    if not np.all(center_val <= patch_vals + 1e-12):
        return False

    # Ist Zentrum NICHT auf einer Patch-Randlinie minimal?
    # Wenn ja → Wand-Minimum → verwerfen
    edge_vals = np.concatenate([
        patch_vals[0, :], patch_vals[-1, :],
        patch_vals[:, 0], patch_vals[:, -1]
    ])
    if np.any(np.abs(center_val - edge_vals) < 1e-10):
        return False

    return True

import numpy as np

def refine_candidates_by_interpolation(r_vals,
                                       z_vals,
                                       phi,
                                       refine_factor=4,
                                       max_candidates=None,
                                       min_dist=None):
    """
    Verfeinert die Kandidaten durch bilineare Interpolation des coarse grids.

    Eingabe:
      r_vals: 1D-Array der coarse r-Koordinaten (monoton)
      z_vals: 1D-Array der coarse z-Koordinaten (monoton)
      phi:    2D-Array (n_r x n_z) mit ||F||-Werten auf dem coarse grid

    Parameter:
      refine_factor: wie viele Unterteilungen pro coarse-Zelle
      max_candidates: wie viele Minima maximal zurückgeben
      min_dist: minimaler Abstand zwischen zwei Kandidaten (in (r,z)-Ebene)

    Ausgabe:
      candidates_refined: Liste von np.array([r,z]) der verfeinerten Minima
    """

    n_r, n_z = phi.shape

    # Feineres Gitter
    n_r_fine = (n_r - 1) * refine_factor + 1
    n_z_fine = (n_z - 1) * refine_factor + 1

    r_fine = np.linspace(r_vals[0], r_vals[-1], n_r_fine)
    z_fine = np.linspace(z_vals[0], z_vals[-1], n_z_fine)

    # Bilineare Interpolation via zwei 1D-Interpolationen:
    # 1) in r-Richtung
    phi_r_interp = np.zeros((n_r_fine, n_z), dtype=float)
    for j in range(n_z):
        phi_r_interp[:, j] = np.interp(r_fine, r_vals, phi[:, j])

    # 2) in z-Richtung
    phi_fine = np.zeros((n_r_fine, n_z_fine), dtype=float)
    for i in range(n_r_fine):
        phi_fine[i, :] = np.interp(z_fine, z_vals, phi_r_interp[i, :])

    # Lokale Minima auf dem feinen Gitter finden (nur innere Punkte)
    minima = []
    for i in range(1, n_r_fine - 1):
        for j in range(1, n_z_fine - 1):
            val = phi_fine[i, j]
            neighbors = [
                phi_fine[i-1, j], phi_fine[i+1, j],
                phi_fine[i, j-1], phi_fine[i, j+1]
            ]
            if all(val <= nb for nb in neighbors):
                minima.append((i, j, val))

    # sortiere nach kleinster phi
    minima.sort(key=lambda t: t[2])

    candidates = []
    if min_dist is None:
        # grobe default-Wahl: halbe coarse-Zellengröße
        dr_coarse = abs(r_vals[1] - r_vals[0])
        dz_coarse = abs(z_vals[1] - z_vals[0])
        min_dist = 0.5 * np.sqrt(dr_coarse**2 + dz_coarse**2)

    for (i, j, val) in minima:
        r0 = r_fine[i]
        z0 = z_fine[j]
        x0 = np.array([r0, z0], float)

        # Duplikate vermeiden
        if any(np.linalg.norm(x0 - c) < min_dist for c in candidates):
            continue

        candidates.append(x0)

        if max_candidates is not None and len(candidates) >= max_candidates:
            break

    return candidates

import numpy as np

def classify_single_equilibrium(fp_eval,
                                x_eq,
                                eps_rel=1e-4,
                                ode_sign=-1.0,
                                tol_eig=1e-6):
    """
    Klassifiziert eine einzelne Gleichgewichtslage x_eq für das ODE

        x_dot = ode_sign * F_p(r,z)

    Standard: ode_sign = -1.0  -->  x_dot = -F_p(r,z)

    Rückgabe: dict mit Jacobian, Eigenwerten, Typ (stabil / Sattel / instabil / unklar)
    """

    x_eq = np.asarray(x_eq, dtype=float)

    # Dynamikfeld G(x) = ode_sign * F_p(x)
    def G(x):
        F = np.asarray(fp_eval.evaluate_F(x), dtype=float)
        return ode_sign * F

    # Jacobian von G an x_eq
    Gx = G(x_eq)
    J = approx_jacobian(G, x_eq, Fx=Gx, eps_rel=eps_rel)

    # Eigenwerte
    eigvals, eigvecs = np.linalg.eig(J)
    real_parts = eigvals.real

    tr = np.trace(J)
    det = np.linalg.det(J)

    # Klassifikation anhand der Realteile
    if np.any(np.isnan(real_parts)):
        eq_type = "unklar (NaN in Eigenwerten)"
    else:
        # Produkt der Realteile gut negativ -> sicher Sattel
        if real_parts[0] * real_parts[1] < -tol_eig**2:
            eq_type = "Sattel"
        # beide Realteile deutlich < 0  -> stabil
        elif np.all(real_parts < -tol_eig):
            eq_type = "stabil"
        # beide Realteile deutlich > 0  -> instabil
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


def classify_equilibria(fp_eval,
                        equilibria,
                        eps_rel=1e-4,
                        ode_sign=-1.0,
                        tol_eig=1e-6,
                        verbose=True):
    """
    Nimmt ein Array von Gleichgewichtslagen (shape (n,2)) und gibt
    für jede Lage eine Klassifikation zurück.

    ode_sign = -1.0 bedeutet, dass wir die Dynamik x_dot = -F_p(r,z)
    betrachten (analog zu (2.35) ohne Drag-Koeffizienten).
    """
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


