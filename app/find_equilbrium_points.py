import numpy as np

from compute_force_grid import _compute_single_force
from scipy.optimize import root

def force_rz(rz, R, H, W, L, a, particle_maxh, global_maxh, Re):
    """
    Wertet die Kraft F(r,z) = (F_r, F_z) an einer gegebenen Partikelposition aus.
    Nutzt dein vorhandenes _compute_single_force.
    """
    r, z = rz
    # i,j sind für die Nullstellensuche irrelevant → Dummy-Werte
    i = j = 0
    _, _, Fr_val, Fz_val = _compute_single_force(
        (i, j, float(r), float(z), R, H, W, L, a, particle_maxh, global_maxh, Re)
    )
    return np.array([Fr_val, Fz_val], dtype=float)


def deflated_force_rz(rz, roots, R, H, W, L, a, particle_maxh, global_maxh, Re,
                      alpha=1e-2, p=2):
    """
    Deflatierte Residuum-Funktion:
        F_defl(rz) = (Produkt_k 1/(||rz - root_k||^p + alpha)) * F(rz)
    """
    F = force_rz(rz, R, H, W, L, a, particle_maxh, global_maxh, Re)

    if not roots:
        return F

    factor = 1.0
    rz = np.asarray(rz, dtype=float)

    for r0 in roots:
        d2 = np.sum((rz - r0)**2)
        factor *= 1.0 / (d2**(p/2) + alpha)

    return factor * F


def find_all_equilibria_deflation(
        R, H, W, Q, L, a,
        particle_maxh, global_maxh, Re,
        n_starts=50,
        alpha=1e-2, p=2,
        tol_root=1e-8,
        tol_dist=5e-3,
        eps=None,
        verbose=True
):
    """
    Findet mehrere Nullstellen der Querkraft F(r,z) mittels Deflation.

    Rückgabe:
        roots: np.ndarray der Form (N_roots, 2) mit [r_star, z_star]
    """

    # Hintergrundströmung EINMAL im Hauptprozess berechnen
    init_bg_worker(R, H, W, Q, Re)

    if eps is None:
        eps = 3 * particle_maxh

    # Gültiger Bereich wie in sample_grid
    r_min = -W / 2 + a + eps
    r_max = W / 2 - a - eps
    z_min = -H / 2 + a + eps
    z_max = H / 2 - a - eps

    roots = []

    def F_defl_local(rz):
        return deflated_force_rz(
            rz, roots,
            R, H, W, L, a, particle_maxh, global_maxh, Re,
            alpha=alpha, p=p
        )

    for k in range(n_starts):
        # zufälliger Startpunkt im erlaubten Rechteck
        r0 = np.random.uniform(r_min, r_max)
        z0 = np.random.uniform(z_min, z_max)
        x0 = np.array([r0, z0], dtype=float)

        sol = root(F_defl_local, x0=x0, tol=tol_root)

        if not sol.success:
            if verbose:
                print(f"[{k + 1}/{n_starts}] root: kein Konvergenz (msg = {sol.message})")
            continue

        r_star, z_star = sol.x

        # bleibt innerhalb des erlaubten Bereichs?
        if not (r_min <= r_star <= r_max and z_min <= z_star <= z_max):
            if verbose:
                print(f"[{k + 1}/{n_starts}] Lösung außerhalb des Bereichs verworfen: {sol.x}")
            continue

        # Sicherheitsabstand zur Wand
        if (min(r_star - r_min, r_max - r_star,
                z_star - z_min, z_max - z_star) < eps):
            if verbose:
                print(f"[{k + 1}/{n_starts}] Lösung zu nahe an der Wand verworfen: {sol.x}")
            continue

        # Ist das wirklich eine neue Nullstelle (nicht nur numerisch dieselbe)?
        if roots:
            dists = [np.linalg.norm(sol.x - r0) for r0 in roots]
            if min(dists) < tol_dist:
                if verbose:
                    print(f"[{k + 1}/{n_starts}] Duplikat (Abstand {min(dists):.3e}) → ignoriert.")
                continue

        roots.append(sol.x)
        if verbose:
            print(f"[{k + 1}/{n_starts}] Neue Nullstelle gefunden: r={r_star:.4f}, z={z_star:.4f}")

    if verbose:
        print(f"Fertig. Gefundene Anzahl Nullstellen: {len(roots)}")

    if roots:
        return np.vstack(roots)
    else:
        return np.zeros((0, 2))
