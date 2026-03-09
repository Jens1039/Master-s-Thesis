import json
import os
import numpy as np
from scipy.optimize import root
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator

from background_flow import background_flow
from find_equilbrium_points import F_p_grid
from nondimensionalization import *
from config_paper_parameters import *

RESULTS_DIR = "optimization_results"

if __name__ == '__main__':
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # ---Input-------------------------------------------------------------

    # Initial domain:
    r_min = -W / 2 + a + eps_rel*a
    r_max = W / 2 - a - eps_rel*a
    z_min = -H / 2 + a + eps_rel*a
    z_max = H / 2 - a - eps_rel*a
    r_vals = np.linspace(r_min, r_max, N_grid)
    z_vals = np.linspace(z_min, z_max, N_grid)

    # Initial guess:
    r_tilde = 0.6
    z_tilde = 0.0016
    a_tilde = 0.133

    # Target bifurcation parameter:
    a_star = 0.08

    # Tolerance
    epsilon = None


    # Numerical Parameter:
    a_vals = [a_tilde - 0.01, a_tilde ,a_tilde + 0.01]

    # ---------------------------------------------------------------------


    # ---1. Solve the eigenvalue problem to generate n initial guesses-----

    from find_equilbrium_points import force_grid
    force_grid_dict = force_grid(R, H, W, Q, rho, mu, a, N_grid, particle_maxh_rel, global_maxh_rel, eps_rel)

    '''
    {
        "grid_points": {"R": R_grid, "Z": Z_grid},
        "F_grid": {"Fr": Fr_grid, "Fz": Fz_grid},
        "interpolators": {"interp_Fr": f_grid.interp_Fr, "interp_Fz": f_grid.interp_Fz},
        "equilibria": classified_equilibria
    }
    '''

    def Solve_the_eigenvalue_problem_to_generate_initial_guesses(r_tilde, z_tilde, a_tilde, interp_Fr, interp_Fz):
        """
        Löst das Eigenwertproblem Fu(u_tilde, lambda_tilde) * phi = mu * phi nach
        Algorithmus 4.1 (Boullé et al.) und gibt alle (hier maximal 2) Eigenpaare zurück.

        Returns:
            Eine Liste mit Dictionaries, die jeweils (u_tilde, lambda_tilde, mu, phi) enthalten.
        """
        # 1. Berechne die Jacobi-Matrix Fu am Punkt (r_tilde, z_tilde)
        dFr_dr = interp_Fr(r_tilde, z_tilde, dx=1, dy=0)[0, 0]
        dFr_dz = interp_Fr(r_tilde, z_tilde, dx=0, dy=1)[0, 0]
        dFz_dr = interp_Fz(r_tilde, z_tilde, dx=1, dy=0)[0, 0]
        dFz_dz = interp_Fz(r_tilde, z_tilde, dx=0, dy=1)[0, 0]

        Fu = np.array([
            [dFr_dr, dFr_dz],
            [dFz_dr, dFz_dz]
        ])

        # 2. Löse das Eigenwertproblem Fu * phi = mu * phi
        eigenvalues, eigenvectors = np.linalg.eig(Fu)

        # 3. Sammle beide Eigenpaare in einer Liste
        initial_guesses = []

        # Da es eine 2x2 Matrix ist, gibt es exakt 2 Eigenwerte/Eigenvektoren
        for i in range(len(eigenvalues)):
            mu = eigenvalues[i]
            phi = eigenvectors[:, i]

            # Normiere den Eigenvektor (sehr wichtig für die Stabilität von Moore-Spence)
            phi = phi / np.linalg.norm(phi)

            # Speichere den Guess in der Liste
            initial_guesses.append({
                'u_tilde': np.array([r_tilde, z_tilde]),
                'lambda_tilde': a_tilde,  # Der Bifurkationsparameter a
                'mu': mu,  # Der Eigenwert
                'phi': phi  # Der zugehörige normierte Eigenvektor
            })

        return initial_guesses


    initial_guesses = Solve_the_eigenvalue_problem_to_generate_initial_guesses(r_tilde, z_tilde, a_tilde,
                                                             force_grid_dict["interpolators"["interp_Fr"]],
                                                             force_grid_dict["interpolators"["interp_Fz"]]
                                                             )

    # ---------------------------------------------------------------------


    # ---2 - 4. Solve the Moore–Spence system to obtain a solution---------

    def solve_moore_spence(guess_dict, interp_Fr_3d, interp_Fz_3d):
        """
        Löst das erweiterte Moore-Spence System H(y) = 0.
        y ist der 5-dimensionale Vektor: [r, z, a, phi_r, phi_z]
        """
        # Startwerte aus dem Initial Guess entpacken
        r0, z0 = guess_dict['u_tilde']
        a0 = guess_dict['lambda_tilde']
        phi0 = guess_dict['phi']

        # Der Vektor l für die Normierungsbedingung (l^T * phi = 1).
        # Wir nehmen den initialen Eigenvektor als Referenzrichtung.
        l_vec = phi0 / np.linalg.norm(phi0)

        # Initialer Vektor für den Solver
        y0 = np.array([r0, z0, a0, phi0[0], phi0[1]])

        def H(y):
            r, z, a = y[0], y[1], y[2]
            phi = np.array([y[3], y[4]])

            pt = np.array([r, z, a])

            # 1. Kraftfeld F(u, lambda)
            F_r = interp_Fr_3d(pt)[0]
            F_z = interp_Fz_3d(pt)[0]

            # 2. Jacobi-Matrix Fu numerisch auf dem 3D-Spline bilden
            eps = 1e-5
            dFr_dr = (interp_Fr_3d(np.array([r + eps, z, a]))[0] - interp_Fr_3d(np.array([r - eps, z, a]))[0]) / (
                        2 * eps)
            dFr_dz = (interp_Fr_3d(np.array([r, z + eps, a]))[0] - interp_Fr_3d(np.array([r, z - eps, a]))[0]) / (
                        2 * eps)
            dFz_dr = (interp_Fz_3d(np.array([r + eps, z, a]))[0] - interp_Fz_3d(np.array([r - eps, z, a]))[0]) / (
                        2 * eps)
            dFz_dz = (interp_Fz_3d(np.array([r, z + eps, a]))[0] - interp_Fz_3d(np.array([r, z - eps, a]))[0]) / (
                        2 * eps)

            Fu = np.array([
                [dFr_dr, dFr_dz],
                [dFz_dr, dFz_dz]
            ])

            # Singularitätsbedingung: Fu(u, lambda) * phi
            Fu_phi = Fu @ phi

            # 3. Normierungsbedingung: l^T * phi - 1
            norm_eq = np.dot(l_vec, phi) - 1.0

            # Gebe das 5D-Residuum zurück
            return np.array([F_r, F_z, Fu_phi[0], Fu_phi[1], norm_eq])

        # Das System mit scipy's hybriden Newton-Verfahren lösen
        solution = root(H, y0, method='hybr')

        if solution.success:
            bif_u = solution.x[0:2]
            bif_a = solution.x[2]
            bif_phi = solution.x[3:5]
            return True, bif_u, bif_a, bif_phi
        else:
            return False, None, None, None


    def build_local_3d_interpolators(R, H, W, Q, rho, mu, N_grid, particle_maxh_rel, global_maxh_rel, eps_rel, r_vals,
                                     z_vals, a_vals):
        """
        Kapselt die Berechnung mehrerer 2D-Kraftfelder für verschiedene a-Werte
        und baut daraus direkt die 3D-Interpolatoren.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        Fr_grid_list = []
        Fz_grid_list = []

        # Alle Ranks müssen durch diese Schleife gehen, da force_grid MPI nutzt!
        for current_a in a_vals:
            if rank == 0:
                print(f"Berechne CFD-Kraftfeld für a = {current_a:.4f}...")

            # Paralleler Aufruf von force_grid für das aktuelle 'current_a'
            current_grid_dict = force_grid(R, H, W, Q, rho, mu, current_a, N_grid, particle_maxh_rel, global_maxh_rel,
                                           eps_rel)

            # Nur Rank 0 sammelt die Matrizen ein
            if rank == 0 and current_grid_dict is not None:
                Fr_grid_list.append(current_grid_dict["F_grid"]["Fr"])
                Fz_grid_list.append(current_grid_dict["F_grid"]["Fz"])

        # Nur Rank 0 baut am Ende die 3D-Splines auf
        if rank == 0:
            # Staple die 2D-Arrays entlang einer neuen 3. Achse (Parameter a)
            Fr_3d_array = np.stack(Fr_grid_list, axis=-1)
            Fz_3d_array = np.stack(Fz_grid_list, axis=-1)

            # Erstelle die 3D-Interpolator-Objekte
            interp_Fr_3d = RegularGridInterpolator((r_vals, z_vals, a_vals), Fr_3d_array, bounds_error=False,
                                                   fill_value=None)
            interp_Fz_3d = RegularGridInterpolator((r_vals, z_vals, a_vals), Fz_3d_array, bounds_error=False,
                                                   fill_value=None)

            return interp_Fr_3d, interp_Fz_3d
        else:
            # Worker-Ranks brauchen die Interpolatoren nicht
            return None, None


    interp_Fr_3d, interp_Fz_3d = build_local_3d_interpolators(R, H, W, Q, rho, mu,
                                                              N_grid, particle_maxh_rel, global_maxh_rel, eps_rel,
                                                              r_vals, z_vals, a_vals)

    if rank == 0:

        valid_solutions = []

        for i, guess in enumerate(initial_guesses):
            success, bif_u, bif_a, bif_phi = solve_moore_spence(guess, interp_Fr_3d, interp_Fz_3d)

            if success:
                valid_solutions.append({
                    'u': bif_u,
                    'a': bif_a,
                    'phi': bif_phi
                })
                print(f"  Guess {i} konvergierte zu Bifurkation bei: r,z = {bif_u}, a = {bif_a}")
            else:
                print(f"  Guess {i} ist nicht konvergiert.")

    # ---------------------------------------------------------------------


    # ---5: Select the initial solution minimizing ||(r_tilde, z_tilde) - u_bar|| ---
    if rank == 0:
        if valid_solutions:
            # Unser ursprünglicher Startpunkt als Vektor
            u_tilde_vec = np.array([r_tilde, z_tilde])

            best_solution = None
            min_distance = float('inf')  # Setze den kleinsten Abstand initial auf unendlich

            # Finde die Lösung mit dem minimalen Abstand zu u_tilde
            for sol in valid_solutions:
                dist = np.linalg.norm(sol['u'] - u_tilde_vec)
                if dist < min_distance:
                    min_distance = dist
                    best_solution = sol

            # Das ist nun unser initialer Bifurkationspunkt (u_0, lambda_0, phi_0)
            u_0 = best_solution['u']
            a_0 = best_solution['a']
            phi_0 = best_solution['phi']

            print("\n--- Bester Bifurkationspunkt (Step 5) ---")
            print(f"Abstand zum Initial Guess: {min_distance:.6f}")
            print(f"u_0 (r, z): {u_0}")
            print(f"lambda_0 (a): {a_0}")

            # Du kannst dir diesen Punkt nun z.B. speichern oder direkt an
            # find_branch_point oder den Continuation-Algorithmus übergeben.

        else:
            print("\nWARNING: Kein Moore-Spence Guess ist konvergiert!")

    # ---------------------------------------------------------------------


    # ---6. a_0 = a_0------------------------------------------------------
    k = 1
    # ---------------------------------------------------------------------


    #---7. while (a - a_star)^2 > epsilon----------------------------------
    while (a - a_star)**2 > epsilon:
        pass

        #---8. Evaluate the shape functional J(Omega) and compute shape update








