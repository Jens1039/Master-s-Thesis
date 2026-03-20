import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from scipy.optimize import root

from firedrake.adjoint import *

from app.perturbed_flow_full_navier_stokes import *


def apply_ale_deformation(mesh3d, tags, a_tilde, dr, dz, a_new):

    V_def = VectorFunctionSpace(mesh3d, "CG", 1)
    disp = TrialFunction(V_def)
    v = TestFunction(V_def)

    a_form = inner(grad(disp), grad(v)) * dx
    L_form = inner(Constant((0.0, 0.0, 0.0)), v) * dx

    x = SpatialCoordinate(mesh3d)
    x_p = Constant(tags["particle_center"])

    scale_factor = a_new / Constant(a_tilde) - 1.0

    disp_particle_expr = scale_factor * (x - x_p) + as_vector([dr, 0.0, dz])
    disp_particle = Function(V_def)
    trial_d = TrialFunction(V_def)
    test_d = TestFunction(V_def)

    solve(
        inner(trial_d, test_d) * dx == inner(disp_particle_expr, test_d) * dx,
        disp_particle,
        solver_parameters={"ksp_type": "cg", "pc_type": "jacobi"}
    )

    bcs = [
        DirichletBC(V_def, Constant((0.0, 0.0, 0.0)), tags["walls"]),
        DirichletBC(V_def, disp_particle, tags["particle"])
    ]

    displacement = Function(V_def)
    solve(a_form == L_form, displacement, bcs=bcs)

    mesh3d.coordinates.assign(mesh3d.coordinates + displacement)
    return displacement


def evaluate_exact_forces(mesh3d, tags, params, dr_val, dz_val, a_val):

    apply_ale_deformation(mesh3d, tags, params['a_tilde'], dr_val, dz_val, a_val)

    updated_tags = tags.copy()
    updated_tags["particle_center"] = (
        tags["particle_center"][0] + dr_val,
        tags["particle_center"][1],
        tags["particle_center"][2] + dz_val
    )

    NS = FullNavierStokesSolver(
        params['R_hat'], params['H_hat'], params['W_hat'],
        params['L_hat'], a_val, params['Re'], mesh3d, updated_tags
    )
    NS.solve_flow()
    F_total_r, _, F_total_z = NS.compute_particle_force()

    return F_total_r, F_total_z


def setup_adjoint_tape(mesh3d, tags, params):

    R_space = FunctionSpace(mesh3d, "R", 0)

    dr_ctrl = Function(R_space).assign(0.0)
    dz_ctrl = Function(R_space).assign(0.0)
    a_ctrl = Function(R_space).assign(params['a_tilde'])

    F_r_0, F_z_0 = evaluate_exact_forces(mesh3d, tags, params, dr_ctrl, dz_ctrl, a_ctrl)

    controls = [Control(dr_ctrl), Control(dz_ctrl), Control(a_ctrl)]
    F_x = ReducedFunctional(F_r_0, controls)
    F_z = ReducedFunctional(F_z_0, controls)

    return F_x, F_z, R_space


def evaluate_force_data(J_Fr, J_Fz, R_space, dr, dz, a):

    def make_scalar_control(value):
        ctrl = Function(R_space)
        ctrl.assign(float(value))
        return ctrl

    eval_ctrls = [make_scalar_control(dr), make_scalar_control(dz), make_scalar_control(a)]

    F_r_val = float(J_Fr(eval_ctrls))
    F_z_val = float(J_Fz(eval_ctrls))

    dFr_d = J_Fr.derivative()
    grad_Fr = np.array([comp.dat.data[0] for comp in dFr_d], dtype=float)

    dFz_d = J_Fz.derivative()
    grad_Fz = np.array([comp.dat.data[0] for comp in dFz_d], dtype=float)

    F_u = np.array([
        [grad_Fr[0], grad_Fr[1]],
        [grad_Fz[0], grad_Fz[1]]
    ], dtype=float)

    return F_r_val, F_z_val, grad_Fr, grad_Fz, F_u


if __name__ == '__main__':

    continue_annotation()

    # =========================================================================
    # --- INPUT
    # =========================================================================

    R_hat = 500.0
    H_hat = 2.0
    W_hat = 2.0
    L_hat = 30 * max(H_hat, W_hat)
    Re = 1.0

    # -------------------------------------------------------------------------

    # Initial guess (from the bifurcation diagram)
    r_tilde = 0.61
    z_tilde = 0.0
    a_tilde = 0.135

    # Initial domain
    mesh3d, tags = make_curved_channel_section_with_spherical_hole_periodic(
        R_hat, H_hat, W_hat, L=L_hat, a=a_tilde,
        particle_maxh=0.2 * a_tilde,
        global_maxh=0.2 * min(H_hat, W_hat),
        r_off=r_tilde, z_off=z_tilde
    )

    # target bifurcation parameter
    a_star = 0.08

    # tolerance
    epsilon = 1e-6

    # =========================================================================
    # --- 1.  Solve the eigenvalue problem to generate initial guesses
    # =========================================================================

    params = {
        'R_hat': R_hat,
        'H_hat': H_hat,
        'W_hat': W_hat,
        'L_hat': L_hat,
        'Re': Re,
        'r_tilde': r_tilde,
        'z_tilde': z_tilde,
        'a_tilde': a_tilde,
        'a_star': a_star,
        'epsilon': epsilon
    }

    F_x, F_z, R_space = setup_adjoint_tape(mesh3d, tags, params)

    F_r_val, F_z_val, grad_Fr, grad_Fz, F_u_matrix = evaluate_force_data(F_x, F_z, R_space,
                                                                         params['r_tilde'], params['z_tilde'], params['a_tilde'])

    eigenvalues, eigenvectors = np.linalg.eig(F_u_matrix)

    initial_guesses = []

    for i in range(len(eigenvalues)):

        phi_norm = eigenvectors[:, i] / np.linalg.norm(eigenvectors[:, i])

        guess_tuple = (
            np.array([params['r_tilde'], params['z_tilde']]),
            params['a_tilde'],
            phi_norm
        )
        initial_guesses.append(guess_tuple)

    print(f"\nGenerated initial guesses (n={len(initial_guesses)}):")
    for idx, guess in enumerate(initial_guesses):
        print(f"  Guess {idx + 1}: u_tilde={guess[0]}, lambda_tilde={guess[1]}, phi={guess[2]}")


    # =========================================================================
    # --- 2 - 4.  Solve the Moore–Spence system to obtain a solution ((r_i_bar, z_i_bar), a_i_bar, phi_i_bar)
    # =========================================================================

    def moore_spence_residual(vars_array):
        moore_spence_residual.eval_count += 1
        dr, dz, a, phi_1, phi_2 = vars_array
        phi_vec = np.array([phi_1, phi_2])

        # 1. Daten über das Tape auswerten
        F_r_val, F_z_val, _, _, F_u_matrix = evaluate_force_data(F_x, F_z, R_space, dr, dz, a)

        # 2. Rohe Teil-Residuen berechnen
        res_state_raw = np.array([F_r_val, F_z_val])
        res_eigen_raw = F_u_matrix @ phi_vec
        res_norm_raw = np.array([np.dot(phi_vec, phi_vec) - 1.0])

        # 3. DYNAMISCHE SKALIERUNG (nur beim allerersten Aufruf pro Guess berechnen)
        if moore_spence_residual.eval_count == 1:
            norm_state = np.linalg.norm(res_state_raw)
            norm_eigen = np.linalg.norm(res_eigen_raw)

            # Wir berechnen Faktoren, die den initialen Fehler auf ~1.0 normieren.
            # max(..., 1e-8) schützt vor Division durch Null, falls wir perfekt starten.
            moore_spence_residual.scale_state = 1.0 / max(norm_state, 1e-8)
            moore_spence_residual.scale_eigen = 1.0 / max(norm_eigen, 1e-8)

            # Die Normierungsbedingung (phi^2 - 1) liegt mathematisch ohnehin
            # in der Größenordnung 1.0, daher braucht sie keine Skalierung.
            moore_spence_residual.scale_norm = 1.0

        # 4. Skalierung auf die aktuellen Residuen anwenden
        res_state = res_state_raw * moore_spence_residual.scale_state
        res_eigen = res_eigen_raw * moore_spence_residual.scale_eigen
        res_norm = res_norm_raw * moore_spence_residual.scale_norm

        # 5. Fehlerkomponenten für den Monitor berechnen
        err_state = np.linalg.norm(res_state)
        err_eigen = np.linalg.norm(res_eigen)
        err_norm = np.linalg.norm(res_norm)
        total_error = np.sqrt(err_state ** 2 + err_eigen ** 2 + err_norm ** 2)

        print(f"    Eval {moore_spence_residual.eval_count:2d} | "
              f"dr: {dr:8.5f}, dz: {dz:8.5f}, a: {a:8.5f} | "
              f"Err: {total_error:7.1e} (F: {err_state:7.1e}, Eig: {err_eigen:7.1e}, Nrm: {err_norm:7.1e})")

        return np.concatenate([res_state, res_eigen, res_norm])

    valid_solutions = []

    for i, guess in enumerate(initial_guesses):
        print(f"\nStarte Newton-Löser für Guess {i + 1}...")

        moore_spence_residual.eval_count = 0

        u_guess, a_guess, phi_guess = guess
        x0 = np.concatenate([u_guess, [a_guess], phi_guess])

        sol = root(
            moore_spence_residual,
            x0,
            method='lm',
            options={'xtol': 1e-4, 'maxiter': 100, 'eps': 1e-5}
        )

        if sol.success:
            print(f"\n  >> Lösung für Guess {i + 1} gefunden in {moore_spence_residual.eval_count} Auswertungen!")
            dr_sol, dz_sol, a_sol, phi_1_sol, phi_2_sol = sol.x
            print(f"  Bifurkationspunkt (dr, dz): ({dr_sol:.5f}, {dz_sol:.5f}), a*: {a_sol:.5f}")
            valid_solutions.append((dr_sol, dz_sol, a_sol))
        else:
            print(f"\n  >> Löser für Guess {i + 1} fehlgeschlagen.")
            print(f"  Grund: {sol.message}")

    # =========================================================================
    # --- 5. Select the initial solution (u^0, a^0, phi^0)
    # =========================================================================

    best_sol = None
    min_dist = float('inf')
    for sol in valid_solutions:
        dr_s, dz_s, _ = sol
        u_s = np.array([dr_s, dz_s])
        u_tilde = np.array([params['r_tilde'], params['z_tilde']])
        dist = np.linalg.norm(u_s - u_tilde)

        if dist < min_dist:
            min_dist = dist
            best_sol = sol

    print(f"\n--- Bester gefundener Branch Point (Schritt 5) ---")
    print(f"  dr_0 = {best_sol[0]:.5f}")
    print(f"  dz_0 = {best_sol[1]:.5f}")
    print(f"  a_0 = {best_sol[2]:.5f}")





