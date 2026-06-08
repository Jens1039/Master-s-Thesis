import os
os.environ["OMP_NUM_THREADS"] = "1"

from firedrake import *
import numpy as np

from problem_setup import *
from background_flow import *
from build_3d_geometry_gmsh import make_curved_channel_section_with_spherical_hole


class perturbed_flow:

    def __init__(self, R, H, W, L, a, Re, mesh3d, tags, u_bar, p_bar):

        self.R = R
        self.H = H
        self.W = W
        self.a = a
        self.L = L
        self.Re = Re

        self.u_bar = u_bar
        self.p_bar = p_bar

        self.mesh3d = mesh3d
        self.tags = tags

        # We therefore formulate all of our vectors in this coordinate system
        self.x = SpatialCoordinate(self.mesh3d)

        # Define the particle center
        self.x_p = Constant(self.tags["particle_center"])

        # Define the unit vectors of the rotating cartesian coordinate system (x', y', z')
        self.e_x_prime = Constant([cos(self.L / self.R * 0.5), sin(self.L / self.R * 0.5), 0])
        self.e_y_prime = Constant([-sin(self.L / self.R * 0.5), cos(self.L / self.R * 0.5), 0])
        self.e_z_prime = Constant([0, 0, 1])

        # Define the unit vectors of the rotating cylindrical coordinate system
        self.e_r_prime = as_vector([self.x[0] / sqrt(self.x[0]**2 + self.x[1]**2), self.x[1] / sqrt(self.x[0]**2 + self.x[1]**2), 0])
        self.e_theta_prime = as_vector([- self.x[1]/ sqrt(self.x[0]**2 + self.x[1]**2), self.x[0] / sqrt(self.x[0]**2 + self.x[1]**2), 0])

        # We solve the homogenous part of the Stokes Problem and assemble the matrix
        self.V = VectorFunctionSpace(self.mesh3d, "CG", 2)
        self.Q = FunctionSpace(self.mesh3d, "CG", 1)
        self.W_mixed = self.V * self.Q

        v_0_hom, q_0 = TrialFunctions(self.W_mixed)
        v, q = TestFunctions(self.W_mixed)

        a_form = 2 * inner(sym(grad(v_0_hom)), sym(grad(v))) * dx - q_0 * div(v) * dx + q * div(v_0_hom) * dx

        self._bcs_hom = [
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["walls"]),
            DirichletBC(self.W_mixed.sub(0), Constant((0.0, 0.0, 0.0)), self.tags["particle"]),
        ]

        A = assemble(a_form, bcs=self._bcs_hom)

        nullspace = MixedVectorSpaceBasis(self.W_mixed, [self.W_mixed.sub(0), VectorSpaceBasis(constant=True, comm=self.W_mixed.comm)])

        self.solver = LinearSolver(
            A,
            nullspace=nullspace,
            solver_parameters={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_24": 1,
                "mat_mumps_icntl_25": 0,
            },
        )

        self.v_0_bc = Function(self.V)


    def Stokes_solver_3d(self, particle_bcs):

        v_test, q_test = TestFunctions(self.W_mixed)
        # v_bc is the function capturing the bcs while having arbitrary values on the interior of the domain
        self.v_0_bc.assign(0.0)
        self.v_0_bc.interpolate(particle_bcs)

        DirichletBC(self.V, Constant((0.0, 0.0, 0.0)), self.tags["walls"]).apply(self.v_0_bc)
        DirichletBC(self.V, particle_bcs, self.tags["particle"]).apply(self.v_0_bc)

        L_bcs = - 2 * inner(sym(grad(self.v_0_bc)), sym(grad(v_test))) * dx - q_test * div(self.v_0_bc) * dx

        b = assemble(L_bcs, tensor=Cofunction(self.W_mixed.dual()), bcs=self._bcs_hom)

        w = Function(self.W_mixed)
        self.solver.solve(w, b)
        v_0_hom, q_0 = w.subfunctions
        v_0 = Function(self.V)
        v_0.assign(v_0_hom)
        v_0 += self.v_0_bc

        return v_0, q_0


    def F_0(self, v_0, q_0, mesh3d):

        # Paper convention: n points outward from the particle (into the fluid).
        # Firedrake's FacetNormal on the fluid mesh points outward from the fluid,
        # i.e., into the particle -- opposite of the paper convention.
        # We therefore use -n below to recover the paper n.
        n = FacetNormal(mesh3d)
        sigma = -q_0 * Identity(3) + (grad(v_0) + grad(v_0).T)
        traction = dot(-n, sigma)
        components = [assemble(traction[i] * ds(self.tags["particle"], degree=8)) for i in range(3)]
        F_0 = np.array([float(c) for c in components])
        return F_0


    def T_0(self, v_0_a, q_0_a, mesh3d):

        n = FacetNormal(mesh3d)
        sigma = -q_0_a * Identity(3) + (grad(v_0_a) + grad(v_0_a).T)
        traction = dot(-n, sigma)
        moment_density = cross((self.x - self.x_p), traction)
        components = [assemble(moment_density[i] * ds(self.tags["particle"], degree=8)) for i in range(3)]
        T_0 = np.array([float(c) for c in components])
        return T_0


    def compute_F_1(self, v_0_a, v_0_s, u_x, u_z, u_bar_a, u_bar_s, Theta_np,
                    return_terms=False):

        x_p_np = np.array(self.x_p.values())
        e_x_prime_np = np.array(self.e_x_prime.values())
        e_z_np = np.array([0, 0, 1])
        Theta = Constant(Theta_np)
        n = FacetNormal(self.mesh3d)

        centrifugal_term = - (4*np.pi)/3 * self.a**3 * (Theta_np**2) * np.cross(e_z_np, np.cross(e_z_np, x_p_np))

        # Refactor the background inertial volume integral into a surface integral using the divergence theorem and incompressibility
        inertial_integrand = dot(u_bar_a, -n) * u_bar_a + (dot(u_bar_s, -n) * u_bar_a + dot(u_bar_a, -n) * u_bar_s) + dot(u_bar_s, -n) * u_bar_s

        inertial_integral = [assemble(inertial_integrand[i] * ds(self.tags["particle"], degree=6)) for i in range(3)]

        # With the assemble command UFL becomes a python list.
        # We want numpy arrays to be able to still apply computational operations later like np.cross
        inertial_term = np.array(inertial_integral)

        fluid_stress_right_inner = (
                cross(Theta * self.e_z_prime, v_0_a)
                      + dot(grad(u_bar_a), v_0_a)
                      + dot(grad(v_0_a), v_0_a + u_bar_a - cross(Theta * self.e_z_prime, self.x))
                + dot(grad(u_bar_s), v_0_s) + dot(grad(v_0_s), v_0_s + u_bar_s)
                + cross(Theta * self.e_z_prime, v_0_s)
                        - dot(grad(v_0_s), cross(Theta * self.e_z_prime, self.x))
                        + dot(grad(u_bar_s), v_0_a)
                        + dot(grad(u_bar_a), v_0_s)
                        + dot(grad(v_0_s), v_0_a + u_bar_a)
                        + dot(grad(v_0_a), v_0_s + u_bar_s)
        )

        fluid_stress_x = - dot(u_x, fluid_stress_right_inner)
        fluid_stress_z = - dot(u_z, fluid_stress_right_inner)

        fluid_stress_integral_x = assemble(fluid_stress_x * dx(degree=6))
        fluid_stress_integral_z = assemble(fluid_stress_z * dx(degree=6))

        fluid_stress_term = fluid_stress_integral_x * e_x_prime_np + fluid_stress_integral_z * e_z_np

        F_1 = fluid_stress_term + inertial_term + centrifugal_term

        if return_terms:
            return F_1, fluid_stress_term, inertial_term, centrifugal_term
        return F_1


    def F_p(self, return_components=False):

        u_bar_a = dot(self.u_bar, self.e_theta_prime) * self.e_theta_prime
        u_bar_s = (dot(self.u_bar, self.e_r_prime) * self.e_r_prime + dot(self.u_bar, self.e_z_prime) * self.e_z_prime)

        bcs_Theta = cross(self.e_z_prime, self.x)
        bcs_Omega_p_x = cross(self.e_x_prime, self.x - self.x_p)
        bcs_Omega_p_y = cross(self.e_y_prime, self.x - self.x_p)
        bcs_Omega_p_z = cross(self.e_z_prime, self.x - self.x_p)
        bcs_bg = -u_bar_a

        v_0_a_Theta, q_0_a_Theta = self.Stokes_solver_3d(bcs_Theta)
        v_0_a_Omega_p_x, q_0_a_Omega_p_x = self.Stokes_solver_3d(bcs_Omega_p_x)
        v_0_a_Omega_p_y, q_0_a_Omega_p_y = self.Stokes_solver_3d(bcs_Omega_p_y)
        v_0_a_Omega_p_z, q_0_a_Omega_p_z = self.Stokes_solver_3d(bcs_Omega_p_z)
        v_0_a_bg, q_0_a_bg = self.Stokes_solver_3d(bcs_bg)

        v_0_s, q_0_s = self.Stokes_solver_3d(-u_bar_s)

        F_0_a_Theta = self.F_0(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        F_0_a_Omega_p_x = self.F_0(v_0_a_Omega_p_x, q_0_a_Omega_p_x, self.mesh3d)
        F_0_a_Omega_p_y = self.F_0(v_0_a_Omega_p_y, q_0_a_Omega_p_y, self.mesh3d)
        F_0_a_Omega_p_z = self.F_0(v_0_a_Omega_p_z, q_0_a_Omega_p_z, self.mesh3d)
        F_0_a_Omega_bg = self.F_0(v_0_a_bg, q_0_a_bg, self.mesh3d)
        F_0_s = self.F_0(v_0_s, q_0_s, self.mesh3d)

        T_0_a_Theta = self.T_0(v_0_a_Theta, q_0_a_Theta, self.mesh3d)
        T_0_a_Omega_p_x = self.T_0(v_0_a_Omega_p_x, q_0_a_Omega_p_x, self.mesh3d)
        T_0_a_Omega_p_y = self.T_0(v_0_a_Omega_p_y, q_0_a_Omega_p_y, self.mesh3d)
        T_0_a_Omega_p_z = self.T_0(v_0_a_Omega_p_z, q_0_a_Omega_p_z, self.mesh3d)
        T_0_a_Omega_bg = self.T_0(v_0_a_bg, q_0_a_bg, self.mesh3d)
        T_0_s = self.T_0(v_0_s, q_0_s, self.mesh3d)

        e_x_np = np.array(self.e_x_prime.values())
        e_y_np = np.array(self.e_y_prime.values())
        e_z_np = np.array(self.e_z_prime.values())

        A = np.array([
            [np.dot(e_y_np, F_0_a_Theta), np.dot(e_y_np, F_0_a_Omega_p_z),
             np.dot(e_y_np, F_0_a_Omega_p_x), np.dot(e_y_np, F_0_a_Omega_p_y)],

            [np.dot(e_x_np, T_0_a_Theta), np.dot(e_x_np, T_0_a_Omega_p_z),
             np.dot(e_x_np, T_0_a_Omega_p_x), np.dot(e_x_np, T_0_a_Omega_p_y)],

            [np.dot(e_y_np, T_0_a_Theta), np.dot(e_y_np, T_0_a_Omega_p_z),
             np.dot(e_y_np, T_0_a_Omega_p_x), np.dot(e_y_np, T_0_a_Omega_p_y)],

            [np.dot(e_z_np, T_0_a_Theta), np.dot(e_z_np, T_0_a_Omega_p_z),
             np.dot(e_z_np, T_0_a_Omega_p_x), np.dot(e_z_np, T_0_a_Omega_p_y)]
        ])

        F_0_bg = F_0_a_Omega_bg + F_0_s
        T_0_bg = T_0_a_Omega_bg + T_0_s

        b = -np.array([
            np.dot(e_y_np, F_0_bg),
            np.dot(e_x_np, T_0_bg),
            np.dot(e_y_np, T_0_bg),
            np.dot(e_z_np, T_0_bg),
        ])

        Theta_and_Omega_p = np.linalg.solve(A, b)

        Theta = float(Theta_and_Omega_p[0])
        Omega_p_z = float(Theta_and_Omega_p[1])
        Omega_p_x = float(Theta_and_Omega_p[2])
        Omega_p_y = float(Theta_and_Omega_p[3])

        v_0_a = Function(v_0_a_Theta.function_space())
        v_0_a.interpolate(Constant(Theta) * v_0_a_Theta
                          + Constant(Omega_p_x) * v_0_a_Omega_p_x
                          + Constant(Omega_p_y) * v_0_a_Omega_p_y
                          + Constant(Omega_p_z) * v_0_a_Omega_p_z
                          + v_0_a_bg)

        # Per Harding et al. (2019), §2.6 and eq (2.47): only F_{-1,s} enters the
        # leading-order force. The antisymmetric problem satisfies F_{-1,a} = 0 as a
        # vector by Stokes reversibility of the axial flow. The 4x4 system above only
        # enforces F_{-1,a} * e_y (plus all three torques); the e_x and e_z components
        # vanish in the continuum, so any nonzero value computed here is discretization
        # error and must not be added to the leading-order force.
        F_0_total = F_0_s

        u_hat_x, _ = self.Stokes_solver_3d(self.e_x_prime)
        u_hat_z, _ = self.Stokes_solver_3d(self.e_z_prime)

        F_1, fluid_stress_term, inertial_term, centrifugal_term = self.compute_F_1(
            v_0_a, v_0_s, u_hat_x, u_hat_z, u_bar_a, u_bar_s, Theta,
            return_terms=True)

        F_p_x = np.dot(e_x_np, 1 / self.Re * F_0_total + F_1)
        F_p_z = np.dot(e_z_np, 1 / self.Re * F_0_total + F_1)

        if not return_components:
            return F_p_x, F_p_z

        # Term-by-term projection onto (e_x', e_z') for cross-implementation
        # decomposition.  F_s is the *raw* Stokes-drag force vector (no 1/Re);
        # the leading-order contribution to F_p is (1/Re) * F_s.
        components = {
            "F_s_x": float(np.dot(e_x_np, F_0_total)),
            "F_s_z": float(np.dot(e_z_np, F_0_total)),
            "inertial_x": float(np.dot(e_x_np, inertial_term)),
            "inertial_z": float(np.dot(e_z_np, inertial_term)),
            "centrifugal_x": float(np.dot(e_x_np, centrifugal_term)),
            "centrifugal_z": float(np.dot(e_z_np, centrifugal_term)),
            "fluid_stress_x": float(np.dot(e_x_np, fluid_stress_term)),
            "fluid_stress_z": float(np.dot(e_z_np, fluid_stress_term)),
            "Theta": float(Theta),
        }
        return F_p_x, F_p_z, components


def _make_pf(u_bar_2d, p_bar_2d, G_val, x_off, z_off, pmh_rel, gmh_rel):
    """Build a perturbed_flow instance at a given particle offset and mesh
    resolution, reusing a precomputed 2D background flow. Helper for the
    verification block below."""
    mesh3d, tags = make_curved_channel_section_with_spherical_hole(
        R, H, W, L_rel * max(H, W), a,
        pmh_rel * a, gmh_rel * min(H, W), x_off=x_off, z_off=z_off)
    u3d, p3d = build_3d_background_flow(R, H, W, G_val, mesh3d, tags, u_bar_2d, p_bar_2d)
    pf = perturbed_flow(R, H, W, L_rel * max(H, W), a, Re, mesh3d, tags, u3d, p3d)
    return pf


if __name__ == "__main__":

    x_off = 0.0
    z_off = 0.0

    bg = background_flow(R, H, W, Re)
    G_val, U_m, u_bar, p_bar_tilde = bg.solve_2D_background_flow()

    mesh3d, tags = make_curved_channel_section_with_spherical_hole(R, H, W, L_rel * max(H, W), a,
                                                                   particle_maxh_rel * a, global_maxh_rel * min(H, W),
                                                                   x_off=x_off, z_off=z_off)

    u_bar_3d, p_bar_3d = build_3d_background_flow(R, H, W, G_val, mesh3d, tags, u_bar, p_bar_tilde)

    pf = perturbed_flow(R, H, W, L_rel * max(H, W), a, Re, mesh3d, tags, u_bar_3d, p_bar_3d)

    F_p_x, F_p_z = pf.F_p()

    print(f"F_p_x = {float(F_p_x)}")
    print(f"F_p_z = {float(F_p_z)}")

    # ======================= Verification =======================
    # NOTE: each block rebuilds the 3D mesh and runs the full 8-Stokes-solve
    # pipeline, so this is heavy -- run the blocks selectively as needed.

    # (1) Self-convergence of F_p at a fixed off-centre position (Table tab:fp_conv)
    x_v, z_v = 0.2, 0.2
    refine_configs = [(0.20, 0.10), (0.14, 0.07), (0.10, 0.05)]  # (particle, bulk) maxh_rel, refining
    print("\nF_p self-convergence (fills Table tab:fp_conv):")
    fp_vals = []
    for pmh, gmh in refine_configs:
        pf_h = _make_pf(u_bar, p_bar_tilde, G_val, x_v, z_v, pmh, gmh)
        ndof = pf_h.V.dim() + pf_h.Q.dim()
        fp = np.array(pf_h.F_p())
        fp_vals.append(fp)
        print(f"  (pmh,gmh)=({pmh},{gmh})  dofs={ndof}  F_p={fp}")
    ref = fp_vals[-1]
    for i in range(len(fp_vals) - 1):
        print(f"  ||F_p[{i}] - F_p[ref]|| = {np.linalg.norm(fp_vals[i] - ref):.3e}")

    # (2) z -> -z reflection parity at a symmetric off-centre position
    print("\nReflection parity check:")
    pf_p = _make_pf(u_bar, p_bar_tilde, G_val, x_v, +z_v, particle_maxh_rel, global_maxh_rel)
    pf_m = _make_pf(u_bar, p_bar_tilde, G_val, x_v, -z_v, particle_maxh_rel, global_maxh_rel)
    Fxp, Fzp = pf_p.F_p()
    Fxm, Fzm = pf_m.F_p()
    print(f"  |Fx(+z) - Fx(-z)| = {abs(float(Fxp) - float(Fxm)):.3e}  (should ~ 0)")
    print(f"  |Fz(+z) + Fz(-z)| = {abs(float(Fzp) + float(Fzm)):.3e}  (should ~ 0)")

    # (3) Stokes drag coefficients vs the unbounded value 6*pi*a (centred particle)
    print("\nStokes drag check:")
    pf_d = _make_pf(u_bar, p_bar_tilde, G_val, 0.0, 0.0, particle_maxh_rel, global_maxh_rel)
    u_hat_x, q_hat_x = pf_d.Stokes_solver_3d(pf_d.e_x_prime)
    u_hat_z, q_hat_z = pf_d.Stokes_solver_3d(pf_d.e_z_prime)
    ex = np.array(pf_d.e_x_prime.values())
    ez = np.array(pf_d.e_z_prime.values())
    Cx = float(np.dot(ex, pf_d.F_0(u_hat_x, q_hat_x, pf_d.mesh3d)))
    Cz = float(np.dot(ez, pf_d.F_0(u_hat_z, q_hat_z, pf_d.mesh3d)))
    print(f"  Cx={Cx:.4f}  Cz={Cz:.4f}  6*pi*a={6 * np.pi * a:.4f}"
          f"  |Cx|/(6 pi a)={abs(Cx) / (6 * np.pi * a):.3f}  |Cz|/(6 pi a)={abs(Cz) / (6 * np.pi * a):.3f}")

    # (4) Truncation-length convergence: vary the duct length L only, holding the
    # mesh resolution AND the particle position fixed. Because the near-particle
    # sizing field depends only on the distance to the particle, refining L leaves
    # the disturbance resolution untouched and isolates the do-nothing truncation
    # error from the discretisation error that Table tab:fp_conv already covers.
    # (_make_pf hard-wires L_rel, so we build the instance locally here instead.)
    def _make_pf_L(x_off, z_off, L_rel_):
        L_abs = L_rel_ * max(H, W)
        m3d, tg = make_curved_channel_section_with_spherical_hole(
            R, H, W, L_abs, a, particle_maxh_rel * a, global_maxh_rel * min(H, W),
            x_off=x_off, z_off=z_off)
        u3d, p3d = build_3d_background_flow(R, H, W, G_val, m3d, tg, u_bar, p_bar_tilde)
        return perturbed_flow(R, H, W, L_abs, a, Re, m3d, tg, u3d, p3d)

    print("\nTruncation-length convergence (L-study, position fixed at (0.2,0.2)):")
    L_factors = [4, 6, 8]
    fp_L = []
    for L_rel_ in L_factors:
        pf_L = _make_pf_L(x_v, z_v, L_rel_)
        fp = np.array(pf_L.F_p())
        fp_L.append(fp)
        print(f"  L_rel={L_rel_}  L={L_rel_ * max(H, W):.1f}  "
              f"n_cells={pf_L.mesh3d.num_cells()}  F_p={fp}")
    for L_rel_, fp in zip(L_factors[:-1], fp_L[:-1]):
        print(f"  ||F_p(L_rel={L_rel_}) - F_p(L_rel={L_factors[-1]})|| "
              f"= {np.linalg.norm(fp - fp_L[-1]):.3e}")

    # (5) Richardson self-convergence of F_p: a FIXED refinement ratio over >=4
    # levels, reporting CONSECUTIVE Cauchy differences and the observed order
    # p = log(d_k / d_{k+1}) / log(ratio) -- the genuine order estimate Table
    # tab:fp_conv lacks. Run at two positions: one off-centre and one near the
    # wall, where mesh grading and lubrication are most demanding.
    # NB: heavy. 2 points x 4 levels = 8 full 8-Stokes-solve pipelines; the
    # finest level (pmh~0.071) carries ~1.5-2e6 dofs -- watch RAM for the LU.
    # Admissible-x bound here: W/2 - a - eps_rel*a = 1 - 0.1 - 0.02 = 0.88.
    RATIO = np.sqrt(2.0)
    n_levels = 4
    levels = [(0.20 * RATIO ** -k, 0.10 * RATIO ** -k) for k in range(n_levels)]
    points = [(0.2, 0.2), (0.7, 0.2)]
    print("\nRichardson F_p self-convergence (fixed ratio sqrt(2), consecutive diffs):")
    for (xv, zv) in points:
        fps = []
        for pmh, gmh in levels:
            pf_h = _make_pf(u_bar, p_bar_tilde, G_val, xv, zv, pmh, gmh)
            fp = np.array(pf_h.F_p())
            fps.append(fp)
            print(f"  pt=({xv},{zv})  (pmh,gmh)=({pmh:.4f},{gmh:.4f})  "
                  f"dofs={pf_h.V.dim() + pf_h.Q.dim()}  F_p={fp}")
        diffs = [np.linalg.norm(fps[i] - fps[i + 1]) for i in range(len(fps) - 1)]
        print(f"  pt=({xv},{zv})  consecutive diffs: "
              f"{['%.3e' % d for d in diffs]}")
        for i in range(len(diffs) - 1):
            if diffs[i + 1] > 0:
                p_obs = np.log(diffs[i] / diffs[i + 1]) / np.log(RATIO)
                print(f"    observed order level {i}->{i + 1}: p = {p_obs:.2f}")


