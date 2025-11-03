import numpy as np

def Nondimensionalizer(d_1, d_2, R, rho, mu, H, W, Q):

    L0_phys = H

    U0_phys = Q / (H**2)
    T0_phys = H / U0_phys

    M0_phys = rho * (H ** 3)

    Q0_phys = U0_phys * (H ** 2)

    mu0_phys = rho * H * U0_phys

    d1_nd = d_1 / H
    d2_nd = d_2 / H
    Rc_nd = R / H

    H_nd = 1.0

    W_nd = W / H

    Q_nd = Q / Q0_phys

    rho_nd = 1.0

    mu_nd = mu / mu0_phys

    Re = (rho * U0_phys * H) / mu

    De = Re * np.sqrt(H/(2*R))

    return d1_nd, d2_nd, Rc_nd, H_nd, W_nd, Q_nd, rho_nd, mu_nd, Re, De
