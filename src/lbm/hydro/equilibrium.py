import numpy as np

from lbm.lattices.d2q9 import CS2, CS4, C, Q, W


def equilibrium(rho: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Compute the D2Q9 equilibrium distribution.

    Parameters
    ----------
    rho : ndarray, shape (nx, ny)
        Density field.

    u : ndarray, shape (2, nx, ny)
        Velocity field.

    Returns
    -------
    feq : ndarray, shape (9, nx, ny)
        Equilibrium populations.
    """

    cu = np.einsum("ia,axy->ixy", C, u)

    u2 = np.einsum("axy,axy->xy", u, u)

    feq = np.empty((Q, *rho.shape), dtype=np.float64)

    for i in range(Q):
        feq[i] = (
            W[i]
            * rho
            * (1.0 + cu[i] / CS2 + cu[i] ** 2 / (2.0 * CS4) - u2 / (2.0 * CS2))
        )

    return feq
