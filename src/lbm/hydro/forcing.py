import numpy as np

from lbm.lattices.d2q9 import CS2, CS4, C, Q, W


def guo_force(
    rho: np.ndarray,
    u: np.ndarray,
    force: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Compute the Guo forcing term for D2Q9.

    Parameters
    ----------
    rho : ndarray, shape (nx, ny)
        Density field.

    u : ndarray, shape (2, nx, ny)
        Physical velocity field.

    force : ndarray, shape (2, nx, ny)
        Body-force density.

    tau : float
        BGK relaxation time.

    Returns
    -------
    source : ndarray, shape (9, nx, ny)
        Discrete forcing term.
    """

    omega = 1.0 / tau

    source = np.empty(
        (Q, *rho.shape),
        dtype=np.float64,
    )

    cu = np.einsum(
        "ia,axy->ixy",
        C,
        u,
    )

    for i in range(Q):
        ci = C[i]

        # Vector:
        #
        # (c_i - u) / cs^2
        # +
        # (c_i . u) c_i / cs^4

        term = (ci[:, None, None] - u) / CS2 + (
            cu[i][None, :, :] * ci[:, None, None] / CS4
        )

        source[i] = W[i] * (1.0 - 0.5 * omega) * np.sum(term * force, axis=0)

    return source
