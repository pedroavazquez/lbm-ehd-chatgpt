import numpy as np

from lbm.lattices.d2q9 import C


def macroscopic(
    f: np.ndarray,
    force: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover density and physical velocity.

    Parameters
    ----------
    f : ndarray, shape (9, nx, ny)

    force : ndarray, shape (2, nx, ny), optional
        Body-force density.

    Returns
    -------
    rho : ndarray, shape (nx, ny)

    u : ndarray, shape (2, nx, ny)
    """

    rho = np.sum(f, axis=0)

    momentum = np.einsum(
        "ixy,ia->axy",
        f,
        C,
    )

    if force is not None:
        momentum += 0.5 * force

    u = momentum / rho[None, :, :]

    return rho, u
