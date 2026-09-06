import numpy as np

from lbm.lattices.d2q9 import C, Q


def stream(f_post: np.ndarray) -> np.ndarray:
    """
    Stream post-collision populations to neighboring lattice nodes.

    Periodic boundary conditions are applied in both x and y directions.

    Parameters
    ----------
    f_post : ndarray, shape (9, nx, ny)
        Post-collision distribution functions.

    Returns
    -------
    f_streamed : ndarray, shape (9, nx, ny)
        Distribution functions after streaming.
    """

    f_streamed = np.empty_like(f_post)

    for i in range(Q):
        cx, cy = C[i]

        f_streamed[i] = np.roll(
            f_post[i],
            shift=(cx, cy),
            axis=(0, 1),
        )

    return f_streamed
