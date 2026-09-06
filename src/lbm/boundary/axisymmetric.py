import numpy as np

from lbm.lattices.d2q9 import C, Q


def stream_axisymmetric_pipe(
    f_post: np.ndarray,
) -> np.ndarray:
    """
    Streaming for an axisymmetric straight circular pipe.

    Coordinates
    -----------
    axis 0 -> z
    axis 1 -> r

    Boundary conditions
    -------------------
    z : periodic

    r = 0 :
        symmetry axis (specular reflection)

    r = R :
        stationary no-slip wall
        (halfway bounce-back)

    Parameters
    ----------
    f_post : ndarray, shape (9, nz, nr)
        Post-collision populations.

    Returns
    -------
    f_streamed : ndarray, shape (9, nz, nr)
    """

    f_streamed = np.empty_like(f_post)

    # --------------------------------------------------------
    # Ordinary streaming.
    #
    # np.roll temporarily makes both directions periodic.
    # Radial boundary populations are overwritten below.
    # --------------------------------------------------------

    for i in range(Q):
        cz, cr = C[i]

        f_streamed[i] = np.roll(
            f_post[i],
            shift=(cz, cr),
            axis=(0, 1),
        )

    nr = f_post.shape[2]

    # ========================================================
    # AXIS r = 0
    #
    # Specular reflection:
    #
    # radial component reverses,
    # axial component remains unchanged.
    #
    # 4 ( 0,-1) -> 2 ( 0,+1)
    # 7 (-1,-1) -> 6 (-1,+1)
    # 8 (+1,-1) -> 5 (+1,+1)
    # ========================================================

    f_streamed[2, :, 0] = f_post[4, :, 0]

    f_streamed[6, :, 0] = f_post[7, :, 0]

    f_streamed[5, :, 0] = f_post[8, :, 0]

    # ========================================================
    # OUTER PIPE WALL r = R
    #
    # No-slip halfway bounce-back.
    #
    # 2 -> 4
    # 5 -> 7
    # 6 -> 8
    #
    # Here the complete velocity reverses.
    # ========================================================

    f_streamed[4, :, nr - 1] = f_post[2, :, nr - 1]

    f_streamed[7, :, nr - 1] = f_post[5, :, nr - 1]

    f_streamed[8, :, nr - 1] = f_post[6, :, nr - 1]

    return f_streamed
