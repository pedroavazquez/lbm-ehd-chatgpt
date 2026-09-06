import numpy as np

from lbm.lattices.d2q9 import OPPOSITE, C, Q


def stream_bounceback_y(f_post: np.ndarray) -> np.ndarray:
    """
    Stream populations with:

    - periodic boundary conditions in x,
    - halfway bounce-back at the lower and upper y walls.

    Fluid nodes are y = 0, ..., ny - 1 in array coordinates.
    The physical walls lie halfway outside the first and last
    fluid nodes.

    Parameters
    ----------
    f_post : ndarray, shape (9, nx, ny)
        Post-collision populations.

    Returns
    -------
    f_streamed : ndarray, shape (9, nx, ny)
        Populations after streaming and wall bounce-back.
    """

    f_streamed = np.empty_like(f_post)

    # --------------------------------------------------------
    # First perform ordinary streaming.
    #
    # np.roll gives periodicity in x and temporarily also in y.
    # The unwanted y-periodic populations are overwritten below
    # by bounce-back values.
    # --------------------------------------------------------

    for i in range(Q):
        cx, cy = C[i]

        f_streamed[i] = np.roll(
            f_post[i],
            shift=(cx, cy),
            axis=(0, 1),
        )

    ny = f_post.shape[2]

    # --------------------------------------------------------
    # Lower wall
    #
    # Populations pointing out of the fluid:
    #
    #   4 : south
    #   7 : south-west
    #   8 : south-east
    #
    # are reflected into their opposite directions:
    #
    #   2 : north
    #   5 : north-east
    #   6 : north-west
    # --------------------------------------------------------

    for i in (4, 7, 8):
        f_streamed[OPPOSITE[i], :, 0] = f_post[i, :, 0]

    # --------------------------------------------------------
    # Upper wall
    #
    # Outgoing directions:
    #
    #   2 : north
    #   5 : north-east
    #   6 : north-west
    #
    # are reflected into:
    #
    #   4 : south
    #   7 : south-west
    #   8 : south-east
    # --------------------------------------------------------

    for i in (2, 5, 6):
        f_streamed[OPPOSITE[i], :, ny - 1] = f_post[i, :, ny - 1]

    return f_streamed
