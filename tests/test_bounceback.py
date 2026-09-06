import numpy as np

from lbm.boundary.bounceback import stream_bounceback_y
from lbm.lattices.d2q9 import Q


def test_lower_wall_normal_reflection():
    nx, ny = 6, 5

    f = np.zeros((Q, nx, ny))

    # South-moving population at the lower wall.
    f[4, 3, 0] = 1.0

    f_streamed = stream_bounceback_y(f)

    # It must return north at the same node.
    assert f_streamed[2, 3, 0] == 1.0


def test_upper_wall_normal_reflection():
    nx, ny = 6, 5

    f = np.zeros((Q, nx, ny))

    # North-moving population at the upper wall.
    f[2, 3, ny - 1] = 1.0

    f_streamed = stream_bounceback_y(f)

    # It must return south at the same node.
    assert f_streamed[4, 3, ny - 1] == 1.0


def test_lower_wall_diagonal_reflection():
    nx, ny = 6, 5

    f = np.zeros((Q, nx, ny))

    # South-west population.
    f[7, 3, 0] = 1.0

    f_streamed = stream_bounceback_y(f)

    # Reflected as north-east at the same node.
    assert f_streamed[5, 3, 0] == 1.0


def test_upper_wall_diagonal_reflection():
    nx, ny = 6, 5

    f = np.zeros((Q, nx, ny))

    # North-east population.
    f[5, 3, ny - 1] = 1.0

    f_streamed = stream_bounceback_y(f)

    # Reflected as south-west.
    assert f_streamed[7, 3, ny - 1] == 1.0


def test_periodic_x_is_preserved():
    nx, ny = 6, 5

    f = np.zeros((Q, nx, ny))

    # East-moving population at right edge.
    f[1, nx - 1, 2] = 1.0

    f_streamed = stream_bounceback_y(f)

    assert f_streamed[1, 0, 2] == 1.0


def test_bounceback_streaming_preserves_mass():
    nx, ny = 8, 7

    rng = np.random.default_rng(1234)

    f = rng.random((Q, nx, ny))

    mass_before = np.sum(f)

    f_streamed = stream_bounceback_y(f)

    mass_after = np.sum(f_streamed)

    assert np.isclose(
        mass_after,
        mass_before,
    )
