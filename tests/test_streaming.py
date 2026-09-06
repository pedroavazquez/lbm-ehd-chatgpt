import numpy as np

from lbm.hydro.streaming import stream
from lbm.lattices.d2q9 import C, Q


def test_rest_population_does_not_move():
    nx, ny = 5, 4

    f = np.zeros((Q, nx, ny))

    f[0, 2, 1] = 1.0

    f_streamed = stream(f)

    assert np.array_equal(f_streamed[0], f[0])


def test_single_population_moves_east():
    nx, ny = 5, 4

    f = np.zeros((Q, nx, ny))

    # Population i = 1 has c = (1, 0)
    f[1, 2, 1] = 1.0

    f_streamed = stream(f)

    assert f_streamed[1, 3, 1] == 1.0
    assert np.sum(f_streamed[1]) == 1.0


def test_single_population_moves_north():
    nx, ny = 5, 4

    f = np.zeros((Q, nx, ny))

    # Population i = 2 has c = (0, 1)
    f[2, 2, 1] = 1.0

    f_streamed = stream(f)

    assert f_streamed[2, 2, 2] == 1.0
    assert np.sum(f_streamed[2]) == 1.0


def test_diagonal_population_moves_correctly():
    nx, ny = 5, 4

    f = np.zeros((Q, nx, ny))

    # Population i = 5 has c = (1, 1)
    f[5, 1, 1] = 1.0

    f_streamed = stream(f)

    assert f_streamed[5, 2, 2] == 1.0
    assert np.sum(f_streamed[5]) == 1.0


def test_periodicity_in_x():
    nx, ny = 5, 4

    f = np.zeros((Q, nx, ny))

    # East-moving population at the right boundary
    f[1, nx - 1, 2] = 1.0

    f_streamed = stream(f)

    # It re-enters at x = 0
    assert f_streamed[1, 0, 2] == 1.0


def test_periodicity_in_y():
    nx, ny = 5, 4

    f = np.zeros((Q, nx, ny))

    # North-moving population at the top boundary
    f[2, 3, ny - 1] = 1.0

    f_streamed = stream(f)

    # It re-enters at y = 0
    assert f_streamed[2, 3, 0] == 1.0


def test_streaming_preserves_total_mass():
    nx, ny = 7, 6

    rng = np.random.default_rng(1234)

    f = rng.random((Q, nx, ny))

    mass_before = np.sum(f)

    f_streamed = stream(f)

    mass_after = np.sum(f_streamed)

    assert np.isclose(mass_after, mass_before)


def test_all_directions():
    nx, ny = 7, 7

    f = np.zeros((Q, nx, ny))

    x0, y0 = 3, 3

    for i in range(Q):
        f[i, x0, y0] = i + 1.0

    f_streamed = stream(f)

    for i in range(Q):
        cx, cy = C[i]

        x1 = (x0 + cx) % nx
        y1 = (y0 + cy) % ny

        assert f_streamed[i, x1, y1] == i + 1.0
