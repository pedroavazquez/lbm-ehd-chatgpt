import numpy as np

from lbm.hydro.equilibrium import equilibrium
from lbm.lattices.d2q9 import CS2, C, Q


def test_equilibrium_shape():
    nx, ny = 8, 6

    rho = np.ones((nx, ny))
    u = np.zeros((2, nx, ny))

    feq = equilibrium(rho, u)

    assert feq.shape == (Q, nx, ny)


def test_equilibrium_at_rest():
    nx, ny = 4, 3

    rho = 2.0 * np.ones((nx, ny))
    u = np.zeros((2, nx, ny))

    feq = equilibrium(rho, u)

    from lbm.lattices.d2q9 import W

    for i in range(Q):
        assert np.allclose(feq[i], W[i] * rho)


def test_zeroth_moment():
    nx, ny = 7, 5

    rho = 1.0 + 0.1 * np.random.default_rng(1).random((nx, ny))

    u = np.zeros((2, nx, ny))
    u[0] = 0.05
    u[1] = -0.02

    feq = equilibrium(rho, u)

    recovered_rho = np.sum(feq, axis=0)

    assert np.allclose(recovered_rho, rho)


def test_first_moment():
    nx, ny = 7, 5

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))
    u[0] = 0.04
    u[1] = -0.03

    feq = equilibrium(rho, u)

    momentum = np.einsum("ixy,ia->axy", feq, C)

    expected = rho[None, :, :] * u

    assert np.allclose(momentum, expected)


def test_second_moment():
    nx, ny = 6, 4

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))
    u[0] = 0.03
    u[1] = 0.02

    feq = equilibrium(rho, u)

    second_moment = np.einsum(
        "ixy,ia,ib->abxy",
        feq,
        C,
        C,
    )

    expected = np.empty_like(second_moment)

    identity = np.eye(2)

    for a in range(2):
        for b in range(2):
            expected[a, b] = rho * CS2 * identity[a, b] + rho * u[a] * u[b]

    assert np.allclose(second_moment, expected)
