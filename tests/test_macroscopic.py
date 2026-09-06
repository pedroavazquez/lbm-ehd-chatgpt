import numpy as np

from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic


def test_macroscopic_at_rest():
    nx, ny = 5, 4

    rho_expected = 1.3 * np.ones((nx, ny))
    u_expected = np.zeros((2, nx, ny))

    f = equilibrium(rho_expected, u_expected)

    rho, u = macroscopic(f)

    assert np.allclose(rho, rho_expected)
    assert np.allclose(u, u_expected)


def test_macroscopic_uniform_velocity():
    nx, ny = 6, 7

    rho_expected = np.ones((nx, ny))

    u_expected = np.zeros((2, nx, ny))
    u_expected[0] = 0.04
    u_expected[1] = -0.02

    f = equilibrium(rho_expected, u_expected)

    rho, u = macroscopic(f)

    assert np.allclose(rho, rho_expected)
    assert np.allclose(u, u_expected)


def test_macroscopic_spatially_varying_fields():
    nx, ny = 8, 6

    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]

    rho_expected = 1.0 + 0.01 * x + 0.02 * y

    u_expected = np.zeros((2, nx, ny))
    u_expected[0] = 0.03 * np.sin(2.0 * np.pi * y / ny)
    u_expected[1] = 0.02 * np.cos(2.0 * np.pi * x / nx)

    f = equilibrium(rho_expected, u_expected)

    rho, u = macroscopic(f)

    assert np.allclose(rho, rho_expected)
    assert np.allclose(u, u_expected)
