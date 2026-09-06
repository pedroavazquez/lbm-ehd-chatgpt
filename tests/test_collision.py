import numpy as np
import pytest

from lbm.hydro.collision import collide_bgk
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic


def test_equilibrium_is_fixed_point():
    nx, ny = 5, 4

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))
    u[0] = 0.03
    u[1] = -0.01

    f = equilibrium(rho, u)

    f_post = collide_bgk(f, tau=0.8)

    assert np.allclose(f_post, f)


def test_collision_conserves_density():
    nx, ny = 6, 5

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))
    u[0] = 0.02

    feq = equilibrium(rho, u)

    rng = np.random.default_rng(1234)

    perturbation = 1.0e-4 * rng.standard_normal(feq.shape)

    # Remove the zeroth moment of the perturbation.
    perturbation -= np.mean(perturbation, axis=0, keepdims=True)

    f = feq + perturbation

    rho_before, _ = macroscopic(f)

    f_post = collide_bgk(f, tau=0.9)

    rho_after, _ = macroscopic(f_post)

    assert np.allclose(rho_after, rho_before)


def test_collision_conserves_momentum():
    nx, ny = 6, 5

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))
    u[0] = 0.03
    u[1] = -0.02

    feq = equilibrium(rho, u)

    rng = np.random.default_rng(5678)
    perturbation = 1.0e-4 * rng.standard_normal(feq.shape)

    # Construct a perturbation, then explicitly project out
    # its conserved moments.
    rho_p, u_p = macroscopic(feq + perturbation)

    correction = equilibrium(rho_p, u_p) - feq

    f = feq + perturbation - correction

    rho_before, u_before = macroscopic(f)

    f_post = collide_bgk(f, tau=0.8)

    rho_after, u_after = macroscopic(f_post)

    assert np.allclose(rho_after, rho_before)
    assert np.allclose(u_after, u_before)


def test_tau_one_relaxes_exactly_to_equilibrium():
    nx, ny = 4, 3

    rng = np.random.default_rng(42)

    f = 0.1 + rng.random((9, nx, ny))

    rho, u = macroscopic(f)

    feq = equilibrium(rho, u)

    f_post = collide_bgk(f, tau=1.0)

    assert np.allclose(f_post, feq)


def test_invalid_tau():
    f = np.ones((9, 2, 2))

    with pytest.raises(ValueError):
        collide_bgk(f, tau=0.5)
