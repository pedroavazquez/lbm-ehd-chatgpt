import numpy as np
import pytest

from lbm.boundary.bounceback import stream_bounceback_y
from lbm.simulation import LBMSimulation


def test_simulation_initial_state():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    rho, u = sim.macroscopic()

    assert np.allclose(
        rho,
        1.0,
    )

    assert np.allclose(
        u,
        0.0,
    )

    assert sim.step_number == 0


def test_periodic_step_preserves_uniform_equilibrium():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    sim.step()

    rho, u = sim.macroscopic()

    assert np.allclose(
        rho,
        1.0,
    )

    assert np.allclose(
        u,
        0.0,
    )

    assert sim.step_number == 1


def test_bounceback_step_preserves_rest_state():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    sim.step(stream_operator=stream_bounceback_y)

    rho, u = sim.macroscopic()

    assert np.allclose(
        rho,
        1.0,
    )

    assert np.allclose(
        u,
        0.0,
    )

    assert sim.step_number == 1


def test_force_shape_check():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    wrong_force = np.zeros((2, 5, 5))

    with pytest.raises(ValueError):
        sim.set_force(wrong_force)


def test_set_state_without_force():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    rho_expected = 1.2 * np.ones((8, 6))

    u_expected = np.zeros((2, 8, 6))

    u_expected[0] = 0.03

    sim.set_state(
        rho_expected,
        u_expected,
    )

    rho, u = sim.macroscopic()

    assert np.allclose(
        rho,
        rho_expected,
    )

    assert np.allclose(
        u,
        u_expected,
    )


def test_set_state_with_force():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    rho_expected = np.ones((8, 6))

    force = np.zeros((2, 8, 6))

    force[0] = 1.0e-6

    sim.set_force(force)

    u_expected = np.zeros((2, 8, 6))

    sim.set_state(
        rho_expected,
        u_expected,
    )

    rho, u = sim.macroscopic()

    assert np.allclose(
        rho,
        rho_expected,
    )

    assert np.allclose(
        u,
        u_expected,
        atol=1.0e-14,
    )


def test_clear_force():

    sim = LBMSimulation(
        nx=8,
        ny=6,
        tau=0.8,
    )

    force = np.zeros((2, 8, 6))

    force[0] = 1.0e-6

    sim.set_force(force)

    assert sim.force is not None

    sim.clear_force()

    assert sim.force is None
