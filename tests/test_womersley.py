import numpy as np
import pytest
from scipy.special import jv

from lbm.axisymmetric_simulation import (
    AxisymmetricLBMSimulation,
)
from lbm.boundary.axisymmetric import (
    stream_axisymmetric_pipe,
)
from lbm.lattices.d2q9 import CS2


def analytical_womersley(
    r,
    t,
    radius,
    nu,
    omega,
    g0,
):

    alpha = radius * np.sqrt(omega / nu)

    lambda_factor = (-1.0 + 1.0j) / np.sqrt(2.0)

    argument = lambda_factor * alpha * r / radius

    argument_wall = lambda_factor * alpha

    amplitude = (
        g0
        / (1.0j * omega)
        * (
            1.0
            - jv(
                0,
                argument,
            )
            / jv(
                0,
                argument_wall,
            )
        )
    )

    return np.real(amplitude * np.exp(1.0j * omega * t))


@pytest.mark.slow
def test_womersley_flow():

    nz = 4
    nr = 24

    rho0 = 1.0

    tau = 0.8

    nu = CS2 * (tau - 0.5)

    R = float(nr)

    r = np.arange(nr) + 0.5

    g0 = 1.0e-6

    alpha_target = 4.0

    omega_target = nu * alpha_target**2 / R**2

    period_steps = round(2.0 * np.pi / omega_target)

    omega = 2.0 * np.pi / period_steps

    sim = AxisymmetricLBMSimulation(
        nz=nz,
        nr=nr,
        tau=tau,
        rho0=rho0,
    )

    rho = rho0 * np.ones((nz, nr))

    u = np.zeros((2, nz, nr))

    sim.set_state(
        rho,
        u,
    )

    # Remove transient.
    transient_periods = 6

    total_steps = transient_periods * period_steps + period_steps // 4

    for step in range(total_steps):
        acceleration = g0 * np.cos(omega * step)

        force = np.zeros((2, nz, nr))

        force[0] = rho0 * acceleration

        sim.set_external_force(force)

        sim.step(stream_operator=(stream_axisymmetric_pipe))

    _, u = sim.macroscopic()

    uz = np.mean(
        u[0],
        axis=0,
    )

    time = float(sim.step_number)

    uz_exact = analytical_womersley(
        r=r,
        t=time,
        radius=R,
        nu=nu,
        omega=omega,
        g0=g0,
    )

    relative_error = np.sqrt(np.sum((uz - uz_exact) ** 2) / np.sum(uz_exact**2))

    print()
    print(f"Womersley relative error = {relative_error:.6e}")

    assert relative_error < 3.0e-2
