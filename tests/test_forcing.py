import numpy as np

from lbm.hydro.forcing import guo_force
from lbm.lattices.d2q9 import C


def test_guo_force_has_zero_mass_moment():

    nx, ny = 5, 4

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))

    force = np.zeros((2, nx, ny))
    force[0] = 1.0e-5

    tau = 0.8

    source = guo_force(
        rho,
        u,
        force,
        tau,
    )

    zeroth_moment = np.sum(
        source,
        axis=0,
    )

    assert np.allclose(
        zeroth_moment,
        0.0,
        atol=1.0e-15,
    )


def test_guo_force_first_moment():

    nx, ny = 5, 4

    rho = np.ones((nx, ny))

    u = np.zeros((2, nx, ny))

    force = np.zeros((2, nx, ny))
    force[0] = 2.0e-5
    force[1] = -1.0e-5

    tau = 0.8

    source = guo_force(
        rho,
        u,
        force,
        tau,
    )

    first_moment = np.einsum(
        "ixy,ia->axy",
        source,
        C,
    )

    expected = (1.0 - 1.0 / (2.0 * tau)) * force

    assert np.allclose(
        first_moment,
        expected,
    )


from lbm.hydro.collision import collide_bgk_forced
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.hydro.streaming import stream


def test_uniform_force_produces_uniform_acceleration():

    nx, ny = 8, 8

    rho0 = 1.0
    tau = 0.8

    acceleration = 1.0e-6

    rho = rho0 * np.ones((nx, ny))

    force = np.zeros((2, nx, ny))

    force[0] = rho0 * acceleration

    # --------------------------------------------------------
    # Initial physical velocity is zero.
    #
    # With forcing:
    #
    # rho u = sum_i f_i c_i + F / 2
    #
    # Therefore, to obtain physical u = 0 initially,
    # the populations must carry momentum -F / 2.
    # --------------------------------------------------------

    u_distribution = np.zeros((2, nx, ny))

    u_distribution -= 0.5 * force / rho[None, :, :]

    f = equilibrium(
        rho,
        u_distribution,
    )

    # Verify that the physical velocity really is zero.
    _, u_initial = macroscopic(
        f,
        force=force,
    )

    assert np.allclose(
        u_initial,
        0.0,
        atol=1.0e-14,
    )

    # --------------------------------------------------------
    # Time integration
    # --------------------------------------------------------

    nsteps = 100

    for _ in range(nsteps):
        f = collide_bgk_forced(
            f,
            force,
            tau,
        )

        f = stream(f)

    _, u = macroscopic(
        f,
        force=force,
    )

    # Exact solution:
    #
    # du/dt = a
    #
    # u(t) = a t
    #
    # because u(0) = 0.
    expected_velocity = acceleration * nsteps

    assert np.allclose(
        u[0],
        expected_velocity,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    assert np.allclose(
        u[1],
        0.0,
        atol=1.0e-12,
    )
