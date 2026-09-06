import numpy as np
import pytest

from lbm.boundary.bounceback import stream_bounceback_y
from lbm.hydro.collision import collide_bgk_forced
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.lattices.d2q9 import CS2


def poiseuille_error(
    ny: int,
    tau: float = 0.8,
    acceleration: float = 1.0e-6,
) -> float:
    """
    Compute the relative L2 error for plane Poiseuille flow.

    Parameters
    ----------
    ny : int
        Number of fluid lattice nodes across the channel.

    tau : float
        BGK relaxation time.

    acceleration : float
        Constant body acceleration in the x direction.

    Returns
    -------
    error : float
        Relative L2 error of the numerical velocity profile.
    """

    nx = 4
    rho0 = 1.0

    nu = CS2 * (tau - 0.5)

    rho = rho0 * np.ones((nx, ny))

    force = np.zeros((2, nx, ny))
    force[0] = rho0 * acceleration

    # --------------------------------------------------------
    # Initialize with zero physical velocity.
    #
    # rho u = sum_i f_i c_i + F/2
    # --------------------------------------------------------

    u_distribution = -0.5 * force / rho[None, :, :]

    f = equilibrium(
        rho,
        u_distribution,
    )

    # --------------------------------------------------------
    # Estimate a suitable simulation time.
    #
    # The viscous relaxation time scales approximately as
    #
    #     H^2 / nu
    #
    # where H = ny in lattice units.
    # --------------------------------------------------------

    H = float(ny)

    viscous_time = H**2 / nu

    nsteps = int(2.0 * viscous_time)

    # --------------------------------------------------------
    # Time integration
    # --------------------------------------------------------

    for _ in range(nsteps):
        f = stream_bounceback_y(
            collide_bgk_forced(
                f,
                force,
                tau,
            )
        )

    # --------------------------------------------------------
    # Numerical velocity profile
    # --------------------------------------------------------

    _, u = macroscopic(
        f,
        force=force,
    )

    ux = np.mean(
        u[0],
        axis=0,
    )

    # --------------------------------------------------------
    # Analytical solution
    #
    # Halfway bounce-back places the physical walls at
    #
    #     y = 0
    #     y = H
    #
    # while fluid nodes lie at
    #
    #     y_j = j + 1/2.
    # --------------------------------------------------------

    y = np.arange(ny) + 0.5

    ux_exact = acceleration / (2.0 * nu) * y * (H - y)

    # --------------------------------------------------------
    # Relative L2 error
    # --------------------------------------------------------

    error = np.sqrt(np.sum((ux - ux_exact) ** 2) / np.sum(ux_exact**2))

    return error


@pytest.mark.slow
def test_poiseuille_grid_convergence():
    """
    Verify approximately second-order convergence of plane
    Poiseuille flow with halfway bounce-back walls.
    """

    tau = 0.8

    resolutions = np.array([8, 16, 32, 64])

    errors = []

    for ny in resolutions:
        error = poiseuille_error(
            int(ny),
            tau=tau,
        )

        errors.append(error)

    errors = np.asarray(errors)

    # --------------------------------------------------------
    # Observed convergence orders
    #
    # Assume
    #
    #     e(h) ~ C h^p
    #
    # and h ~ 1 / ny.
    #
    # Doubling ny therefore halves h:
    #
    #     p = log(e_N / e_2N) / log(2).
    # --------------------------------------------------------

    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)

    # --------------------------------------------------------
    # Print convergence table
    # --------------------------------------------------------

    print()
    print(" ny        rel_error        order")
    print("--------------------------------------")

    for j, ny in enumerate(resolutions):
        if j == 0:
            print(f"{ny:4d}   {errors[j]:.6e}      ---")

        else:
            print(f"{ny:4d}   {errors[j]:.6e}   {orders[j - 1]:.4f}")

    # --------------------------------------------------------
    # Verification criteria
    # --------------------------------------------------------

    # Error should decrease monotonically.
    assert np.all(errors[1:] < errors[:-1])

    # Straight halfway bounce-back should give approximately
    # second-order spatial accuracy.
    assert np.all(orders > 1.7)

    # Finest grid should give a small error.
    assert errors[-1] < 1.0e-3
