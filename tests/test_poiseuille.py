import numpy as np
import pytest

from lbm.boundary.bounceback import stream_bounceback_y
from lbm.hydro.collision import collide_bgk_forced
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.lattices.d2q9 import CS2


@pytest.mark.slow
def test_poiseuille_profile():

    nx = 4
    ny = 32

    tau = 0.8

    rho0 = 1.0
    acceleration = 1.0e-6

    nu = CS2 * (tau - 0.5)

    rho = rho0 * np.ones((nx, ny))

    force = np.zeros((2, nx, ny))

    force[0] = rho0 * acceleration

    # Initial physical velocity = 0.
    u_distribution = -0.5 * force / rho[None, :, :]

    f = equilibrium(
        rho,
        u_distribution,
    )

    nsteps = 12000

    for _ in range(nsteps):
        f = stream_bounceback_y(
            collide_bgk_forced(
                f,
                force,
                tau,
            )
        )

    _, u = macroscopic(
        f,
        force=force,
    )

    ux = np.mean(
        u[0],
        axis=0,
    )

    # Analytical coordinates.
    H = float(ny)

    y = np.arange(ny) + 0.5

    ux_exact = acceleration / (2.0 * nu) * y * (H - y)

    relative_l2_error = np.sqrt(np.sum((ux - ux_exact) ** 2) / np.sum(ux_exact**2))

    # Verify the parabolic velocity profile.
    assert relative_l2_error < 2.0e-3

    # No transverse flow should appear.
    assert np.max(np.abs(u[1])) < 1.0e-12

    # The profile must be symmetric about the
    # channel centreline.
    assert np.allclose(
        ux,
        ux[::-1],
        atol=1.0e-12,
    )
