import numpy as np

from lbm.hydro.collision import collide_bgk
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.hydro.streaming import stream
from lbm.lattices.d2q9 import CS2


def test_shear_wave_viscosity():

    nx = 4
    ny = 64

    tau = 0.8
    nu_theory = CS2 * (tau - 0.5)

    rho0 = 1.0
    u0 = 0.01

    nsteps = 1000
    output_every = 10

    k = 2.0 * np.pi / ny

    y = np.arange(ny)

    rho = rho0 * np.ones((nx, ny))

    u = np.zeros((2, nx, ny))

    u[0] = u0 * np.sin(k * y)[None, :]

    f = equilibrium(rho, u)

    times = []
    amplitudes = []

    for step in range(nsteps + 1):
        if step % output_every == 0:
            _, u = macroscopic(f)

            ux_mean = np.mean(u[0], axis=0)

            amplitude = 2.0 / ny * np.sum(ux_mean * np.sin(k * y))

            times.append(step)
            amplitudes.append(amplitude)

        if step < nsteps:
            f = stream(collide_bgk(f, tau))

    times = np.asarray(times)
    amplitudes = np.asarray(amplitudes)

    slope, _ = np.polyfit(
        times,
        np.log(amplitudes),
        1,
    )

    nu_measured = -slope / k**2

    assert np.isclose(
        nu_measured,
        nu_theory,
        rtol=5.0e-3,
    )
