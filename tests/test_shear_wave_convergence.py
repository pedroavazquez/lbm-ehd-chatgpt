import numpy as np
import pytest

from lbm.hydro.collision import collide_bgk
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.hydro.streaming import stream
from lbm.lattices.d2q9 import CS2


def measure_viscosity(ny: int, tau: float = 0.8) -> float:
    """
    Measure the effective kinematic viscosity from the decay
    of a transverse shear wave.

    Parameters
    ----------
    ny : int
        Number of lattice nodes in the y direction.

    tau : float
        BGK relaxation time.

    Returns
    -------
    nu_measured : float
        Kinematic viscosity obtained from the exponential decay rate.
    """

    nx = 4
    rho0 = 1.0
    u0 = 0.01

    nu_theory = CS2 * (tau - 0.5)

    k = 2.0 * np.pi / ny

    # Characteristic viscous decay time:
    #
    # A(t) = A0 exp(-nu k^2 t)
    #
    # so t_decay = 1 / (nu k^2).
    decay_time = 1.0 / (nu_theory * k**2)

    # Simulate long enough to obtain a clear exponential decay.
    nsteps = int(1.5 * decay_time)

    # Store about 100 measurements independently of resolution.
    output_every = max(1, nsteps // 100)

    y = np.arange(ny)

    rho = rho0 * np.ones((nx, ny))

    u = np.zeros((2, nx, ny))

    # Initial shear wave:
    #
    # ux(y, 0) = u0 sin(k y)
    # uy(y, 0) = 0
    u[0] = u0 * np.sin(k * y)[None, :]

    # Initialize populations at equilibrium.
    f = equilibrium(rho, u)

    times = []
    amplitudes = []

    sin_ky = np.sin(k * y)

    for step in range(nsteps + 1):
        if step % output_every == 0:
            _, u = macroscopic(f)

            # The exact solution is independent of x.
            ux_mean = np.mean(u[0], axis=0)

            # Fourier projection onto the initial shear mode.
            amplitude = 2.0 / ny * np.sum(ux_mean * sin_ky)

            times.append(step)
            amplitudes.append(amplitude)

        if step < nsteps:
            f = stream(collide_bgk(f, tau))

    times = np.asarray(times, dtype=float)
    amplitudes = np.asarray(amplitudes)

    # Analytical decay:
    #
    # A(t) = A0 exp(-nu k^2 t)
    #
    # therefore
    #
    # log(A) = log(A0) - nu k^2 t.
    slope, _ = np.polyfit(
        times,
        np.log(amplitudes),
        1,
    )

    nu_measured = -slope / k**2

    return nu_measured


@pytest.mark.slow
def test_shear_wave_grid_convergence():
    """
    Verify convergence of the measured viscosity toward

        nu = cs^2 (tau - 1/2)

    as the shear wave becomes increasingly well resolved.
    """

    tau = 0.8

    nu_theory = CS2 * (tau - 0.5)

    resolutions = np.array([16, 32, 64, 128])

    measured_viscosities = []
    errors = []

    for ny in resolutions:
        nu_measured = measure_viscosity(
            int(ny),
            tau,
        )

        measured_viscosities.append(nu_measured)

        relative_error = abs(nu_measured - nu_theory) / nu_theory

        errors.append(relative_error)

    measured_viscosities = np.asarray(measured_viscosities)

    errors = np.asarray(errors)

    # --------------------------------------------------------
    # Observed convergence order
    #
    # Assume
    #
    #     e(h) ~ C h^p
    #
    # Since h ~ 1 / ny, doubling ny halves h.
    #
    # Therefore
    #
    #     p = log(e_coarse / e_fine) / log(2)
    # --------------------------------------------------------

    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)

    # --------------------------------------------------------
    # Print convergence table
    # --------------------------------------------------------

    print()
    print(" ny      nu_measured       rel_error       order")
    print("------------------------------------------------------")

    for j, ny in enumerate(resolutions):
        if j == 0:
            print(
                f"{ny:4d}   {measured_viscosities[j]:.10f}   {errors[j]:.6e}      ---"
            )
        else:
            print(
                f"{ny:4d}   "
                f"{measured_viscosities[j]:.10f}   "
                f"{errors[j]:.6e}   "
                f"{orders[j - 1]:.4f}"
            )

    print()
    print(f"Theoretical viscosity: {nu_theory:.10f}")

    # --------------------------------------------------------
    # Verification criteria
    # --------------------------------------------------------

    # Error must decrease as the resolution increases.
    assert np.all(errors[1:] < errors[:-1])

    # We expect approximately second-order convergence.
    #
    # A modest lower bound is used because the coarsest
    # resolution may not yet be fully in the asymptotic regime.
    assert np.all(orders > 1.7)

    # The finest-grid result must be close to the
    # hydrodynamic-limit viscosity.
    assert errors[-1] < 2.0e-3
