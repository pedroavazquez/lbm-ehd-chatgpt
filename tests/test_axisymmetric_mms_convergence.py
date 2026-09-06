import numpy as np
import pytest

from lbm.axisymmetric_simulation import AxisymmetricLBMSimulation
from lbm.boundary.axisymmetric import stream_axisymmetric_pipe
from lbm.units import LatticeScaling
from lbm.utils.convergence import relative_change
from lbm.verification.axisymmetric_mms import (
    manufactured_acceleration,
    manufactured_velocity,
)


def run_axisymmetric_mms(
    nr: int,
) -> tuple[float, float, float, float, int, float]:
    """
    Run the steady manufactured axisymmetric problem.

    The physical problem is held fixed while the lattice is refined.

    Returns
    -------
    error_z : float
        Axisymmetrically weighted relative L2 error in u_z.

    error_r : float
        Axisymmetrically weighted relative L2 error in u_r.

    error_z_unweighted : float
        Ordinary discrete relative L2 error in u_z.

    error_r_unweighted : float
        Ordinary discrete relative L2 error in u_r.

    max_error_index : int
        Radial index where the RMS radial-velocity error is largest.

    max_profile_error : float
        Maximum RMS radial-velocity error over the radial coordinate.
    """

    # ========================================================
    # Fixed physical problem
    # ========================================================

    radius_phys = 1.0
    length_phys = 2.0

    nu_phys = 1.0e-2
    amplitude_phys = 1.0e-2

    rho0 = 1.0

    tau = 0.8

    # One axial wavelength over the periodic domain.
    k_phys = 2.0 * np.pi / length_phys

    # L = 2R, with identical dx in z and r.
    nz = 2 * nr

    # ========================================================
    # Physical <-> lattice scaling
    # ========================================================

    scaling = LatticeScaling(
        length_phys=radius_phys,
        n_length=nr,
        nu_phys=nu_phys,
        tau=tau,
    )

    dx = scaling.dx

    # Cell-centred physical coordinates.
    z_phys = (
        np.arange(
            nz,
            dtype=float,
        )
        + 0.5
    ) * dx

    r_phys = (
        np.arange(
            nr,
            dtype=float,
        )
        + 0.5
    ) * dx

    # ========================================================
    # Exact physical solution
    # ========================================================

    u_exact_phys = manufactured_velocity(
        z=z_phys,
        r=r_phys,
        amplitude=amplitude_phys,
        radius=radius_phys,
        wavenumber=k_phys,
    )

    acceleration_phys = manufactured_acceleration(
        z=z_phys,
        r=r_phys,
        amplitude=amplitude_phys,
        radius=radius_phys,
        wavenumber=k_phys,
        nu=nu_phys,
    )

    # ========================================================
    # Convert exact fields to lattice units
    # ========================================================

    u_exact_lb = u_exact_phys * scaling.dt / scaling.dx

    acceleration_lb = acceleration_phys * scaling.dt**2 / scaling.dx

    external_force = rho0 * acceleration_lb

    # ========================================================
    # Create simulation
    # ========================================================

    sim = AxisymmetricLBMSimulation(
        nz=nz,
        nr=nr,
        tau=tau,
        rho0=rho0,
    )

    sim.set_external_force(external_force)

    rho_initial = rho0 * np.ones((nz, nr))

    # Start from the exact physical steady solution.
    sim.set_state(
        rho_initial,
        u_exact_lb,
    )

    # ========================================================
    # Steady convergence
    # ========================================================

    check_every = 100

    tolerance = 5.0e-7

    viscous_time_phys = radius_phys**2 / nu_phys

    max_physical_time = 5.0 * viscous_time_phys

    max_steps = int(np.ceil(scaling.time_to_lattice(max_physical_time)))

    previous_u = None
    converged = False
    last_change = np.inf

    for _ in range(max_steps):
        sim.step(stream_operator=(stream_axisymmetric_pipe))

        if sim.step_number % check_every == 0:
            _, u_current = sim.macroscopic()

            if previous_u is not None:
                last_change = relative_change(
                    u_current,
                    previous_u,
                )

                if last_change < tolerance:
                    converged = True
                    break

            previous_u = u_current.copy()

    assert converged, (
        f"MMS simulation did not converge for nr={nr} "
        f"after {max_steps} steps; "
        f"last relative change = {last_change:.3e}"
    )

    # ========================================================
    # Numerical solution
    # ========================================================

    _, u_lb = sim.macroscopic()

    u_phys = u_lb * scaling.dx / scaling.dt

    # ========================================================
    # Axisymmetric weighted relative L2 errors
    #
    # Physical volume element:
    #
    #     dV = 2 pi r dr dz
    #
    # The constant 2 pi cancels in a relative norm.
    # ========================================================

    weights = r_phys[None, :]

    error_z = np.sqrt(
        np.sum(weights * (u_phys[0] - u_exact_phys[0]) ** 2)
        / np.sum(weights * u_exact_phys[0] ** 2)
    )

    error_r = np.sqrt(
        np.sum(weights * (u_phys[1] - u_exact_phys[1]) ** 2)
        / np.sum(weights * u_exact_phys[1] ** 2)
    )

    # ========================================================
    # Ordinary unweighted norms for diagnostics
    # ========================================================

    error_z_unweighted = np.sqrt(
        np.sum((u_phys[0] - u_exact_phys[0]) ** 2) / np.sum(u_exact_phys[0] ** 2)
    )

    error_r_unweighted = np.sqrt(
        np.sum((u_phys[1] - u_exact_phys[1]) ** 2) / np.sum(u_exact_phys[1] ** 2)
    )

    # ========================================================
    # Radial distribution of u_r error
    #
    # For each radial position, compute the RMS error over z.
    # ========================================================

    radial_error_profile = np.sqrt(
        np.mean(
            (u_phys[1] - u_exact_phys[1]) ** 2,
            axis=0,
        )
    )

    max_error_index = int(np.argmax(radial_error_profile))

    max_profile_error = float(np.max(radial_error_profile))

    print()
    print(f"nr = {nr}")

    print(f"  steps                    = {sim.step_number}")

    print(f"  final steady change      = {last_change:.6e}")

    print(f"  weighted error uz        = {error_z:.6e}")

    print(f"  weighted error ur        = {error_r:.6e}")

    print(f"  unweighted error uz      = {error_z_unweighted:.6e}")

    print(f"  unweighted error ur      = {error_r_unweighted:.6e}")

    print(f"  max ur error index       = {max_error_index}")

    print(f"  r/R at max ur error      = {r_phys[max_error_index] / radius_phys:.6f}")

    print(f"  max ur profile RMS error = {max_profile_error:.6e}")

    return (
        error_z,
        error_r,
        error_z_unweighted,
        error_r_unweighted,
        max_error_index,
        max_profile_error,
    )


@pytest.mark.slow
def test_axisymmetric_mms_grid_convergence():
    """
    Verify spatial convergence of the general axisymmetric solver
    for a manufactured flow with both u_z and u_r nonzero.
    """

    resolutions = np.array([8, 16, 32])

    errors_z = []
    errors_r = []

    errors_z_unweighted = []
    errors_r_unweighted = []

    max_error_indices = []
    max_profile_errors = []

    for nr in resolutions:
        (
            error_z,
            error_r,
            error_z_unweighted,
            error_r_unweighted,
            max_error_index,
            max_profile_error,
        ) = run_axisymmetric_mms(int(nr))

        errors_z.append(error_z)

        errors_r.append(error_r)

        errors_z_unweighted.append(error_z_unweighted)

        errors_r_unweighted.append(error_r_unweighted)

        max_error_indices.append(max_error_index)

        max_profile_errors.append(max_profile_error)

    errors_z = np.asarray(errors_z)

    errors_r = np.asarray(errors_r)

    errors_z_unweighted = np.asarray(errors_z_unweighted)

    errors_r_unweighted = np.asarray(errors_r_unweighted)

    max_error_indices = np.asarray(max_error_indices)

    max_profile_errors = np.asarray(max_profile_errors)

    # ========================================================
    # Observed convergence orders
    # ========================================================

    orders_z = np.log(errors_z[:-1] / errors_z[1:]) / np.log(2.0)

    orders_r = np.log(errors_r[:-1] / errors_r[1:]) / np.log(2.0)

    orders_z_unweighted = np.log(
        errors_z_unweighted[:-1] / errors_z_unweighted[1:]
    ) / np.log(2.0)

    orders_r_unweighted = np.log(
        errors_r_unweighted[:-1] / errors_r_unweighted[1:]
    ) / np.log(2.0)

    # ========================================================
    # Weighted convergence table
    # ========================================================

    print()
    print("Axisymmetric weighted L2 errors")

    print("---------------------------------------------------------------")

    print(" nr      error(uz)      order_z      error(ur)      order_r")

    print("---------------------------------------------------------------")

    for j, nr in enumerate(resolutions):
        if j == 0:
            print(
                f"{nr:4d}   {errors_z[j]:.6e}      ---      {errors_r[j]:.6e}      ---"
            )

        else:
            print(
                f"{nr:4d}   "
                f"{errors_z[j]:.6e}   "
                f"{orders_z[j - 1]:7.4f}   "
                f"{errors_r[j]:.6e}   "
                f"{orders_r[j - 1]:7.4f}"
            )

    # ========================================================
    # Unweighted convergence table
    # ========================================================

    print()
    print("Ordinary unweighted L2 errors")

    print("---------------------------------------------------------------")

    print(" nr      error(uz)      order_z      error(ur)      order_r")

    print("---------------------------------------------------------------")

    for j, nr in enumerate(resolutions):
        if j == 0:
            print(
                f"{nr:4d}   "
                f"{errors_z_unweighted[j]:.6e}      ---      "
                f"{errors_r_unweighted[j]:.6e}      ---"
            )

        else:
            print(
                f"{nr:4d}   "
                f"{errors_z_unweighted[j]:.6e}   "
                f"{orders_z_unweighted[j - 1]:7.4f}   "
                f"{errors_r_unweighted[j]:.6e}   "
                f"{orders_r_unweighted[j - 1]:7.4f}"
            )

    # ========================================================
    # Boundary/localization diagnostics
    # ========================================================

    print()
    print("Radial-error localization")

    print("---------------------------------------------")

    print(" nr      max-index      max RMS error")

    print("---------------------------------------------")

    for (
        nr,
        index,
        profile_error,
    ) in zip(
        resolutions,
        max_error_indices,
        max_profile_errors,
    ):
        print(f"{nr:4d}      {index:5d}        {profile_error:.6e}")

    # ========================================================
    # Verification criteria
    # ========================================================

    assert np.all(errors_z[1:] < errors_z[:-1])

    assert np.all(errors_r[1:] < errors_r[:-1])

    # For now retain the axial convergence requirement.
    #
    # Do NOT yet weaken the radial requirement merely to
    # make the test pass: the diagnostic above is intended
    # to tell us why its convergence is slower.
    assert orders_z[-1] > 1.5

    assert orders_r[-1] > 1.5
