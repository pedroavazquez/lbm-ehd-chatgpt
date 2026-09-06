import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv

from lbm.axisymmetric_simulation import (
    AxisymmetricLBMSimulation,
)
from lbm.boundary.axisymmetric import (
    stream_axisymmetric_pipe,
)
from lbm.lattices.d2q9 import CS2

# ============================================================
# Geometry
# ============================================================

nz = 8
nr = 32

rho0 = 1.0

R = float(nr)

r = np.arange(nr) + 0.5


# ============================================================
# Fluid properties
# ============================================================

tau = 0.8

nu = CS2 * (tau - 0.5)


# ============================================================
# Oscillatory forcing
#
# g(t) = g0 cos(omega t)
# ============================================================

g0 = 1.0e-6

womersley = 5.0


# From
#
# alpha = R sqrt(omega / nu)
#
# obtain
#
# omega = nu alpha^2 / R^2
#
omega = nu * womersley**2 / R**2

period = 2.0 * np.pi / omega

period_steps = round(period)

# Use the actual lattice frequency corresponding
# to an integer number of time steps per period.
omega = 2.0 * np.pi / period_steps

womersley_actual = R * np.sqrt(omega / nu)


print()
print("Womersley-flow parameters")
print("--------------------------")

print(f"tau:                 {tau:.6f}")

print(f"nu:                  {nu:.8f}")

print(f"Target alpha:        {womersley:.6f}")

print(f"Actual alpha:        {womersley_actual:.6f}")

print(f"omega:               {omega:.8e}")

print(f"Period:              {period_steps} steps")


# ============================================================
# Simulation
# ============================================================

sim = AxisymmetricLBMSimulation(
    nz=nz,
    nr=nr,
    tau=tau,
    rho0=rho0,
)


rho_initial = rho0 * np.ones((nz, nr))

u_initial = np.zeros((2, nz, nr))

sim.set_state(
    rho_initial,
    u_initial,
)


# ============================================================
# Run several periods first to eliminate the initial transient
# ============================================================

transient_periods = 8

measurement_periods = 1

total_steps = (transient_periods + measurement_periods) * period_steps


# Store profiles at four phases in the final period:
#
# 0
# pi/2
# pi
# 3pi/2
#
sample_offsets = np.array(
    [
        0,
        period_steps // 4,
        period_steps // 2,
        3 * period_steps // 4,
    ]
)

sample_steps = transient_periods * period_steps + sample_offsets

numerical_profiles = []
sample_times = []


# ============================================================
# Time stepping
# ============================================================

for step in range(total_steps):
    time = float(step)

    acceleration = g0 * np.cos(omega * time)

    external_force = np.zeros((2, nz, nr))

    external_force[0] = rho0 * acceleration

    sim.set_external_force(external_force)

    sim.step(stream_operator=(stream_axisymmetric_pipe))

    current_step = sim.step_number

    if current_step in sample_steps:
        _, u = sim.macroscopic()

        uz = np.mean(
            u[0],
            axis=0,
        )

        numerical_profiles.append(uz.copy())

        sample_times.append(float(current_step))


numerical_profiles = np.asarray(numerical_profiles)

sample_times = np.asarray(sample_times)


# ============================================================
# Analytical Womersley solution
# ============================================================


def womersley_solution(
    r,
    t,
    radius,
    nu,
    omega,
    g0,
):
    """
    Analytical fully developed Womersley velocity profile.
    """

    alpha = radius * np.sqrt(omega / nu)

    # i^(3/2)
    lambda_factor = (-1.0 + 1.0j) / np.sqrt(2.0)

    argument = lambda_factor * alpha * r / radius

    argument_wall = lambda_factor * alpha

    profile_complex = (
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

    return np.real(profile_complex * np.exp(1.0j * omega * t))


analytical_profiles = []

for t in sample_times:
    analytical_profiles.append(
        womersley_solution(
            r=r,
            t=t,
            radius=R,
            nu=nu,
            omega=omega,
            g0=g0,
        )
    )


analytical_profiles = np.asarray(analytical_profiles)


# ============================================================
# Relative errors
# ============================================================

errors = []

for numerical, analytical in zip(
    numerical_profiles,
    analytical_profiles,
):
    error = np.sqrt(np.sum((numerical - analytical) ** 2) / np.sum(analytical**2))

    errors.append(error)


errors = np.asarray(errors)


print()
print("Phase-dependent relative L2 errors")

print("----------------------------------")

for t, error in zip(
    sample_times,
    errors,
):
    phase = (omega * t) % (2.0 * np.pi)

    print(f"phase = {phase:8.4f}   error = {error:.6e}")


# ============================================================
# Plot four phases
# ============================================================

fig, axes = plt.subplots(
    1,
    4,
    figsize=(13, 4),
    sharey=True,
)


for j, ax in enumerate(axes):
    phase = (omega * sample_times[j]) % (2.0 * np.pi)

    ax.plot(
        numerical_profiles[j],
        r / R,
        "o",
        label="LBM",
    )

    ax.plot(
        analytical_profiles[j],
        r / R,
        "-",
        label="Analytical",
    )

    ax.set_title(rf"$\omega t={phase:.2f}$")

    ax.set_xlabel(r"$u_z$")

    ax.grid(True)


axes[0].set_ylabel(r"$r/R$")

axes[0].legend()

fig.suptitle(rf"Womersley flow, $\alpha={womersley_actual:.2f}$")

fig.tight_layout()

plt.show()
