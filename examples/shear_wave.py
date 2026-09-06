import matplotlib.pyplot as plt
import numpy as np

from lbm.hydro.collision import collide_bgk
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.hydro.streaming import stream
from lbm.lattices.d2q9 import CS2

# ------------------------------------------------------------
# Simulation parameters
# ------------------------------------------------------------

nx = 8
ny = 64

tau = 0.8

nsteps = 2000
output_every = 10

rho0 = 1.0
u0 = 0.01


# ------------------------------------------------------------
# Derived quantities
# ------------------------------------------------------------

nu_theory = CS2 * (tau - 0.5)

k = 2.0 * np.pi / ny

print(f"Theoretical viscosity: {nu_theory:.8f}")
print(f"Wave number:           {k:.8f}")


# ------------------------------------------------------------
# Initial condition
# ------------------------------------------------------------

rho = rho0 * np.ones((nx, ny))

u = np.zeros((2, nx, ny))

y = np.arange(ny)

u[0, :, :] = u0 * np.sin(k * y)[None, :]


# Initialize populations at equilibrium
f = equilibrium(rho, u)


# ------------------------------------------------------------
# Storage
# ------------------------------------------------------------

times = []
amplitudes = []


# ------------------------------------------------------------
# Time integration
# ------------------------------------------------------------

for step in range(nsteps + 1):
    if step % output_every == 0:
        rho, u = macroscopic(f)

        ux_mean = np.mean(u[0], axis=0)

        # Projection of ux onto sin(k y)
        amplitude = 2.0 / ny * np.sum(ux_mean * np.sin(k * y))

        times.append(step)
        amplitudes.append(amplitude)

    if step == nsteps:
        break

    f = collide_bgk(f, tau)
    f = stream(f)


# ------------------------------------------------------------
# Convert to arrays
# ------------------------------------------------------------

times = np.asarray(times, dtype=float)
amplitudes = np.asarray(amplitudes)


# ------------------------------------------------------------
# Fit exponential decay
#
# A(t) = A0 exp(-nu k^2 t)
#
# log(A) = log(A0) - nu k^2 t
# ------------------------------------------------------------

mask = amplitudes > 0.0

fit = np.polyfit(
    times[mask],
    np.log(amplitudes[mask]),
    1,
)

slope = fit[0]

nu_measured = -slope / k**2


print()
print(f"Measured viscosity:    {nu_measured:.8f}")
print(f"Theoretical viscosity: {nu_theory:.8f}")
print(f"Relative error:       {abs(nu_measured - nu_theory) / nu_theory:.3e}")


# ------------------------------------------------------------
# Analytical solution
# ------------------------------------------------------------

amplitude_theory = u0 * np.exp(-nu_theory * k**2 * times)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

plt.figure()

plt.plot(
    times,
    amplitudes,
    "o",
    markersize=3,
    label="LBM",
)

plt.plot(
    times,
    amplitude_theory,
    "-",
    label="Navier-Stokes theory",
)

plt.xlabel("Time step")
plt.ylabel("Shear-wave amplitude")
plt.legend()
plt.tight_layout()

plt.show()
