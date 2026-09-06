import matplotlib.pyplot as plt
import numpy as np

from lbm.boundary.bounceback import stream_bounceback_y
from lbm.lattices.d2q9 import CS2
from lbm.simulation import LBMSimulation
from lbm.utils.convergence import relative_change

# ============================================================
# Simulation parameters
# ============================================================

nx = 8
ny = 32

tau = 0.8
rho0 = 1.0

acceleration = 1.0e-6


# ============================================================
# Steady-state convergence parameters
# ============================================================

max_steps = 100000
check_every = 100
tolerance = 1.0e-10


# ============================================================
# Derived quantities
# ============================================================

nu = CS2 * (tau - 0.5)

print(f"Kinematic viscosity = {nu:.8f}")


# ============================================================
# Create simulation
# ============================================================

sim = LBMSimulation(
    nx=nx,
    ny=ny,
    tau=tau,
    rho0=rho0,
)


# ============================================================
# Body-force field
# ============================================================

force = np.zeros((2, nx, ny))

force[0] = rho0 * acceleration

sim.set_force(force)


# ============================================================
# Initial physical state
#
# set_state() automatically handles the half-force correction,
# so here we simply specify the physical velocity u = 0.
# ============================================================

rho_initial = rho0 * np.ones((nx, ny))

u_initial = np.zeros((2, nx, ny))

sim.set_state(
    rho_initial,
    u_initial,
)


# ============================================================
# Verify initial state
# ============================================================

rho, u = sim.macroscopic()

print(f"Initial maximum velocity = {np.max(np.abs(u)):.3e}")


# ============================================================
# Time integration until steady state
# ============================================================

previous_u = None

converged = False

last_relative_change = np.inf


for _ in range(max_steps):
    sim.step(stream_operator=stream_bounceback_y)

    if sim.step_number % check_every == 0:
        _, u_current = sim.macroscopic()

        if previous_u is not None:
            last_relative_change = relative_change(
                u_current,
                previous_u,
            )

            if last_relative_change < tolerance:
                converged = True

                print()
                print(f"Converged at step {sim.step_number}")

                print(f"Relative velocity change = {last_relative_change:.3e}")

                break

        previous_u = u_current.copy()


if not converged:
    print()
    print(f"Warning: steady state was not reached after {max_steps} steps.")

    print(f"Last relative velocity change = {last_relative_change:.3e}")


# ============================================================
# Recover final macroscopic fields
# ============================================================

rho, u = sim.macroscopic()

ux = np.mean(
    u[0],
    axis=0,
)

uy = np.mean(
    u[1],
    axis=0,
)


# ============================================================
# Analytical Poiseuille solution
#
# nu d²u/dy² + g = 0
#
# with:
#
# u(0) = u(H) = 0
#
# gives:
#
# u(y) = g/(2 nu) y(H-y)
#
# For halfway bounce-back:
#
# walls:      y = 0 and y = H
# fluid nodes y_j = j + 1/2
#
# therefore H = ny.
# ============================================================

H = float(ny)

y = np.arange(ny) + 0.5

u_exact = acceleration / (2.0 * nu) * y * (H - y)


# ============================================================
# Relative L2 error
# ============================================================

relative_l2_error = np.sqrt(np.sum((ux - u_exact) ** 2) / np.sum(u_exact**2))


# ============================================================
# Diagnostics
# ============================================================

print()
print("Poiseuille-flow verification")
print("-----------------------------")

print(f"Grid:                       {nx} x {ny}")

print(f"Time steps:                 {sim.step_number}")

print(f"tau:                        {tau:.6f}")

print(f"Kinematic viscosity:        {nu:.8f}")

print(f"Acceleration:                {acceleration:.8e}")

print(f"Relative L2 error:           {relative_l2_error:.6e}")

print(f"Maximum numerical velocity:  {np.max(ux):.8e}")

print(f"Maximum analytical velocity: {np.max(u_exact):.8e}")

print(f"Maximum transverse velocity: {np.max(np.abs(uy)):.8e}")

print(f"Mean density:                {np.mean(rho):.12f}")

print(f"Maximum density deviation:   {np.max(np.abs(rho - rho0)):.3e}")


# ============================================================
# Plot
# ============================================================

plt.figure()

plt.plot(
    ux,
    y,
    "o",
    label="LBM",
)

plt.plot(
    u_exact,
    y,
    "-",
    label="Analytical",
)

plt.xlabel(r"$u_x$")
plt.ylabel(r"$y$")

plt.title("Plane Poiseuille flow")

plt.legend()
plt.tight_layout()

plt.show()
