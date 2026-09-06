import matplotlib.pyplot as plt
import numpy as np

from lbm.axisymmetric_simulation import (
    AxisymmetricLBMSimulation,
)
from lbm.boundary.axisymmetric import (
    stream_axisymmetric_pipe,
)
from lbm.lattices.d2q9 import CS2
from lbm.utils.convergence import relative_change

nz = 8
nr = 32

rho0 = 1.0

tau = 0.8

acceleration = 1.0e-6

max_steps = 100000
check_every = 100
tolerance = 1.0e-10


nu = CS2 * (tau - 0.5)

R = float(nr)

r = np.arange(nr) + 0.5


# ============================================================
# Simulation
# ============================================================

sim = AxisymmetricLBMSimulation(
    nz=nz,
    nr=nr,
    tau=tau,
    rho0=rho0,
)


# ============================================================
# External force density
#
# F_z = rho g
# ============================================================

external_force = np.zeros((2, nz, nr))

external_force[0] = rho0 * acceleration

sim.set_external_force(external_force)


# ============================================================
# Zero physical velocity
# ============================================================

rho_initial = rho0 * np.ones((nz, nr))

u_initial = np.zeros((2, nz, nr))

sim.set_state(
    rho_initial,
    u_initial,
)


# ============================================================
# Time integration
# ============================================================

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


# ============================================================
# Final solution
# ============================================================

rho, u = sim.macroscopic()

uz = np.mean(
    u[0],
    axis=0,
)

ur = np.mean(
    u[1],
    axis=0,
)


# ============================================================
# Analytical Hagen-Poiseuille solution
# ============================================================

uz_exact = acceleration / (4.0 * nu) * (R**2 - r**2)


relative_l2_error = np.sqrt(np.sum((uz - uz_exact) ** 2) / np.sum(uz_exact**2))


print()
print("General axisymmetric Hagen-Poiseuille test")
print("------------------------------------------")

print(f"tau:                       {tau:.6f}")

print(f"nu:                        {nu:.8f}")

print(f"Steps:                     {sim.step_number}")

print(f"Converged:                 {converged}")

print(f"Relative L2 error:          {relative_l2_error:.6e}")

print(f"Maximum |u_r|:              {np.max(np.abs(ur)):.3e}")

print(f"Maximum density deviation:  {np.max(np.abs(rho - rho0)):.3e}")


plt.figure()

plt.plot(
    uz,
    r / R,
    "o",
    label="Axisymmetric LBM",
)

plt.plot(
    uz_exact,
    r / R,
    "-",
    label="Analytical",
)

plt.xlabel(r"$u_z$")
plt.ylabel(r"$r/R$")

plt.title("Hagen-Poiseuille flow")

plt.legend()
plt.tight_layout()

plt.show()
