import numpy as np

from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.forcing import guo_force
from lbm.hydro.macroscopic import macroscopic


def collide_bgk(f: np.ndarray, tau: float) -> np.ndarray:
    """
    Perform one BGK collision step.

    Parameters
    ----------
    f : ndarray, shape (9, nx, ny)
        Pre-collision distribution functions.

    tau : float
        Dimensionless relaxation time.

    Returns
    -------
    f_post : ndarray, shape (9, nx, ny)
        Post-collision distribution functions.
    """

    if tau <= 0.5:
        raise ValueError("tau must be greater than 0.5")

    rho, u = macroscopic(f)

    feq = equilibrium(rho, u)

    omega = 1.0 / tau

    f_post = f - omega * (f - feq)

    return f_post


def collide_bgk_forced(
    f: np.ndarray,
    force: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Perform BGK collision including Guo forcing.
    """

    if tau <= 0.5:
        raise ValueError("tau must be greater than 0.5")

    rho, u = macroscopic(
        f,
        force=force,
    )

    feq = equilibrium(
        rho,
        u,
    )

    source = guo_force(
        rho,
        u,
        force,
        tau,
    )

    omega = 1.0 / tau

    return f - omega * (f - feq) + source
