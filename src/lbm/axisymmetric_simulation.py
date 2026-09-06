import numpy as np

from lbm.hydro.axisymmetric import (
    axisymmetric_macroscopic,
    collide_axisymmetric,
    initialize_axisymmetric,
)
from lbm.hydro.streaming import stream


class AxisymmetricLBMSimulation:
    """
    General non-swirling axisymmetric D2Q9 simulation.

    Coordinates
    -----------
    axis 0 : z
    axis 1 : r
    """

    def __init__(
        self,
        nz: int,
        nr: int,
        tau: float,
        rho0: float = 1.0,
    ):

        if tau <= 0.5:
            raise ValueError("tau must be greater than 0.5")

        self.nz = nz
        self.nr = nr

        self.tau = tau
        self.rho0 = rho0

        self.external_force = None

        rho = rho0 * np.ones((nz, nr))

        u = np.zeros((2, nz, nr))

        self.f = initialize_axisymmetric(
            rho,
            u,
            tau=tau,
            rho0=rho0,
        )

        self.step_number = 0

    def set_external_force(
        self,
        force: np.ndarray,
    ) -> None:

        expected_shape = (
            2,
            self.nz,
            self.nr,
        )

        if force.shape != expected_shape:
            raise ValueError(f"force must have shape {expected_shape}")

        self.external_force = force.copy()

    def clear_external_force(
        self,
    ) -> None:

        self.external_force = None

    def set_state(
        self,
        rho: np.ndarray,
        u: np.ndarray,
    ) -> None:

        self.f = initialize_axisymmetric(
            rho,
            u,
            tau=self.tau,
            external_force=(self.external_force),
            rho0=self.rho0,
        )

    def macroscopic(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:

        return axisymmetric_macroscopic(
            self.f,
            tau=self.tau,
            external_force=(self.external_force),
            rho0=self.rho0,
        )

    def step(
        self,
        stream_operator=stream,
    ) -> None:

        f_post = collide_axisymmetric(
            self.f,
            tau=self.tau,
            external_force=(self.external_force),
            rho0=self.rho0,
        )

        self.f = stream_operator(f_post)

        self.step_number += 1
