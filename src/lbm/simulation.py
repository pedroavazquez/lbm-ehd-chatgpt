from collections.abc import Callable

import numpy as np

from lbm.hydro.collision import (
    collide_bgk,
    collide_bgk_forced,
)
from lbm.hydro.equilibrium import equilibrium
from lbm.hydro.macroscopic import macroscopic
from lbm.hydro.streaming import stream

StreamOperator = Callable[[np.ndarray], np.ndarray]


class LBMSimulation:
    """
    Minimal D2Q9 BGK simulation object.

    The class stores the population field and basic
    hydrodynamic parameters. Numerical kernels remain
    separate pure functions.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float,
        rho0: float = 1.0,
    ):
        if tau <= 0.5:
            raise ValueError("tau must be greater than 0.5")

        self.nx = nx
        self.ny = ny
        self.tau = tau

        rho = rho0 * np.ones((nx, ny))

        u = np.zeros((2, nx, ny))

        self.f = equilibrium(
            rho,
            u,
        )

        self.force = None

        self.step_number = 0

    def set_force(
        self,
        force: np.ndarray,
    ) -> None:
        """
        Set the body-force density field.
        """

        expected_shape = (
            2,
            self.nx,
            self.ny,
        )

        if force.shape != expected_shape:
            raise ValueError(f"force must have shape {expected_shape}")

        self.force = force.copy()

    def clear_force(
        self,
    ) -> None:
        """
        Remove the body-force field.
        """

        self.force = None

    def set_state(
        self,
        rho: np.ndarray,
        u: np.ndarray,
    ) -> None:
        """
        Initialize populations from density and velocity.

        If forcing is active, u is interpreted as the
        physical velocity and the half-force correction
        is applied consistently.
        """

        expected_rho_shape = (
            self.nx,
            self.ny,
        )

        expected_u_shape = (
            2,
            self.nx,
            self.ny,
        )

        if rho.shape != expected_rho_shape:
            raise ValueError(f"rho must have shape {expected_rho_shape}")

        if u.shape != expected_u_shape:
            raise ValueError(f"u must have shape {expected_u_shape}")

        if self.force is None:
            u_distribution = u

        else:
            u_distribution = u - 0.5 * self.force / rho[None, :, :]

        self.f = equilibrium(
            rho,
            u_distribution,
        )

    def macroscopic(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Recover density and physical velocity.
        """

        return macroscopic(
            self.f,
            force=self.force,
        )

    def collide(
        self,
    ) -> np.ndarray:
        """
        Perform local BGK collision.
        """

        if self.force is None:
            return collide_bgk(
                self.f,
                self.tau,
            )

        return collide_bgk_forced(
            self.f,
            self.force,
            self.tau,
        )

    def step(
        self,
        stream_operator: StreamOperator = stream,
    ) -> None:
        """
        Advance the simulation by one time step.

        Parameters
        ----------
        stream_operator : callable
            Function taking the post-collision populations
            and returning populations after streaming and
            boundary treatment.

            The default is fully periodic streaming.
        """

        f_post = self.collide()

        self.f = stream_operator(f_post)

        self.step_number += 1
