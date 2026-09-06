from collections.abc import Callable
from typing import Protocol

import numpy as np


class HydrodynamicSolver(Protocol):
    nx: int
    ny: int

    def set_force(
        self,
        force: np.ndarray,
    ) -> None: ...

    def set_state(
        self,
        rho: np.ndarray,
        u: np.ndarray,
    ) -> None: ...

    def macroscopic(
        self,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def step(
        self,
        stream_operator=None,
    ) -> None: ...


class ElectrostaticSolver(Protocol):
    def solve(
        self,
        source_function: Callable,
    ) -> np.ndarray: ...

    def reconstruct_electric_field(
        self,
    ) -> np.ndarray: ...

    def electric_field_grid(
        self,
    ) -> np.ndarray: ...


class FieldInterpolator(Protocol):
    nx_lbm: int
    ny_lbm: int

    def interpolate_vector_to_lbm_storage(
        self,
        field_sem: np.ndarray,
    ) -> np.ndarray: ...
