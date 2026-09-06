from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

FieldFunction = Callable[
    [np.ndarray, np.ndarray],
    object,
]


@dataclass(frozen=True)
class DirichletBC:
    """
    phi = value(x, y)
    """

    value: FieldFunction


@dataclass(frozen=True)
class NeumannBC:
    """
    n . kappa grad(phi) = flux(x, y)
    """

    flux: FieldFunction


@dataclass(frozen=True)
class RobinBC:
    """
    n . kappa grad(phi)
    + beta phi
    = rhs
    """

    beta: FieldFunction
    rhs: FieldFunction


BoundaryCondition = DirichletBC | NeumannBC | RobinBC


def to_legacy_boundary_conditions(
    boundary_conditions: dict[
        str,
        BoundaryCondition,
    ],
) -> dict:
    """
    Convert the class-based representation to the tuple
    representation used by the already-verified kernels.
    """

    result = {}

    for side, bc in boundary_conditions.items():
        if isinstance(
            bc,
            DirichletBC,
        ):
            result[side] = (
                "dirichlet",
                bc.value,
            )

        elif isinstance(
            bc,
            NeumannBC,
        ):
            result[side] = (
                "neumann",
                bc.flux,
            )

        elif isinstance(
            bc,
            RobinBC,
        ):
            result[side] = (
                "robin",
                bc.beta,
                bc.rhs,
            )

        else:
            raise TypeError(f"Unsupported boundary condition on side '{side}'")

    return result
