import numpy as np

from lbm.electrostatics.sem.mesh1d import (
    assemble_poisson_1d,
)


def solve_poisson_dirichlet_multielement(
    num_elements: int,
    order: int,
    x_left: float,
    x_right: float,
    source_function,
    phi_left: float,
    phi_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve

        -phi'' = f

    on a multi-element 1-D SEM mesh with Dirichlet
    boundary conditions.
    """

    (
        x,
        stiffness,
        rhs,
    ) = assemble_poisson_1d(
        num_elements=num_elements,
        order=order,
        x_left=x_left,
        x_right=x_right,
        source_function=source_function,
    )

    ndof = x.size

    phi = np.zeros(
        ndof,
        dtype=np.result_type(
            rhs,
            phi_left,
            phi_right,
        ),
    )

    phi[0] = phi_left
    phi[-1] = phi_right

    interior = np.arange(
        1,
        ndof - 1,
    )

    boundary = np.array(
        [
            0,
            ndof - 1,
        ]
    )

    rhs_interior = (
        rhs[interior]
        - stiffness[
            np.ix_(
                interior,
                boundary,
            )
        ]
        @ phi[boundary]
    )

    stiffness_interior = stiffness[
        np.ix_(
            interior,
            interior,
        )
    ]

    phi[interior] = np.linalg.solve(
        stiffness_interior,
        rhs_interior,
    )

    return x, phi
