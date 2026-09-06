import numpy as np

from lbm.electrostatics.sem.mesh2d import (
    assemble_poisson_2d,
)


def solve_poisson_dirichlet_multielement_2d(
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    source_function,
    boundary_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Solve

        -laplacian(phi) = f

    on a structured multi-element rectangular SEM mesh
    with Dirichlet boundary conditions.
    """

    (
        coordinates,
        stiffness,
        rhs,
        global_shape,
    ) = assemble_poisson_2d(
        num_elements_x=num_elements_x,
        num_elements_y=num_elements_y,
        order_x=order_x,
        order_y=order_y,
        x_left=x_left,
        x_right=x_right,
        y_bottom=y_bottom,
        y_top=y_top,
        source_function=source_function,
    )

    nx_global, ny_global = global_shape

    ndof = coordinates.shape[0]

    boundary_mask = np.zeros(
        ndof,
        dtype=bool,
    )

    for J in range(ny_global):
        for I in range(nx_global):
            index = J * nx_global + I

            if I == 0 or I == nx_global - 1 or J == 0 or J == ny_global - 1:
                boundary_mask[index] = True

    boundary = np.where(boundary_mask)[0]

    interior = np.where(~boundary_mask)[0]

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    boundary_values = boundary_function(
        x[boundary],
        y[boundary],
    )

    phi = np.zeros(
        ndof,
        dtype=np.result_type(
            rhs,
            boundary_values,
        ),
    )

    phi[boundary] = boundary_values

    stiffness_interior = stiffness[
        np.ix_(
            interior,
            interior,
        )
    ]

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

    phi[interior] = np.linalg.solve(
        stiffness_interior,
        rhs_interior,
    )

    return (
        coordinates,
        phi,
        global_shape,
    )
