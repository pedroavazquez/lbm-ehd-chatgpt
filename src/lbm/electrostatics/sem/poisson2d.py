import numpy as np

from lbm.electrostatics.sem.element2d import (
    physical_element_matrices_2d,
)


def solve_poisson_dirichlet_single_element_2d(
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
]:
    """
    Solve

        -laplacian(phi) = f

    on a single rectangular spectral element with
    Dirichlet boundary conditions on all four sides.

    Parameters
    ----------
    source_function : callable
        f(x, y)

    boundary_function : callable
        phi(x, y) on the boundary.

    Returns
    -------
    coordinates : ndarray, shape (ndof, 2)

    phi : ndarray, shape (ndof,)
        Numerical potential at tensor-product GLL nodes.
    """

    (
        coordinates,
        mass,
        stiffness,
        shape,
    ) = physical_element_matrices_2d(
        order_x=order_x,
        order_y=order_y,
        x_left=x_left,
        x_right=x_right,
        y_bottom=y_bottom,
        y_top=y_top,
    )

    nx_local = int(shape[0])

    ny_local = int(shape[1])

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    source_values = source_function(
        x,
        y,
    )

    rhs = mass @ source_values

    # --------------------------------------------------------
    # Identify boundary nodes.
    #
    # Flattened ordering:
    #
    # global local index = j * nx_local + i
    # --------------------------------------------------------

    boundary_mask = np.zeros(
        nx_local * ny_local,
        dtype=bool,
    )

    for j in range(ny_local):
        for i in range(nx_local):
            index = j * nx_local + i

            if i == 0 or i == nx_local - 1 or j == 0 or j == ny_local - 1:
                boundary_mask[index] = True

    boundary = np.where(boundary_mask)[0]

    interior = np.where(~boundary_mask)[0]

    phi = np.zeros(
        nx_local * ny_local,
        dtype=np.result_type(
            rhs,
            boundary_function(
                x[boundary],
                y[boundary],
            ),
        ),
    )

    phi[boundary] = boundary_function(
        x[boundary],
        y[boundary],
    )

    # --------------------------------------------------------
    # Interior Dirichlet system:
    #
    # K_II phi_I =
    #
    #     b_I - K_IB phi_B
    # --------------------------------------------------------

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
    )
