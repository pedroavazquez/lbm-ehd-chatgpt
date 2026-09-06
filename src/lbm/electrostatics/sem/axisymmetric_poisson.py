import numpy as np

from lbm.electrostatics.sem.axisymmetric import (
    assemble_axisymmetric_2d,
)


def solve_axisymmetric_dirichlet_2d(
    num_elements_z: int,
    num_elements_r: int,
    order_z: int,
    order_r: int,
    z_left: float,
    z_right: float,
    r_bottom: float,
    r_top: float,
    coefficient_function,
    source_function,
    boundary_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Solve an axisymmetric variable-coefficient Poisson problem
    with Dirichlet conditions on the outer rectangular boundary.

    This is primarily a verification solver. A mixed-boundary
    version will follow.
    """

    (
        coordinates,
        stiffness,
        rhs,
        shape,
    ) = assemble_axisymmetric_2d(
        num_elements_z=num_elements_z,
        num_elements_r=num_elements_r,
        order_z=order_z,
        order_r=order_r,
        z_left=z_left,
        z_right=z_right,
        r_bottom=r_bottom,
        r_top=r_top,
        coefficient_function=coefficient_function,
        source_function=source_function,
    )

    nz_global, nr_global = shape

    ndof = coordinates.shape[0]

    boundary_mask = np.zeros(
        ndof,
        dtype=bool,
    )

    for j in range(nr_global):
        for i in range(nz_global):
            index = j * nz_global + i

            if i == 0 or i == nz_global - 1 or j == 0 or j == nr_global - 1:
                boundary_mask[index] = True

    boundary = np.where(boundary_mask)[0]

    interior = np.where(~boundary_mask)[0]

    z = coordinates[:, 0]
    r = coordinates[:, 1]

    boundary_values = np.asarray(
        boundary_function(
            z[boundary],
            r[boundary],
        )
    )

    dtype = np.result_type(
        stiffness,
        rhs,
        boundary_values,
    )

    phi = np.zeros(
        ndof,
        dtype=dtype,
    )

    phi[boundary] = boundary_values

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
        stiffness[
            np.ix_(
                interior,
                interior,
            )
        ],
        rhs_interior,
    )

    return (
        coordinates,
        phi,
        shape,
    )
