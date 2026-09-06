import numpy as np

from lbm.electrostatics.sem.element2d import (
    physical_element_matrices_2d,
)


def assemble_poisson_2d(
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    source_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Assemble the global SEM system for

        -laplacian(phi) = f

    on a uniform structured rectangular mesh.

    Neighboring elements share their interface GLL nodes.

    Returns
    -------
    coordinates : ndarray, shape (ndof, 2)

    stiffness : ndarray, shape (ndof, ndof)

    rhs : ndarray, shape (ndof,)

    global_shape : tuple
        (nx_global, ny_global)
    """

    if num_elements_x < 1:
        raise ValueError("num_elements_x must be at least 1")

    if num_elements_y < 1:
        raise ValueError("num_elements_y must be at least 1")

    if order_x < 1:
        raise ValueError("order_x must be at least 1")

    if order_y < 1:
        raise ValueError("order_y must be at least 1")

    if x_right <= x_left:
        raise ValueError("x_right must be greater than x_left")

    if y_top <= y_bottom:
        raise ValueError("y_top must be greater than y_bottom")

    # ========================================================
    # Global tensor-product node counts
    #
    # Each element contributes p new intervals, while
    # neighboring endpoints are shared.
    # ========================================================

    nx_global = num_elements_x * order_x + 1

    ny_global = num_elements_y * order_y + 1

    ndof = nx_global * ny_global

    coordinates = np.empty(
        (ndof, 2),
        dtype=float,
    )

    stiffness = np.zeros(
        (ndof, ndof),
        dtype=float,
    )

    rhs = np.zeros(
        ndof,
        dtype=float,
    )

    element_width = (x_right - x_left) / num_elements_x

    element_height = (y_top - y_bottom) / num_elements_y

    # ========================================================
    # Element loop
    # ========================================================

    for ey in range(num_elements_y):
        ya = y_bottom + ey * element_height

        yb = ya + element_height

        for ex in range(num_elements_x):
            xa = x_left + ex * element_width

            xb = xa + element_width

            (
                coordinates_local,
                mass_local,
                stiffness_local,
                local_shape,
            ) = physical_element_matrices_2d(
                order_x=order_x,
                order_y=order_y,
                x_left=xa,
                x_right=xb,
                y_bottom=ya,
                y_top=yb,
            )

            nx_local = int(local_shape[0])

            ny_local = int(local_shape[1])

            # ------------------------------------------------
            # Local-to-global mapping.
            #
            # Local flattening:
            #
            #     local = j * nx_local + i
            #
            # Global flattening:
            #
            #     global = J * nx_global + I
            # ------------------------------------------------

            global_indices = np.empty(
                nx_local * ny_local,
                dtype=int,
            )

            for j in range(ny_local):
                J = ey * order_y + j

                for i in range(nx_local):
                    I = ex * order_x + i

                    local_index = j * nx_local + i

                    global_index = J * nx_global + I

                    global_indices[local_index] = global_index

                    coordinates[global_index] = coordinates_local[local_index]

            # ------------------------------------------------
            # Local source and load vector
            # ------------------------------------------------

            x_local = coordinates_local[:, 0]

            y_local = coordinates_local[:, 1]

            source_local = source_function(
                x_local,
                y_local,
            )

            rhs_local = mass_local @ source_local

            # ------------------------------------------------
            # Assembly
            # ------------------------------------------------

            for a_local, a_global in enumerate(global_indices):
                rhs[a_global] += rhs_local[a_local]

                for (
                    b_local,
                    b_global,
                ) in enumerate(global_indices):
                    stiffness[
                        a_global,
                        b_global,
                    ] += stiffness_local[
                        a_local,
                        b_local,
                    ]

    return (
        coordinates,
        stiffness,
        rhs,
        (
            nx_global,
            ny_global,
        ),
    )
