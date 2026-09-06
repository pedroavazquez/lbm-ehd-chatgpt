import numpy as np

from lbm.electrostatics.sem.element import (
    physical_element_matrices,
)


def physical_element_matrices_2d(
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct tensor-product SEM matrices for one rectangular element.

    Returns
    -------
    coordinates : ndarray, shape (ndof, 2)
        Physical node coordinates, flattened with x varying fastest.

    mass : ndarray, shape (ndof, ndof)
        2-D tensor-product mass matrix.

    stiffness : ndarray, shape (ndof, ndof)
        2-D Laplacian stiffness matrix

            K = Kx kron My + Mx kron Ky.

    shape : ndarray, shape (2,)
        [nx_local, ny_local].
    """

    (
        x_nodes,
        mass_x,
        stiffness_x,
    ) = physical_element_matrices(
        order=order_x,
        x_left=x_left,
        x_right=x_right,
    )

    (
        y_nodes,
        mass_y,
        stiffness_y,
    ) = physical_element_matrices(
        order=order_y,
        x_left=y_bottom,
        x_right=y_top,
    )

    # x varies fastest in the flattened ordering.
    X, Y = np.meshgrid(
        x_nodes,
        y_nodes,
        indexing="xy",
    )

    coordinates = np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
        )
    )

    mass = np.kron(
        mass_y,
        mass_x,
    )

    stiffness = np.kron(
        mass_y,
        stiffness_x,
    ) + np.kron(
        stiffness_y,
        mass_x,
    )

    shape = np.array(
        [
            order_x + 1,
            order_y + 1,
        ]
    )

    return (
        coordinates,
        mass,
        stiffness,
        shape,
    )
