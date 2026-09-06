import numpy as np

from lbm.electrostatics.sem.differentiation import (
    differentiation_matrix,
)
from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def reference_element_matrices(
    order: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct GLL nodes and the reference-element SEM matrices.

    The reference element is

        xi in [-1, 1].

    Parameters
    ----------
    order : int
        Polynomial order p.

    Returns
    -------
    nodes : ndarray, shape (p + 1,)
        GLL interpolation nodes.

    weights : ndarray, shape (p + 1,)
        GLL quadrature weights.

    mass : ndarray, shape (p + 1, p + 1)
        Diagonal quadrature mass matrix

            M = diag(w_i).

    stiffness : ndarray, shape (p + 1, p + 1)
        Reference stiffness matrix

            K = D^T M D.
    """

    nodes, weights = gll_nodes_weights(order)

    D = differentiation_matrix(nodes)

    mass = np.diag(weights)

    stiffness = D.T @ mass @ D

    return (
        nodes,
        weights,
        mass,
        stiffness,
    )


def physical_element_matrices(
    order: int,
    x_left: float,
    x_right: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct physical 1-D SEM element matrices on

        x in [x_left, x_right].

    Mapping
    -------

        x(xi)
        =
        x_center + h/2 * xi

    with

        h = x_right - x_left.

    Returns
    -------
    x_nodes : ndarray
        Physical GLL node coordinates.

    mass : ndarray
        Physical mass matrix

            M_x = h/2 M_ref.

    stiffness : ndarray
        Physical stiffness matrix

            K_x = 2/h K_ref.
    """

    if x_right <= x_left:
        raise ValueError("x_right must be greater than x_left")

    (
        xi,
        _,
        mass_ref,
        stiffness_ref,
    ) = reference_element_matrices(order)

    h = x_right - x_left

    center = 0.5 * (x_left + x_right)

    x_nodes = center + 0.5 * h * xi

    mass = 0.5 * h * mass_ref

    stiffness = 2.0 / h * stiffness_ref

    return (
        x_nodes,
        mass,
        stiffness,
    )
