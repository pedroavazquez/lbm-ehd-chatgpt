import numpy as np

from lbm.electrostatics.sem.differentiation import (
    differentiation_matrix,
)
from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def reconstruct_gradient_2d(
    phi: np.ndarray,
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> np.ndarray:
    """
    Reconstruct the gradient of a global SEM nodal field.

    The structured mesh uses continuous GLL endpoint sharing.

    Parameters
    ----------
    phi : ndarray, shape (ndof,)
        Global SEM nodal field. May be real or complex.

    num_elements_x, num_elements_y : int
        Number of spectral elements.

    order_x, order_y : int
        Polynomial order in each element.

    x_left, x_right : float
        Domain limits in the first coordinate.

    y_bottom, y_top : float
        Domain limits in the second coordinate.

    Returns
    -------
    gradient : ndarray, shape (2, ndof)

        gradient[0] = dphi/dx
        gradient[1] = dphi/dy

    Notes
    -----
    At nodes shared by several elements, each neighboring
    element provides a derivative estimate. These values are
    averaged arithmetically.

    For a smooth solution the estimates converge to the same
    derivative under h- or p-refinement.
    """

    nx_global = num_elements_x * order_x + 1

    ny_global = num_elements_y * order_y + 1

    ndof = nx_global * ny_global

    if phi.shape != (ndof,):
        raise ValueError(f"phi must have shape ({ndof},)")

    if x_right <= x_left:
        raise ValueError("x_right must be greater than x_left")

    if y_top <= y_bottom:
        raise ValueError("y_top must be greater than y_bottom")

    # ========================================================
    # Reference differentiation matrices
    # ========================================================

    xi, _ = gll_nodes_weights(order_x)

    eta, _ = gll_nodes_weights(order_y)

    Dx_ref = differentiation_matrix(xi)

    Dy_ref = differentiation_matrix(eta)

    hx = (x_right - x_left) / num_elements_x

    hy = (y_top - y_bottom) / num_elements_y

    Dx = 2.0 / hx * Dx_ref

    Dy = 2.0 / hy * Dy_ref

    nx_local = order_x + 1

    ny_local = order_y + 1

    dtype = np.result_type(
        phi,
        float,
    )

    gradient_sum = np.zeros(
        (2, ndof),
        dtype=dtype,
    )

    contribution_count = np.zeros(
        ndof,
        dtype=int,
    )

    # ========================================================
    # Element loop
    # ========================================================

    for ey in range(num_elements_y):
        for ex in range(num_elements_x):
            global_indices = np.empty(
                (
                    ny_local,
                    nx_local,
                ),
                dtype=int,
            )

            for j in range(ny_local):
                J = ey * order_y + j

                for i in range(nx_local):
                    I = ex * order_x + i

                    global_indices[
                        j,
                        i,
                    ] = J * nx_global + I

            # ------------------------------------------------
            # Local nodal field, arranged as
            #
            #     phi_local[j, i]
            #
            # with x varying along columns.
            # ------------------------------------------------

            phi_local = phi[global_indices]

            # ------------------------------------------------
            # Spectral derivatives
            #
            # For each y row:
            #
            #     dphi/dx = D_x phi
            #
            # which in matrix-row storage becomes
            #
            #     phi_local @ D_x^T
            #
            # For y:
            #
            #     dphi/dy = D_y @ phi_local
            # ------------------------------------------------

            dphi_dx_local = phi_local @ Dx.T

            dphi_dy_local = Dy @ phi_local

            # ------------------------------------------------
            # Assemble derivative contributions.
            # ------------------------------------------------

            for j in range(ny_local):
                for i in range(nx_local):
                    index = global_indices[
                        j,
                        i,
                    ]

                    gradient_sum[
                        0,
                        index,
                    ] += dphi_dx_local[
                        j,
                        i,
                    ]

                    gradient_sum[
                        1,
                        index,
                    ] += dphi_dy_local[
                        j,
                        i,
                    ]

                    contribution_count[index] += 1

    if np.any(contribution_count == 0):
        raise RuntimeError("Some SEM nodes received no gradient contribution")

    gradient = (
        gradient_sum
        / contribution_count[
            None,
            :,
        ]
    )

    return gradient


def reconstruct_electric_field_2d(
    phi: np.ndarray,
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
) -> np.ndarray:
    """
    Reconstruct the electric field

        E = -grad(phi)

    at the global SEM nodes.

    The returned field may be real or complex.
    """

    gradient = reconstruct_gradient_2d(
        phi=phi,
        num_elements_x=num_elements_x,
        num_elements_y=num_elements_y,
        order_x=order_x,
        order_y=order_y,
        x_left=x_left,
        x_right=x_right,
        y_bottom=y_bottom,
        y_top=y_top,
    )

    return -gradient
