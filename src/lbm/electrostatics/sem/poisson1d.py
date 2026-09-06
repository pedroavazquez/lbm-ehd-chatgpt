import numpy as np

from lbm.electrostatics.sem.element import (
    physical_element_matrices,
)


def solve_poisson_dirichlet_single_element(
    order: int,
    x_left: float,
    x_right: float,
    source_function,
    phi_left: float,
    phi_right: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve

        -d2phi/dx2 = f(x)

    on a single spectral element with Dirichlet boundary conditions.

    Parameters
    ----------
    order : int
        Polynomial order p.

    x_left, x_right : float
        Element endpoints.

    source_function : callable
        Function f(x).

    phi_left, phi_right : float
        Dirichlet boundary values.

    Returns
    -------
    x : ndarray
        Physical GLL node coordinates.

    phi : ndarray
        Numerical solution at the GLL nodes.
    """

    x, mass, stiffness = physical_element_matrices(
        order=order,
        x_left=x_left,
        x_right=x_right,
    )

    # --------------------------------------------------------
    # Weak-form right-hand side:
    #
    # b_i = integral l_i(x) f(x) dx
    #
    # With GLL quadrature this becomes
    #
    # b = M f
    # --------------------------------------------------------

    source_values = source_function(x)

    rhs = mass @ source_values

    # --------------------------------------------------------
    # Apply Dirichlet boundary conditions strongly.
    #
    # Unknowns are the interior GLL nodes.
    # --------------------------------------------------------

    interior = np.arange(
        1,
        order,
    )

    boundary = np.array(
        [
            0,
            order,
        ]
    )

    phi = np.zeros(
        order + 1,
        dtype=np.result_type(
            rhs,
            phi_left,
            phi_right,
        ),
    )

    phi[0] = phi_left
    phi[-1] = phi_right

    # Interior system:
    #
    # K_II phi_I
    # =
    # b_I - K_IB phi_B
    #
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
