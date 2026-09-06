import numpy as np

from lbm.electrostatics.sem.variable_coefficient import (
    solve_variable_coefficient_dirichlet_2d,
)


def test_uniform_complex_coefficient_linear_potential():
    """
    Solve

        div(kappa grad(phi)) = 0

    with uniform complex kappa on a rectangular domain.

    Boundary conditions:

        phi(0, y) = 1 + 0.5j
        phi(1, y) = 0

    and we impose the exact linear solution on the full boundary.

    The exact solution is

        phi(x) = phi0 * (1 - x)

    which is independent of kappa when kappa is uniform.
    """

    sigma = 2.0
    epsilon = 3.0
    omega = 5.0

    kappa0 = sigma + 1.0j * omega * epsilon

    phi0 = 1.0 + 0.5j

    def coefficient(
        x,
        y,
    ):
        return kappa0 * np.ones_like(
            x,
            dtype=complex,
        )

    def source(
        x,
        y,
    ):
        return np.zeros_like(
            x,
            dtype=complex,
        )

    def exact_potential(
        x,
        y,
    ):
        return phi0 * (1.0 - x)

    (
        coordinates,
        phi,
        _,
    ) = solve_variable_coefficient_dirichlet_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
        source_function=source,
        boundary_function=exact_potential,
    )

    exact = exact_potential(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    assert np.iscomplexobj(phi)

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-12,
    )
