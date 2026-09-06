import numpy as np
import pytest

from lbm.electrostatics.sem.mesh2d import (
    assemble_poisson_2d,
)
from lbm.electrostatics.sem.variable_coefficient import (
    assemble_variable_coefficient_2d,
    solve_variable_coefficient_dirichlet_2d,
)


def zero_source(
    x,
    y,
):
    return np.zeros_like(x)


def zero_boundary(
    x,
    y,
):
    return np.zeros_like(x)


def test_constant_coefficient_matches_existing_operator():

    def coefficient(
        x,
        y,
    ):
        return np.ones_like(x)

    (
        coordinates_old,
        stiffness_old,
        rhs_old,
        shape_old,
    ) = assemble_poisson_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=4,
        order_y=4,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=zero_source,
    )

    (
        coordinates_new,
        stiffness_new,
        rhs_new,
        shape_new,
    ) = assemble_variable_coefficient_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=4,
        order_y=4,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
        source_function=zero_source,
    )

    assert shape_new == shape_old

    assert np.allclose(
        coordinates_new,
        coordinates_old,
    )

    assert np.allclose(
        stiffness_new,
        stiffness_old,
        atol=1.0e-12,
    )

    assert np.allclose(
        rhs_new,
        rhs_old,
        atol=1.0e-14,
    )


def exact_solution(
    x,
    y,
):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def variable_epsilon(
    x,
    y,
):
    return 1.0 + x + y


def manufactured_source(
    x,
    y,
):
    """
    For

        epsilon = 1 + x + y

        phi = sin(pi x) sin(pi y),

    compute

        rho_e = -div(epsilon grad(phi)).

    Since

        grad epsilon = (1, 1)

    and

        Laplacian(phi) = -2 pi^2 phi,

    then

        -div(epsilon grad phi)

        = 2 pi^2 epsilon phi
          - dphi/dx
          - dphi/dy.
    """

    phi = np.sin(np.pi * x) * np.sin(np.pi * y)

    dphi_dx = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

    dphi_dy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

    epsilon = variable_epsilon(
        x,
        y,
    )

    return 2.0 * np.pi**2 * epsilon * phi - dphi_dx - dphi_dy


@pytest.mark.parametrize(
    "order",
    [4, 6, 8],
)
def test_variable_coefficient_manufactured_solution(
    order,
):

    (
        coordinates,
        phi,
        _,
    ) = solve_variable_coefficient_dirichlet_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=order,
        order_y=order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=variable_epsilon,
        source_function=manufactured_source,
        boundary_function=zero_boundary,
    )

    exact = exact_solution(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    relative_error = np.sqrt(np.sum((phi - exact) ** 2) / np.sum(exact**2))

    tolerance = {
        4: 2.0e-3,
        6: 2.0e-5,
        8: 2.0e-7,
    }[order]

    assert relative_error < tolerance


def test_constant_nonunity_coefficient():

    epsilon0 = 3.7

    def coefficient(
        x,
        y,
    ):
        return epsilon0 * np.ones_like(x)

    def source(
        x,
        y,
    ):
        return epsilon0 * 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    (
        coordinates,
        phi,
        _,
    ) = solve_variable_coefficient_dirichlet_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=6,
        order_y=6,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
        source_function=source,
        boundary_function=zero_boundary,
    )

    exact = exact_solution(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    assert np.allclose(
        phi,
        exact,
        atol=2.0e-5,
    )
