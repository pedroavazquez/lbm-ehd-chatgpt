import numpy as np
import pytest

from lbm.electrostatics.sem.poisson2d import (
    solve_poisson_dirichlet_single_element_2d,
)


def exact_solution(
    x,
    y,
):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_function(
    x,
    y,
):
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def zero_boundary(
    x,
    y,
):
    return np.zeros_like(x)


@pytest.mark.parametrize(
    "order",
    [4, 6, 8, 10],
)
def test_poisson_2d_single_element(
    order,
):

    coordinates, phi = solve_poisson_dirichlet_single_element_2d(
        order_x=order,
        order_y=order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=source_function,
        boundary_function=zero_boundary,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    exact = exact_solution(
        x,
        y,
    )

    relative_error = np.sqrt(np.sum((phi - exact) ** 2) / np.sum(exact**2))

    tolerance = {
        4: 1.0e-2,
        6: 1.0e-4,
        8: 1.0e-6,
        10: 1.0e-8,
    }[order]

    assert relative_error < tolerance


def test_2d_linear_harmonic_solution():

    # phi = 1 + 2x - 3y
    #
    # laplacian(phi) = 0.
    def source(x, y):
        return np.zeros_like(x)

    def boundary(x, y):
        return 1.0 + 2.0 * x - 3.0 * y

    coordinates, phi = solve_poisson_dirichlet_single_element_2d(
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=source,
        boundary_function=boundary,
    )

    exact = boundary(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-12,
    )
