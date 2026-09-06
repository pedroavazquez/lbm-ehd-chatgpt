import numpy as np
import pytest

from lbm.electrostatics.sem.poisson2d_multielement import (
    solve_poisson_dirichlet_multielement_2d,
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
    "num_elements",
    [1, 2, 3, 4],
)
def test_multielement_poisson_2d(
    num_elements,
):

    (
        coordinates,
        phi,
        _,
    ) = solve_poisson_dirichlet_multielement_2d(
        num_elements_x=num_elements,
        num_elements_y=num_elements,
        order_x=4,
        order_y=4,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=source_function,
        boundary_function=zero_boundary,
    )

    exact = exact_solution(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    relative_error = np.sqrt(np.sum((phi - exact) ** 2) / np.sum(exact**2))

    assert relative_error < 2.0e-2


def test_multielement_linear_harmonic_solution():

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(x)

    def exact_linear(
        x,
        y,
    ):
        return 1.0 + 2.0 * x - 3.0 * y

    (
        coordinates,
        phi,
        _,
    ) = solve_poisson_dirichlet_multielement_2d(
        num_elements_x=3,
        num_elements_y=2,
        order_x=4,
        order_y=4,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=zero_source,
        boundary_function=exact_linear,
    )

    exact = exact_linear(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-11,
    )
