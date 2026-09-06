import numpy as np
import pytest

from lbm.electrostatics.sem.poisson1d_multielement import (
    solve_poisson_dirichlet_multielement,
)


def exact_solution(
    x,
):
    return np.sin(np.pi * x)


def source_function(
    x,
):
    return np.pi**2 * np.sin(np.pi * x)


@pytest.mark.parametrize(
    "num_elements",
    [1, 2, 4, 8],
)
def test_multielement_poisson(
    num_elements,
):

    x, phi = solve_poisson_dirichlet_multielement(
        num_elements=num_elements,
        order=4,
        x_left=0.0,
        x_right=1.0,
        source_function=source_function,
        phi_left=0.0,
        phi_right=0.0,
    )

    exact = exact_solution(x)

    error = np.sqrt(np.mean((phi - exact) ** 2))

    assert error < 1.0e-2


def test_multielement_linear_solution():

    def source(x):
        return np.zeros_like(x)

    x, phi = solve_poisson_dirichlet_multielement(
        num_elements=4,
        order=4,
        x_left=0.0,
        x_right=1.0,
        source_function=source,
        phi_left=1.0,
        phi_right=3.0,
    )

    exact = 1.0 + 2.0 * x

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-12,
    )
