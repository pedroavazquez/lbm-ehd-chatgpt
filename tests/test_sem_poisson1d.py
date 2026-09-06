import numpy as np
import pytest

from lbm.electrostatics.sem.poisson1d import (
    solve_poisson_dirichlet_single_element,
)


def exact_solution(
    x: np.ndarray,
) -> np.ndarray:

    return np.sin(np.pi * x)


def source_function(
    x: np.ndarray,
) -> np.ndarray:

    return np.pi**2 * np.sin(np.pi * x)


@pytest.mark.parametrize(
    "order",
    [4, 6, 8, 10],
)
def test_single_element_poisson(order):

    x, phi = solve_poisson_dirichlet_single_element(
        order=order,
        x_left=0.0,
        x_right=1.0,
        source_function=source_function,
        phi_left=0.0,
        phi_right=0.0,
    )

    phi_exact = exact_solution(x)

    error = np.sqrt(np.mean((phi - phi_exact) ** 2))

    # Spectral convergence should make the error
    # rapidly small as p increases.
    tolerance = {
        4: 5.0e-3,
        6: 5.0e-5,
        8: 5.0e-7,
        10: 5.0e-9,
    }[order]

    assert error < tolerance


def test_nonzero_dirichlet_conditions():

    # Exact solution:
    #
    # phi(x) = 1 + 2x
    #
    # so
    #
    # -phi'' = 0.
    def zero_source(x):
        return np.zeros_like(x)

    x, phi = solve_poisson_dirichlet_single_element(
        order=4,
        x_left=0.0,
        x_right=1.0,
        source_function=zero_source,
        phi_left=1.0,
        phi_right=3.0,
    )

    exact = 1.0 + 2.0 * x

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-13,
    )
