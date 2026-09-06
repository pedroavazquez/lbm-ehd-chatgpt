import numpy as np
import pytest

from lbm.electrostatics.sem.differentiation import (
    differentiation_matrix,
)
from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def test_constant_derivative_is_zero():

    nodes, _ = gll_nodes_weights(5)

    D = differentiation_matrix(nodes)

    f = np.ones_like(nodes)

    df = D @ f

    assert np.allclose(
        df,
        0.0,
        atol=1.0e-14,
    )


def test_linear_function():

    nodes, _ = gll_nodes_weights(5)

    D = differentiation_matrix(nodes)

    f = nodes.copy()

    df = D @ f

    assert np.allclose(
        df,
        1.0,
        atol=1.0e-13,
    )


@pytest.mark.parametrize(
    "order",
    [2, 3, 4, 5, 6, 7, 8],
)
def test_polynomial_derivatives(order):

    nodes, _ = gll_nodes_weights(order)

    D = differentiation_matrix(nodes)

    # With p+1 interpolation nodes,
    # polynomials up to degree p are represented exactly.
    for degree in range(order + 1):
        f = nodes**degree

        if degree == 0:
            exact = np.zeros_like(nodes)

        else:
            exact = degree * nodes ** (degree - 1)

        numerical = D @ f

        assert np.allclose(
            numerical,
            exact,
            atol=1.0e-12,
        )


def test_rows_sum_to_zero():

    nodes, _ = gll_nodes_weights(6)

    D = differentiation_matrix(nodes)

    assert np.allclose(
        np.sum(
            D,
            axis=1,
        ),
        0.0,
        atol=1.0e-14,
    )
