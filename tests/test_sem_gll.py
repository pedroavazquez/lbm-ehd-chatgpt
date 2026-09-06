import numpy as np
import pytest

from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def test_gll_order_one():

    nodes, weights = gll_nodes_weights(1)

    assert np.allclose(
        nodes,
        [-1.0, 1.0],
    )

    assert np.allclose(
        weights,
        [1.0, 1.0],
    )


def test_gll_endpoints():

    for p in range(1, 9):
        nodes, _ = gll_nodes_weights(p)

        assert np.isclose(
            nodes[0],
            -1.0,
        )

        assert np.isclose(
            nodes[-1],
            1.0,
        )


def test_gll_symmetry():

    for p in range(1, 9):
        nodes, weights = gll_nodes_weights(p)

        assert np.allclose(
            nodes,
            -nodes[::-1],
        )

        assert np.allclose(
            weights,
            weights[::-1],
        )


def test_gll_weights_sum_to_two():

    for p in range(1, 9):
        _, weights = gll_nodes_weights(p)

        assert np.isclose(
            np.sum(weights),
            2.0,
        )


@pytest.mark.parametrize(
    "order",
    [2, 3, 4, 5, 6, 7],
)
def test_gll_polynomial_exactness(order):

    nodes, weights = gll_nodes_weights(order)

    # GLL quadrature with p+1 nodes
    # integrates polynomials up to degree
    #
    #     2p - 1
    #
    # exactly.

    max_degree = 2 * order - 1

    for degree in range(max_degree + 1):
        numerical = np.sum(weights * nodes**degree)

        if degree % 2 == 0:
            exact = 2.0 / (degree + 1)

        else:
            exact = 0.0

        assert np.isclose(
            numerical,
            exact,
            atol=1.0e-13,
        )
