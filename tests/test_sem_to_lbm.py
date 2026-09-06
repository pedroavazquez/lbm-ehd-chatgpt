import numpy as np
import pytest

from lbm.coupling.sem_to_lbm import (
    element_interpolation_matrices,
    interpolate_element_field,
    lagrange_interpolation_matrix,
)
from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def test_interpolation_matrix_identity_at_same_nodes():

    nodes, _ = gll_nodes_weights(5)

    P = lagrange_interpolation_matrix(
        source_nodes=nodes,
        target_nodes=nodes,
    )

    assert np.allclose(
        P,
        np.eye(nodes.size),
        atol=1.0e-14,
    )


@pytest.mark.parametrize(
    "order",
    [2, 3, 4, 5, 6],
)
def test_polynomial_interpolation_exact(order):

    source_nodes, _ = gll_nodes_weights(order)

    target_nodes = np.linspace(
        -1.0,
        1.0,
        21,
    )

    P = lagrange_interpolation_matrix(
        source_nodes=source_nodes,
        target_nodes=target_nodes,
    )

    for degree in range(order + 1):
        source_values = source_nodes**degree

        interpolated = P @ source_values

        exact = target_nodes**degree

        assert np.allclose(
            interpolated,
            exact,
            atol=1.0e-12,
        )


def test_tensor_product_interpolation():

    order_x = 4
    order_y = 4

    (
        xi_lbm,
        eta_lbm,
        Px,
        Py,
    ) = element_interpolation_matrices(
        order_x=order_x,
        order_y=order_y,
        num_lbm_nodes_x=11,
        num_lbm_nodes_y=9,
    )

    xi_sem, _ = gll_nodes_weights(order_x)

    eta_sem, _ = gll_nodes_weights(order_y)

    XI_sem, ETA_sem = np.meshgrid(
        xi_sem,
        eta_sem,
        indexing="xy",
    )

    # Polynomial exactly representable by p=4.
    field_sem = (
        1.0
        + 2.0 * XI_sem
        - 3.0 * ETA_sem
        + XI_sem**2
        + 2.0 * XI_sem * ETA_sem
        - ETA_sem**3
    )

    field_lbm = interpolate_element_field(
        field_sem=field_sem,
        Px=Px,
        Py=Py,
    )

    XI_lbm, ETA_lbm = np.meshgrid(
        xi_lbm,
        eta_lbm,
        indexing="xy",
    )

    exact = (
        1.0
        + 2.0 * XI_lbm
        - 3.0 * ETA_lbm
        + XI_lbm**2
        + 2.0 * XI_lbm * ETA_lbm
        - ETA_lbm**3
    )

    assert np.allclose(
        field_lbm,
        exact,
        atol=1.0e-12,
    )


def test_complex_tensor_product_interpolation():

    (
        xi_lbm,
        eta_lbm,
        Px,
        Py,
    ) = element_interpolation_matrices(
        order_x=5,
        order_y=5,
        num_lbm_nodes_x=13,
        num_lbm_nodes_y=10,
    )

    xi_sem, _ = gll_nodes_weights(5)

    eta_sem, _ = gll_nodes_weights(5)

    XI_sem, ETA_sem = np.meshgrid(
        xi_sem,
        eta_sem,
        indexing="xy",
    )

    amplitude = 1.0 + 0.7j

    field_sem = amplitude * (1.0 + XI_sem + 2.0 * ETA_sem**2)

    field_lbm = interpolate_element_field(
        field_sem=field_sem,
        Px=Px,
        Py=Py,
    )

    XI_lbm, ETA_lbm = np.meshgrid(
        xi_lbm,
        eta_lbm,
        indexing="xy",
    )

    exact = amplitude * (1.0 + XI_lbm + 2.0 * ETA_lbm**2)

    assert np.iscomplexobj(field_lbm)

    assert np.allclose(
        field_lbm,
        exact,
        atol=1.0e-12,
    )


def test_cell_centred_target_nodes():

    (
        xi_lbm,
        eta_lbm,
        _,
        _,
    ) = element_interpolation_matrices(
        order_x=4,
        order_y=4,
        num_lbm_nodes_x=4,
        num_lbm_nodes_y=2,
        include_endpoints=False,
    )

    assert np.allclose(
        xi_lbm,
        [
            -0.75,
            -0.25,
            0.25,
            0.75,
        ],
    )

    assert np.allclose(
        eta_lbm,
        [
            -0.5,
            0.5,
        ],
    )
