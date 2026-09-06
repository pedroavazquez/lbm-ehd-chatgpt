import numpy as np

from lbm.electrostatics.sem.mesh2d import (
    assemble_poisson_2d,
)


def zero_source(
    x,
    y,
):
    return np.zeros_like(x)


def test_global_shape():

    coordinates, _, _, shape = assemble_poisson_2d(
        num_elements_x=3,
        num_elements_y=2,
        order_x=4,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=zero_source,
    )

    assert shape == (
        3 * 4 + 1,
        2 * 5 + 1,
    )

    assert coordinates.shape[0] == (shape[0] * shape[1])


def test_global_coordinates_cover_domain():

    coordinates, _, _, _ = assemble_poisson_2d(
        num_elements_x=2,
        num_elements_y=3,
        order_x=4,
        order_y=4,
        x_left=-1.0,
        x_right=2.0,
        y_bottom=-2.0,
        y_top=4.0,
        source_function=zero_source,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    assert np.isclose(
        np.min(x),
        -1.0,
    )

    assert np.isclose(
        np.max(x),
        2.0,
    )

    assert np.isclose(
        np.min(y),
        -2.0,
    )

    assert np.isclose(
        np.max(y),
        4.0,
    )


def test_global_stiffness_is_symmetric():

    _, stiffness, _, _ = assemble_poisson_2d(
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

    assert np.allclose(
        stiffness,
        stiffness.T,
        atol=1.0e-11,
    )


def test_global_stiffness_annihilates_constant():

    coordinates, stiffness, _, _ = assemble_poisson_2d(
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

    ones = np.ones(coordinates.shape[0])

    assert np.allclose(
        stiffness @ ones,
        0.0,
        atol=1.0e-10,
    )
