import numpy as np

from lbm.electrostatics.sem.element2d import (
    physical_element_matrices_2d,
)


def test_2d_matrix_shapes():

    order_x = 4
    order_y = 3

    coordinates, mass, stiffness, shape = physical_element_matrices_2d(
        order_x=order_x,
        order_y=order_y,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
    )

    ndof = (order_x + 1) * (order_y + 1)

    assert coordinates.shape == (
        ndof,
        2,
    )

    assert mass.shape == (
        ndof,
        ndof,
    )

    assert stiffness.shape == (
        ndof,
        ndof,
    )

    assert np.array_equal(
        shape,
        [order_x + 1, order_y + 1],
    )


def test_2d_mass_integrates_constant():

    x_length = 2.5
    y_length = 1.7

    _, mass, _, _ = physical_element_matrices_2d(
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=x_length,
        y_bottom=0.0,
        y_top=y_length,
    )

    ones = np.ones(mass.shape[0])

    integral = ones @ mass @ ones

    exact_area = x_length * y_length

    assert np.isclose(
        integral,
        exact_area,
    )


def test_2d_stiffness_is_symmetric():

    _, _, stiffness, _ = physical_element_matrices_2d(
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    assert np.allclose(
        stiffness,
        stiffness.T,
        atol=1.0e-12,
    )


def test_2d_stiffness_annihilates_constant():

    _, _, stiffness, _ = physical_element_matrices_2d(
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    ones = np.ones(stiffness.shape[0])

    assert np.allclose(
        stiffness @ ones,
        0.0,
        atol=1.0e-11,
    )


def test_linear_function_energy():

    coordinates, _, stiffness, _ = physical_element_matrices_2d(
        order_x=4,
        order_y=4,
        x_left=0.0,
        x_right=2.0,
        y_bottom=0.0,
        y_top=3.0,
    )

    x = coordinates[:, 0]

    # phi = x
    #
    # |grad phi|^2 = 1
    #
    # integral = area = 6.
    phi = x

    energy = phi @ stiffness @ phi

    assert np.isclose(
        energy,
        6.0,
        atol=1.0e-11,
    )
