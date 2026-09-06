import numpy as np
import pytest

from lbm.electrostatics.sem.element import (
    physical_element_matrices,
    reference_element_matrices,
)


def test_reference_mass_is_diagonal():

    _, weights, mass, _ = reference_element_matrices(5)

    assert np.allclose(
        mass,
        np.diag(weights),
    )


def test_reference_mass_integrates_constant():

    _, _, mass, _ = reference_element_matrices(5)

    ones = np.ones(mass.shape[0])

    integral = ones @ mass @ ones

    # Integral over [-1, 1].
    assert np.isclose(
        integral,
        2.0,
    )


def test_reference_stiffness_is_symmetric():

    _, _, _, stiffness = reference_element_matrices(6)

    assert np.allclose(
        stiffness,
        stiffness.T,
        atol=1.0e-13,
    )


def test_reference_stiffness_annihilates_constant():

    _, _, _, stiffness = reference_element_matrices(6)

    ones = np.ones(stiffness.shape[0])

    result = stiffness @ ones

    assert np.allclose(
        result,
        0.0,
        atol=1.0e-12,
    )


def test_reference_stiffness_energy_for_linear_function():

    nodes, _, _, stiffness = reference_element_matrices(5)

    # f(xi) = xi
    f = nodes

    energy = f @ stiffness @ f

    # Integral_{-1}^{1} (df/dxi)^2 dxi
    #
    # = integral 1 dxi = 2.
    assert np.isclose(
        energy,
        2.0,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "length",
    [0.5, 1.0, 2.0, 3.7],
)
def test_physical_mass_integrates_constant(
    length,
):

    x_left = 1.3
    x_right = x_left + length

    _, mass, _ = physical_element_matrices(
        order=5,
        x_left=x_left,
        x_right=x_right,
    )

    ones = np.ones(mass.shape[0])

    integral = ones @ mass @ ones

    assert np.isclose(
        integral,
        length,
    )


@pytest.mark.parametrize(
    "length",
    [0.5, 1.0, 2.0, 3.7],
)
def test_physical_linear_energy(
    length,
):

    x_left = -0.4
    x_right = x_left + length

    x, _, stiffness = physical_element_matrices(
        order=5,
        x_left=x_left,
        x_right=x_right,
    )

    # f(x) = x
    f = x

    energy = f @ stiffness @ f

    # Integral (df/dx)^2 dx
    #
    # = Integral 1 dx
    # = element length.
    assert np.isclose(
        energy,
        length,
        atol=1.0e-12,
    )


def test_physical_nodes_include_element_endpoints():

    x_left = 2.0
    x_right = 5.5

    x, _, _ = physical_element_matrices(
        order=4,
        x_left=x_left,
        x_right=x_right,
    )

    assert np.isclose(
        x[0],
        x_left,
    )

    assert np.isclose(
        x[-1],
        x_right,
    )


def test_invalid_element_interval():

    with pytest.raises(ValueError):
        physical_element_matrices(
            order=4,
            x_left=1.0,
            x_right=1.0,
        )
