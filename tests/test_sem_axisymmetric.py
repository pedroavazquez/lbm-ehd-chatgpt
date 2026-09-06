import numpy as np
import pytest

from lbm.electrostatics.sem.axisymmetric import (
    axisymmetric_element_2d,
)
from lbm.electrostatics.sem.axisymmetric_poisson import (
    solve_axisymmetric_dirichlet_2d,
)


def constant_coefficient(
    z,
    r,
):
    return np.ones_like(z)


def exact_solution(
    z,
    r,
):
    return np.sin(np.pi * z) * (1.0 - r**2)


def source_function(
    z,
    r,
):
    return (np.pi**2 * (1.0 - r**2) + 4.0) * np.sin(np.pi * z)


def test_axisymmetric_mass_integrates_volume_weight():

    coordinates, mass, _ = axisymmetric_element_2d(
        order_z=5,
        order_r=5,
        z_left=0.0,
        z_right=2.0,
        r_bottom=0.0,
        r_top=3.0,
        coefficient_function=constant_coefficient,
    )

    ones = np.ones(coordinates.shape[0])

    numerical = ones @ mass @ ones

    # Integral:
    #
    # int_0^2 dz int_0^3 r dr
    #
    # = 2 * 9/2 = 9.
    exact = 9.0

    assert np.isclose(
        numerical,
        exact,
        atol=1.0e-12,
    )


def test_axisymmetric_stiffness_symmetric():

    _, _, stiffness = axisymmetric_element_2d(
        order_z=5,
        order_r=5,
        z_left=0.0,
        z_right=1.0,
        r_bottom=0.0,
        r_top=1.0,
        coefficient_function=constant_coefficient,
    )

    assert np.allclose(
        stiffness,
        stiffness.T,
        atol=1.0e-12,
    )


@pytest.mark.parametrize(
    "order",
    [4, 6, 8],
)
def test_axisymmetric_poisson(
    order,
):

    (
        coordinates,
        phi,
        _,
    ) = solve_axisymmetric_dirichlet_2d(
        num_elements_z=2,
        num_elements_r=2,
        order_z=order,
        order_r=order,
        z_left=0.0,
        z_right=1.0,
        r_bottom=0.0,
        r_top=1.0,
        coefficient_function=constant_coefficient,
        source_function=source_function,
        boundary_function=exact_solution,
    )

    exact = exact_solution(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    # Axisymmetric weighted relative L2 norm.
    weights = coordinates[:, 1]

    error = np.sqrt(np.sum(weights * (phi - exact) ** 2) / np.sum(weights * exact**2))

    tolerance = {
        4: 2.0e-3,
        6: 2.0e-5,
        8: 2.0e-7,
    }[order]

    assert error < tolerance
