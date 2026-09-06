import numpy as np

from lbm.electrostatics.sem.mesh1d import (
    assemble_poisson_1d,
)


def zero_source(
    x,
):
    return np.zeros_like(x)


def test_global_number_of_dofs():

    ne = 4
    p = 5

    x, _, _ = assemble_poisson_1d(
        num_elements=ne,
        order=p,
        x_left=0.0,
        x_right=1.0,
        source_function=zero_source,
    )

    assert x.size == (ne * p + 1)


def test_global_endpoints():

    x, _, _ = assemble_poisson_1d(
        num_elements=3,
        order=4,
        x_left=-2.0,
        x_right=3.0,
        source_function=zero_source,
    )

    assert np.isclose(
        x[0],
        -2.0,
    )

    assert np.isclose(
        x[-1],
        3.0,
    )


def test_global_coordinates_increase():

    x, _, _ = assemble_poisson_1d(
        num_elements=4,
        order=5,
        x_left=0.0,
        x_right=1.0,
        source_function=zero_source,
    )

    assert np.all(np.diff(x) > 0.0)


def test_global_stiffness_is_symmetric():

    _, stiffness, _ = assemble_poisson_1d(
        num_elements=3,
        order=5,
        x_left=0.0,
        x_right=1.0,
        source_function=zero_source,
    )

    assert np.allclose(
        stiffness,
        stiffness.T,
        atol=1.0e-12,
    )


def test_global_stiffness_annihilates_constant():

    x, stiffness, _ = assemble_poisson_1d(
        num_elements=3,
        order=5,
        x_left=0.0,
        x_right=1.0,
        source_function=zero_source,
    )

    ones = np.ones_like(x)

    assert np.allclose(
        stiffness @ ones,
        0.0,
        atol=1.0e-11,
    )
