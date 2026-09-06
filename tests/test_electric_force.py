import numpy as np
import pytest

from lbm.electrohydrodynamics.force import (
    electric_force_density,
)


def test_uniform_electric_force():

    nx = 8
    ny = 6

    charge = 2.0 * np.ones((nx, ny))

    electric_field = np.zeros((2, nx, ny))

    electric_field[0] = 3.0
    electric_field[1] = -1.0

    force = electric_force_density(
        charge,
        electric_field,
    )

    assert np.allclose(
        force[0],
        6.0,
    )

    assert np.allclose(
        force[1],
        -2.0,
    )


def test_force_shape_validation():

    charge = np.ones((4, 5))

    field = np.zeros((2, 5, 4))

    with pytest.raises(ValueError):
        electric_force_density(
            charge,
            field,
        )
