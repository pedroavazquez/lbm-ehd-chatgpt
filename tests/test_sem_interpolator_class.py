import numpy as np

from lbm.coupling.interpolator import (
    SEMToLBMInterpolator,
)
from lbm.electrostatics.sem.mesh import (
    StructuredSEMMesh2D,
)
from lbm.electrostatics.sem.mesh2d import (
    assemble_poisson_2d,
)


def zero_source(
    x,
    y,
):
    return np.zeros_like(x)


def test_interpolator_class():

    mesh = StructuredSEMMesh2D(
        num_elements_x=2,
        num_elements_y=3,
        order_x=4,
        order_y=4,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    (
        coordinates,
        _,
        _,
        _,
    ) = assemble_poisson_2d(
        num_elements_x=mesh.num_elements_x,
        num_elements_y=mesh.num_elements_y,
        order_x=mesh.order_x,
        order_y=mesh.order_y,
        x_left=mesh.x_left,
        x_right=mesh.x_right,
        y_bottom=mesh.y_bottom,
        y_top=mesh.y_top,
        source_function=zero_source,
    )

    X = coordinates[:, 0].reshape(mesh.array_shape)

    Y = coordinates[:, 1].reshape(mesh.array_shape)

    field_sem = X**2 + 2.0 * Y

    interpolator = SEMToLBMInterpolator(
        mesh=mesh,
        lbm_cells_per_element_x=5,
        lbm_cells_per_element_y=4,
    )

    field_lbm = interpolator.interpolate_scalar(field_sem)

    nx = interpolator.nx_lbm
    ny = interpolator.ny_lbm

    x = (np.arange(nx) + 0.5) / nx

    y = (np.arange(ny) + 0.5) / ny

    X_lbm, Y_lbm = np.meshgrid(
        x,
        y,
        indexing="xy",
    )

    exact = X_lbm**2 + 2.0 * Y_lbm

    assert np.allclose(
        field_lbm,
        exact,
        atol=1.0e-12,
    )
