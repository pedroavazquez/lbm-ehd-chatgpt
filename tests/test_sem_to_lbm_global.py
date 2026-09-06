import numpy as np

from lbm.coupling.sem_to_lbm_global import (
    interpolate_global_sem_field_to_lbm,
    interpolate_global_sem_vector_to_lbm,
)
from lbm.electrostatics.sem.mesh2d import (
    assemble_poisson_2d,
)


def zero_source(
    x,
    y,
):
    return np.zeros_like(x)


def build_sem_coordinates(
    num_elements_x,
    num_elements_y,
    order_x,
    order_y,
):

    (
        coordinates,
        _,
        _,
        shape,
    ) = assemble_poisson_2d(
        num_elements_x=num_elements_x,
        num_elements_y=num_elements_y,
        order_x=order_x,
        order_y=order_y,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=zero_source,
    )

    nx_sem, ny_sem = shape

    X = coordinates[:, 0].reshape(
        ny_sem,
        nx_sem,
    )

    Y = coordinates[:, 1].reshape(
        ny_sem,
        nx_sem,
    )

    return X, Y


def lbm_cell_centres(
    num_elements_x,
    num_elements_y,
    cells_x,
    cells_y,
):

    nx = num_elements_x * cells_x

    ny = num_elements_y * cells_y

    x = (np.arange(nx) + 0.5) / nx

    y = (np.arange(ny) + 0.5) / ny

    return np.meshgrid(
        x,
        y,
        indexing="xy",
    )


def test_global_scalar_polynomial_interpolation():

    nex = 3
    ney = 2

    px = 4
    py = 4

    cells_x = 5
    cells_y = 6

    X_sem, Y_sem = build_sem_coordinates(
        nex,
        ney,
        px,
        py,
    )

    field_sem = (
        1.0 + 2.0 * X_sem - 3.0 * Y_sem + X_sem**2 + X_sem * Y_sem - 0.5 * Y_sem**3
    )

    field_lbm = interpolate_global_sem_field_to_lbm(
        field_sem=field_sem,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        lbm_cells_per_element_x=cells_x,
        lbm_cells_per_element_y=cells_y,
    )

    X_lbm, Y_lbm = lbm_cell_centres(
        nex,
        ney,
        cells_x,
        cells_y,
    )

    exact = 1.0 + 2.0 * X_lbm - 3.0 * Y_lbm + X_lbm**2 + X_lbm * Y_lbm - 0.5 * Y_lbm**3

    assert np.allclose(
        field_lbm,
        exact,
        atol=1.0e-12,
    )


def test_global_vector_interpolation():

    nex = 2
    ney = 3

    px = 5
    py = 5

    cells_x = 4
    cells_y = 5

    X_sem, Y_sem = build_sem_coordinates(
        nex,
        ney,
        px,
        py,
    )

    field_sem = np.empty(
        (
            2,
            Y_sem.shape[0],
            X_sem.shape[1],
        )
    )

    field_sem[0] = -2.0 * X_sem

    field_sem[1] = 2.0 * Y_sem

    field_lbm = interpolate_global_sem_vector_to_lbm(
        field_sem=field_sem,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        lbm_cells_per_element_x=cells_x,
        lbm_cells_per_element_y=cells_y,
    )

    X_lbm, Y_lbm = lbm_cell_centres(
        nex,
        ney,
        cells_x,
        cells_y,
    )

    assert np.allclose(
        field_lbm[0],
        -2.0 * X_lbm,
        atol=1.0e-12,
    )

    assert np.allclose(
        field_lbm[1],
        2.0 * Y_lbm,
        atol=1.0e-12,
    )


def test_global_complex_interpolation():

    nex = 2
    ney = 2

    px = 4
    py = 4

    cells_x = 6
    cells_y = 6

    X_sem, Y_sem = build_sem_coordinates(
        nex,
        ney,
        px,
        py,
    )

    amplitude = 1.0 + 0.4j

    field_sem = amplitude * (X_sem + 2.0 * Y_sem)

    field_lbm = interpolate_global_sem_field_to_lbm(
        field_sem=field_sem,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        lbm_cells_per_element_x=cells_x,
        lbm_cells_per_element_y=cells_y,
    )

    X_lbm, Y_lbm = lbm_cell_centres(
        nex,
        ney,
        cells_x,
        cells_y,
    )

    exact = amplitude * (X_lbm + 2.0 * Y_lbm)

    assert np.iscomplexobj(field_lbm)

    assert np.allclose(
        field_lbm,
        exact,
        atol=1.0e-12,
    )
