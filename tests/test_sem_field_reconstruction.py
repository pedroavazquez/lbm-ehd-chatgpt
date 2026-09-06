import numpy as np

from lbm.electrostatics.sem.field_reconstruction import (
    reconstruct_electric_field_2d,
    reconstruct_gradient_2d,
)
from lbm.electrostatics.sem.mesh2d import (
    assemble_poisson_2d,
)


def zero_source(
    x,
    y,
):
    return np.zeros_like(x)


def get_coordinates(
    num_elements_x,
    num_elements_y,
    order_x,
    order_y,
):

    (
        coordinates,
        _,
        _,
        _,
    ) = assemble_poisson_2d(
        num_elements_x=num_elements_x,
        num_elements_y=num_elements_y,
        order_x=order_x,
        order_y=order_y,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
        source_function=zero_source,
    )

    return coordinates


def test_gradient_of_linear_function():

    nex = 3
    ney = 2

    px = 4
    py = 4

    coordinates = get_coordinates(
        nex,
        ney,
        px,
        py,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    phi = 2.0 * x - 3.0 * y + 1.5

    gradient = reconstruct_gradient_2d(
        phi=phi,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
    )

    assert np.allclose(
        gradient[0],
        2.0,
        atol=1.0e-12,
    )

    assert np.allclose(
        gradient[1],
        -3.0,
        atol=1.0e-12,
    )


def test_gradient_of_polynomial_function():

    nex = 3
    ney = 2

    px = 4
    py = 4

    coordinates = get_coordinates(
        nex,
        ney,
        px,
        py,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    # Polynomial represented exactly by p >= 2:
    #
    # phi = x^2 + 3xy - 2y^2
    phi = x**2 + 3.0 * x * y - 2.0 * y**2

    exact_dx = 2.0 * x + 3.0 * y

    exact_dy = 3.0 * x - 4.0 * y

    gradient = reconstruct_gradient_2d(
        phi=phi,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
    )

    assert np.allclose(
        gradient[0],
        exact_dx,
        atol=1.0e-11,
    )

    assert np.allclose(
        gradient[1],
        exact_dy,
        atol=1.0e-11,
    )


def test_electric_field_sign():

    nex = 2
    ney = 2

    px = 4
    py = 4

    coordinates = get_coordinates(
        nex,
        ney,
        px,
        py,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    phi = 2.0 * x - y

    electric_field = reconstruct_electric_field_2d(
        phi=phi,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
    )

    # E = -grad(phi)
    assert np.allclose(
        electric_field[0],
        -2.0,
        atol=1.0e-12,
    )

    assert np.allclose(
        electric_field[1],
        1.0,
        atol=1.0e-12,
    )


def test_complex_electric_field():

    nex = 2
    ney = 2

    px = 5
    py = 5

    coordinates = get_coordinates(
        nex,
        ney,
        px,
        py,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    amplitude = 1.0 + 2.0j

    phi = amplitude * (x + 2.0 * y)

    electric_field = reconstruct_electric_field_2d(
        phi=phi,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=px,
        order_y=py,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
    )

    assert np.iscomplexobj(electric_field)

    assert np.allclose(
        electric_field[0],
        -amplitude,
        atol=1.0e-12,
    )

    assert np.allclose(
        electric_field[1],
        -2.0 * amplitude,
        atol=1.0e-12,
    )


from lbm.electrostatics.sem.poisson2d_multielement import (
    solve_poisson_dirichlet_multielement_2d,
)


def test_electric_field_from_poisson_solution():

    num_elements = 2
    order = 8

    def exact_phi(
        x,
        y,
    ):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def source(
        x,
        y,
    ):
        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    def zero_boundary(
        x,
        y,
    ):
        return np.zeros_like(x)

    (
        coordinates,
        phi,
        _,
    ) = solve_poisson_dirichlet_multielement_2d(
        num_elements_x=num_elements,
        num_elements_y=num_elements,
        order_x=order,
        order_y=order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=source,
        boundary_function=zero_boundary,
    )

    electric_field = reconstruct_electric_field_2d(
        phi=phi,
        num_elements_x=num_elements,
        num_elements_y=num_elements,
        order_x=order,
        order_y=order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    exact_ex = -np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)

    exact_ey = -np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)

    relative_error = np.sqrt(
        (
            np.sum((electric_field[0] - exact_ex) ** 2)
            + np.sum((electric_field[1] - exact_ey) ** 2)
        )
        / (np.sum(exact_ex**2) + np.sum(exact_ey**2))
    )

    assert relative_error < 1.0e-5
