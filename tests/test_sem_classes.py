import numpy as np

from lbm.electrostatics.sem.boundary_conditions import (
    DirichletBC,
    NeumannBC,
)
from lbm.electrostatics.sem.mesh import (
    StructuredSEMMesh2D,
)
from lbm.electrostatics.sem.solver import (
    ElectrostaticSolver2D,
)


def test_mesh_properties():

    mesh = StructuredSEMMesh2D(
        num_elements_x=3,
        num_elements_y=2,
        order_x=4,
        order_y=5,
        x_left=0.0,
        x_right=2.0,
        y_bottom=-1.0,
        y_top=1.0,
    )

    assert mesh.global_shape == (
        13,
        11,
    )

    assert mesh.array_shape == (
        11,
        13,
    )

    assert mesh.ndof == (13 * 11)

    assert np.isclose(
        mesh.element_width,
        2.0 / 3.0,
    )

    assert np.isclose(
        mesh.element_height,
        1.0,
    )


def test_electrostatic_solver_class():

    mesh = StructuredSEMMesh2D(
        num_elements_x=2,
        num_elements_y=2,
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    def coefficient(
        x,
        y,
    ):
        return np.ones_like(x)

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(x)

    def left_voltage(
        x,
        y,
    ):
        return np.ones_like(x)

    def right_voltage(
        x,
        y,
    ):
        return np.zeros_like(x)

    def zero_flux(
        x,
        y,
    ):
        return np.zeros_like(x)

    bc = {
        "left": DirichletBC(left_voltage),
        "right": DirichletBC(right_voltage),
        "bottom": NeumannBC(zero_flux),
        "top": NeumannBC(zero_flux),
    }

    solver = ElectrostaticSolver2D(
        mesh=mesh,
        coefficient_function=coefficient,
        boundary_conditions=bc,
    )

    solver.solve(zero_source)

    electric_field = solver.reconstruct_electric_field()

    coordinates = solver.coordinates

    exact_phi = 1.0 - coordinates[:, 0]

    assert np.allclose(
        solver.potential,
        exact_phi,
        atol=1.0e-11,
    )

    assert np.allclose(
        electric_field[0],
        1.0,
        atol=1.0e-11,
    )

    assert np.allclose(
        electric_field[1],
        0.0,
        atol=1.0e-11,
    )
