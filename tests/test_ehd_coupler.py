import numpy as np

from lbm.coupling.interpolator import (
    SEMToLBMInterpolator,
)
from lbm.electrohydrodynamics.coupler import (
    EHDCoupler,
)
from lbm.electrostatics.sem.boundary_conditions import (
    DirichletBC,
)
from lbm.electrostatics.sem.mesh import (
    StructuredSEMMesh2D,
)
from lbm.electrostatics.sem.solver import (
    ElectrostaticSolver2D,
)
from lbm.simulation import (
    LBMSimulation,
)


def test_class_based_ehd_coupling():

    # ========================================================
    # SEM mesh
    # ========================================================

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

    cells_per_element = 8

    interpolator = SEMToLBMInterpolator(
        mesh=mesh,
        lbm_cells_per_element_x=(cells_per_element),
        lbm_cells_per_element_y=(cells_per_element),
    )

    # ========================================================
    # Electric problem
    #
    # phi = -E0 x
    # ========================================================

    e0 = 2.0e-6

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

    def exact_potential(
        x,
        y,
    ):
        return -e0 * x

    bc = {
        "left": DirichletBC(exact_potential),
        "right": DirichletBC(exact_potential),
        "bottom": DirichletBC(exact_potential),
        "top": DirichletBC(exact_potential),
    }

    electrostatics = ElectrostaticSolver2D(
        mesh=mesh,
        coefficient_function=coefficient,
        boundary_conditions=bc,
    )

    # ========================================================
    # Hydrodynamics
    # ========================================================

    rho0 = 1.0

    hydro = LBMSimulation(
        nx=interpolator.nx_lbm,
        ny=interpolator.ny_lbm,
        tau=0.8,
        rho0=rho0,
    )

    # ========================================================
    # Coupler
    # ========================================================

    coupler = EHDCoupler(
        hydrodynamics=hydro,
        electrostatics=electrostatics,
        interpolator=interpolator,
    )

    coupler.solve_electric_problem(zero_source)

    electric_field = coupler.interpolate_electric_field()

    assert np.allclose(
        electric_field[0],
        e0,
        atol=1.0e-13,
    )

    assert np.allclose(
        electric_field[1],
        0.0,
        atol=1.0e-13,
    )

    # ========================================================
    # Charge and force
    # ========================================================

    charge0 = 0.5

    charge = charge0 * np.ones(
        (
            hydro.nx,
            hydro.ny,
        )
    )

    force = coupler.update_electric_force(charge)

    acceleration = charge0 * e0 / rho0

    assert np.allclose(
        force[0],
        charge0 * e0,
    )

    # ========================================================
    # Hydrodynamic response
    # ========================================================

    nsteps = 100

    coupler.advance_hydrodynamics(nsteps=nsteps)

    _, u = hydro.macroscopic()

    assert np.allclose(
        u[0],
        acceleration * nsteps,
        rtol=1.0e-9,
        atol=1.0e-12,
    )

    assert np.allclose(
        u[1],
        0.0,
        atol=1.0e-12,
    )
