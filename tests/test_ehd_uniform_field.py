import numpy as np

from lbm.coupling.sem_to_lbm import (
    element_interpolation_matrices,
    interpolate_element_field,
)
from lbm.electrohydrodynamics.force import (
    electric_force_density,
)
from lbm.electrostatics.sem.field_reconstruction import (
    reconstruct_electric_field_2d,
)
from lbm.electrostatics.sem.poisson2d import (
    solve_poisson_dirichlet_single_element_2d,
)
from lbm.simulation import LBMSimulation


def test_uniform_electric_field_drives_uniform_acceleration():
    """
    End-to-end electrohydrodynamic coupling test.

    Electric problem
    ----------------
        phi(x,y) = -E0 x

    therefore

        E = (E0, 0).

    With uniform free charge density rho_e,

        F_e = rho_e E.

    Hydrodynamic response
    ---------------------
    In a periodic domain,

        du_x/dt = F_x / rho

    and therefore

        u_x(t) = a_x t.
    """

    # ========================================================
    # LBM grid
    # ========================================================

    nx = 16
    ny = 12

    rho0 = 1.0
    tau = 0.8

    # ========================================================
    # Electric problem
    # ========================================================

    e0 = 2.0e-6

    charge0 = 0.5

    sem_order = 6

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(x)

    def potential_exact(
        x,
        y,
    ):
        return -e0 * x

    # --------------------------------------------------------
    # Solve electrostatic problem on one SEM element.
    # --------------------------------------------------------

    (
        _,
        phi_sem,
    ) = solve_poisson_dirichlet_single_element_2d(
        order_x=sem_order,
        order_y=sem_order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=zero_source,
        boundary_function=potential_exact,
    )

    # --------------------------------------------------------
    # Reconstruct SEM electric field.
    # --------------------------------------------------------

    electric_field_sem = reconstruct_electric_field_2d(
        phi=phi_sem,
        num_elements_x=1,
        num_elements_y=1,
        order_x=sem_order,
        order_y=sem_order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    nx_sem = sem_order + 1
    ny_sem = sem_order + 1

    # Global SEM flattening has x varying fastest.
    ex_sem = electric_field_sem[0].reshape(
        ny_sem,
        nx_sem,
    )

    ey_sem = electric_field_sem[1].reshape(
        ny_sem,
        nx_sem,
    )

    # ========================================================
    # SEM -> LBM interpolation
    #
    # Use cell-centred target points inside the SEM element.
    # ========================================================

    (
        _,
        _,
        px,
        py,
    ) = element_interpolation_matrices(
        order_x=sem_order,
        order_y=sem_order,
        num_lbm_nodes_x=nx,
        num_lbm_nodes_y=ny,
        include_endpoints=False,
    )

    ex_lbm_yx = interpolate_element_field(
        field_sem=ex_sem,
        Px=px,
        Py=py,
    )

    ey_lbm_yx = interpolate_element_field(
        field_sem=ey_sem,
        Px=px,
        Py=py,
    )

    # LBM code stores fields as (nx, ny),
    # whereas interpolation returns (ny, nx).
    electric_field_lbm = np.empty((2, nx, ny))

    electric_field_lbm[0] = ex_lbm_yx.T

    electric_field_lbm[1] = ey_lbm_yx.T

    # Verify electric field before coupling.
    assert np.allclose(
        electric_field_lbm[0],
        e0,
        atol=1.0e-14,
    )

    assert np.allclose(
        electric_field_lbm[1],
        0.0,
        atol=1.0e-14,
    )

    # ========================================================
    # Electric body force
    # ========================================================

    charge_density = charge0 * np.ones((nx, ny))

    force = electric_force_density(
        charge_density,
        electric_field_lbm,
    )

    expected_acceleration = charge0 * e0 / rho0

    # ========================================================
    # LBM
    # ========================================================

    sim = LBMSimulation(
        nx=nx,
        ny=ny,
        tau=tau,
        rho0=rho0,
    )

    sim.set_force(force)

    rho_initial = rho0 * np.ones((nx, ny))

    u_initial = np.zeros((2, nx, ny))

    sim.set_state(
        rho_initial,
        u_initial,
    )

    nsteps = 100

    for _ in range(nsteps):
        sim.step()

    _, u = sim.macroscopic()

    expected_velocity = expected_acceleration * nsteps

    assert np.allclose(
        u[0],
        expected_velocity,
        rtol=1.0e-9,
        atol=1.0e-12,
    )

    assert np.allclose(
        u[1],
        0.0,
        atol=1.0e-12,
    )
