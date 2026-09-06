import numpy as np

from lbm.coupling.sem_to_lbm_global import (
    interpolate_global_sem_vector_to_lbm,
)
from lbm.electrohydrodynamics.force import (
    electric_force_density,
)
from lbm.electrostatics.sem.field_reconstruction import (
    reconstruct_electric_field_2d,
)
from lbm.electrostatics.sem.poisson2d_multielement import (
    solve_poisson_dirichlet_multielement_2d,
)


def test_spatially_varying_electric_force():

    nex = 2
    ney = 2

    order = 6

    cells_x = 8
    cells_y = 8

    nx = nex * cells_x

    ny = ney * cells_y

    # ========================================================
    # Exact electrostatic problem
    #
    # phi = x^2 - y^2
    #
    # Laplacian(phi) = 0
    # ========================================================

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(x)

    def exact_phi(
        x,
        y,
    ):
        return x**2 - y**2

    (
        _,
        phi,
        shape,
    ) = solve_poisson_dirichlet_multielement_2d(
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=order,
        order_y=order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        source_function=zero_source,
        boundary_function=exact_phi,
    )

    # ========================================================
    # Spectral electric field
    # ========================================================

    e_sem_flat = reconstruct_electric_field_2d(
        phi=phi,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=order,
        order_y=order,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
    )

    nx_sem, ny_sem = shape

    e_sem = np.empty(
        (
            2,
            ny_sem,
            nx_sem,
        )
    )

    e_sem[0] = e_sem_flat[0].reshape(
        ny_sem,
        nx_sem,
    )

    e_sem[1] = e_sem_flat[1].reshape(
        ny_sem,
        nx_sem,
    )

    # ========================================================
    # Interpolate E to LBM
    # ========================================================

    e_lbm_yx = interpolate_global_sem_vector_to_lbm(
        field_sem=e_sem,
        num_elements_x=nex,
        num_elements_y=ney,
        order_x=order,
        order_y=order,
        lbm_cells_per_element_x=cells_x,
        lbm_cells_per_element_y=cells_y,
    )

    # Convert from (2, ny, nx) to LBM storage (2, nx, ny).
    e_lbm = np.empty((2, nx, ny))

    e_lbm[0] = e_lbm_yx[0].T

    e_lbm[1] = e_lbm_yx[1].T

    # ========================================================
    # Exact field at LBM cell centres
    # ========================================================

    x = (np.arange(nx) + 0.5) / nx

    y = (np.arange(ny) + 0.5) / ny

    X, Y = np.meshgrid(
        x,
        y,
        indexing="ij",
    )

    exact_ex = -2.0 * X

    exact_ey = 2.0 * Y

    assert np.allclose(
        e_lbm[0],
        exact_ex,
        atol=1.0e-10,
    )

    assert np.allclose(
        e_lbm[1],
        exact_ey,
        atol=1.0e-10,
    )

    # ========================================================
    # Electric force
    # ========================================================

    charge0 = 0.3

    charge = charge0 * np.ones((nx, ny))

    force = electric_force_density(
        charge,
        e_lbm,
    )

    assert np.allclose(
        force[0],
        charge0 * exact_ex,
        atol=1.0e-10,
    )

    assert np.allclose(
        force[1],
        charge0 * exact_ey,
        atol=1.0e-10,
    )
