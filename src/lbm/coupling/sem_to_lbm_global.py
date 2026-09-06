import numpy as np

from lbm.coupling.sem_to_lbm import (
    element_interpolation_matrices,
    interpolate_element_field,
)


def interpolate_global_sem_field_to_lbm(
    field_sem: np.ndarray,
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    lbm_cells_per_element_x: int,
    lbm_cells_per_element_y: int,
) -> np.ndarray:
    """
    Interpolate a global structured SEM nodal field onto an aligned
    cell-centred LBM grid.

    Parameters
    ----------
    field_sem : ndarray, shape (ny_sem_global, nx_sem_global)
        Global SEM nodal field.

    num_elements_x, num_elements_y : int
        Number of SEM elements.

    order_x, order_y : int
        Polynomial orders.

    lbm_cells_per_element_x, lbm_cells_per_element_y : int
        Number of LBM cells covered by each SEM element.

    Returns
    -------
    field_lbm : ndarray, shape (ny_lbm, nx_lbm)
        Interpolated field at cell-centred LBM nodes.
    """

    nx_sem_global = num_elements_x * order_x + 1

    ny_sem_global = num_elements_y * order_y + 1

    expected_shape = (
        ny_sem_global,
        nx_sem_global,
    )

    if field_sem.shape != expected_shape:
        raise ValueError(f"field_sem must have shape {expected_shape}")

    nx_lbm = num_elements_x * lbm_cells_per_element_x

    ny_lbm = num_elements_y * lbm_cells_per_element_y

    field_lbm = np.empty(
        (ny_lbm, nx_lbm),
        dtype=np.result_type(
            field_sem,
            float,
        ),
    )

    (
        _,
        _,
        Px,
        Py,
    ) = element_interpolation_matrices(
        order_x=order_x,
        order_y=order_y,
        num_lbm_nodes_x=lbm_cells_per_element_x,
        num_lbm_nodes_y=lbm_cells_per_element_y,
        include_endpoints=False,
    )

    nx_local = order_x + 1

    ny_local = order_y + 1

    for ey in range(num_elements_y):
        for ex in range(num_elements_x):
            # ------------------------------------------------
            # Extract SEM nodal field for this element.
            # ------------------------------------------------

            i0_sem = ex * order_x

            j0_sem = ey * order_y

            field_local = field_sem[
                j0_sem : j0_sem + ny_local,
                i0_sem : i0_sem + nx_local,
            ]

            # ------------------------------------------------
            # Interpolate to local LBM cell centres.
            # ------------------------------------------------

            field_local_lbm = interpolate_element_field(
                field_sem=field_local,
                Px=Px,
                Py=Py,
            )

            # ------------------------------------------------
            # Place in global LBM array.
            # ------------------------------------------------

            i0_lbm = ex * lbm_cells_per_element_x

            j0_lbm = ey * lbm_cells_per_element_y

            field_lbm[
                j0_lbm : (j0_lbm + lbm_cells_per_element_y),
                i0_lbm : (i0_lbm + lbm_cells_per_element_x),
            ] = field_local_lbm

    return field_lbm


def interpolate_global_sem_vector_to_lbm(
    field_sem: np.ndarray,
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    lbm_cells_per_element_x: int,
    lbm_cells_per_element_y: int,
) -> np.ndarray:
    """
    Interpolate a 2-component global SEM vector field.

    Parameters
    ----------
    field_sem : ndarray, shape (2, ny_sem_global, nx_sem_global)

    Returns
    -------
    field_lbm : ndarray, shape (2, ny_lbm, nx_lbm)
    """

    if field_sem.ndim != 3:
        raise ValueError("field_sem must have shape (2, ny_sem_global, nx_sem_global)")

    if field_sem.shape[0] != 2:
        raise ValueError("field_sem must have two components")

    components = []

    for component in range(2):
        components.append(
            interpolate_global_sem_field_to_lbm(
                field_sem=field_sem[component],
                num_elements_x=num_elements_x,
                num_elements_y=num_elements_y,
                order_x=order_x,
                order_y=order_y,
                lbm_cells_per_element_x=(lbm_cells_per_element_x),
                lbm_cells_per_element_y=(lbm_cells_per_element_y),
            )
        )

    return np.stack(
        components,
        axis=0,
    )
