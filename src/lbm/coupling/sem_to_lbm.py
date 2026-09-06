import numpy as np

from lbm.electrostatics.sem.gll import gll_nodes_weights


def lagrange_interpolation_matrix(
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
) -> np.ndarray:
    """
    Construct the interpolation matrix from source interpolation
    nodes to arbitrary target nodes.

    Parameters
    ----------
    source_nodes : ndarray, shape (ns,)
        Interpolation nodes.

    target_nodes : ndarray, shape (nt,)
        Points at which the interpolating polynomial is evaluated.

    Returns
    -------
    P : ndarray, shape (nt, ns)

        target_values = P @ source_values
    """

    source_nodes = np.asarray(
        source_nodes,
        dtype=float,
    )

    target_nodes = np.asarray(
        target_nodes,
        dtype=float,
    )

    ns = source_nodes.size
    nt = target_nodes.size

    if ns < 2:
        raise ValueError("At least two source nodes are required")

    # --------------------------------------------------------
    # Barycentric weights
    # --------------------------------------------------------

    barycentric_weights = np.ones(
        ns,
        dtype=float,
    )

    for j in range(ns):
        differences = source_nodes[j] - np.delete(
            source_nodes,
            j,
        )

        barycentric_weights[j] = 1.0 / np.prod(differences)

    P = np.empty(
        (nt, ns),
        dtype=float,
    )

    # --------------------------------------------------------
    # Barycentric interpolation formula
    # --------------------------------------------------------

    for i, x in enumerate(target_nodes):
        differences = x - source_nodes

        # If a target point coincides with a source node,
        # interpolation is exact and should reduce to selecting
        # that nodal value.
        matching = np.where(
            np.isclose(
                differences,
                0.0,
                atol=1.0e-14,
                rtol=0.0,
            )
        )[0]

        if matching.size > 0:
            P[i] = 0.0
            P[i, matching[0]] = 1.0

        else:
            terms = barycentric_weights / differences

            P[i] = terms / np.sum(terms)

    return P


def element_interpolation_matrices(
    order_x: int,
    order_y: int,
    num_lbm_nodes_x: int,
    num_lbm_nodes_y: int,
    include_endpoints: bool = True,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct interpolation matrices from one SEM element
    to a uniform tensor-product LBM grid inside that element.

    The interpolation is constructed on the reference element
    [-1, 1]^2, so it can be reused for every element having
    the same polynomial order and number of LBM target nodes.

    Returns
    -------
    xi_lbm : ndarray
        Uniform target nodes in reference x coordinate.

    eta_lbm : ndarray
        Uniform target nodes in reference y coordinate.

    Px : ndarray
        x-direction interpolation matrix.

    Py : ndarray
        y-direction interpolation matrix.
    """

    if num_lbm_nodes_x < 1:
        raise ValueError("num_lbm_nodes_x must be positive")

    if num_lbm_nodes_y < 1:
        raise ValueError("num_lbm_nodes_y must be positive")

    xi_sem, _ = gll_nodes_weights(order_x)

    eta_sem, _ = gll_nodes_weights(order_y)

    if include_endpoints:
        xi_lbm = np.linspace(
            -1.0,
            1.0,
            num_lbm_nodes_x,
        )

        eta_lbm = np.linspace(
            -1.0,
            1.0,
            num_lbm_nodes_y,
        )

    else:
        # Uniform cell-centred points inside the element.
        xi_lbm = -1.0 + (
            np.arange(
                num_lbm_nodes_x,
                dtype=float,
            )
            + 0.5
        ) * (2.0 / num_lbm_nodes_x)

        eta_lbm = -1.0 + (
            np.arange(
                num_lbm_nodes_y,
                dtype=float,
            )
            + 0.5
        ) * (2.0 / num_lbm_nodes_y)

    Px = lagrange_interpolation_matrix(
        source_nodes=xi_sem,
        target_nodes=xi_lbm,
    )

    Py = lagrange_interpolation_matrix(
        source_nodes=eta_sem,
        target_nodes=eta_lbm,
    )

    return (
        xi_lbm,
        eta_lbm,
        Px,
        Py,
    )


def interpolate_element_field(
    field_sem: np.ndarray,
    Px: np.ndarray,
    Py: np.ndarray,
) -> np.ndarray:
    """
    Interpolate one tensor-product SEM field to an aligned
    uniform target grid.

    Parameters
    ----------
    field_sem : ndarray, shape (ny_sem, nx_sem)
        SEM nodal field on one element.

    Px : ndarray, shape (nx_lbm, nx_sem)
    Py : ndarray, shape (ny_lbm, ny_sem)

    Returns
    -------
    field_lbm : ndarray, shape (ny_lbm, nx_lbm)

        field_lbm = Py @ field_sem @ Px.T
    """

    expected_shape = (
        Py.shape[1],
        Px.shape[1],
    )

    if field_sem.shape != expected_shape:
        raise ValueError(f"field_sem must have shape {expected_shape}")

    return Py @ field_sem @ Px.T
