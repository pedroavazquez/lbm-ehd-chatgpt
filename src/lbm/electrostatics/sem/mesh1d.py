import numpy as np

from lbm.electrostatics.sem.element import (
    physical_element_matrices,
)


def assemble_poisson_1d(
    num_elements: int,
    order: int,
    x_left: float,
    x_right: float,
    source_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Assemble the global SEM stiffness matrix and load vector
    for

        -phi'' = f

    on a uniform 1-D mesh.

    Parameters
    ----------
    num_elements : int
        Number of spectral elements.

    order : int
        Polynomial order p within each element.

    x_left, x_right : float
        Global domain limits.

    source_function : callable
        Function f(x).

    Returns
    -------
    x_global : ndarray, shape (ndof,)
        Global GLL node coordinates.

    stiffness_global : ndarray, shape (ndof, ndof)
        Assembled global stiffness matrix.

    rhs_global : ndarray, shape (ndof,)
        Assembled global load vector.
    """

    if num_elements < 1:
        raise ValueError("num_elements must be at least 1")

    if order < 1:
        raise ValueError("order must be at least 1")

    if x_right <= x_left:
        raise ValueError("x_right must be greater than x_left")

    # --------------------------------------------------------
    # Number of global DOFs
    #
    # Each element has p+1 nodes, but neighboring elements
    # share one endpoint:
    #
    # ndof = ne * p + 1
    # --------------------------------------------------------

    ndof = num_elements * order + 1

    x_global = np.empty(
        ndof,
        dtype=float,
    )

    stiffness_global = np.zeros(
        (ndof, ndof),
        dtype=float,
    )

    rhs_global = np.zeros(
        ndof,
        dtype=float,
    )

    element_length = (x_right - x_left) / num_elements

    # ========================================================
    # Element loop
    # ========================================================

    for element in range(num_elements):
        xa = x_left + element * element_length

        xb = xa + element_length

        (
            x_local,
            mass_local,
            stiffness_local,
        ) = physical_element_matrices(
            order=order,
            x_left=xa,
            x_right=xb,
        )

        # ----------------------------------------------------
        # Local-to-global mapping
        #
        # Element e uses global DOFs:
        #
        #     e*p, ..., e*p+p
        # ----------------------------------------------------

        global_indices = element * order + np.arange(order + 1)

        # ----------------------------------------------------
        # Global coordinates
        # ----------------------------------------------------

        x_global[global_indices] = x_local

        # ----------------------------------------------------
        # Local RHS:
        #
        # b_e = M_e f_e
        # ----------------------------------------------------

        source_local = source_function(x_local)

        rhs_local = mass_local @ source_local

        # ----------------------------------------------------
        # Assemble matrix and RHS
        # ----------------------------------------------------

        for a_local, a_global in enumerate(global_indices):
            rhs_global[a_global] += rhs_local[a_local]

            for (
                b_local,
                b_global,
            ) in enumerate(global_indices):
                stiffness_global[
                    a_global,
                    b_global,
                ] += stiffness_local[
                    a_local,
                    b_local,
                ]

    return (
        x_global,
        stiffness_global,
        rhs_global,
    )
