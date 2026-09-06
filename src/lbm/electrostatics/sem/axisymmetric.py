import numpy as np

from lbm.electrostatics.sem.differentiation import (
    differentiation_matrix,
)
from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def axisymmetric_element_2d(
    order_z: int,
    order_r: int,
    z_left: float,
    z_right: float,
    r_bottom: float,
    r_top: float,
    coefficient_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct one axisymmetric SEM element for

        -div_axi(kappa grad(phi)) = f

    in (z, r), with cylindrical weighting r.

    Parameters
    ----------
    order_z, order_r : int
        Polynomial orders.

    z_left, z_right : float
        Axial element limits.

    r_bottom, r_top : float
        Radial element limits.

    coefficient_function : callable
        kappa(z, r), real or complex.

    Returns
    -------
    coordinates : ndarray, shape (ndof, 2)
        (z, r) coordinates.

    mass : ndarray
        Axisymmetric weighted mass matrix.

    stiffness : ndarray
        Axisymmetric weighted stiffness matrix.
    """

    if z_right <= z_left:
        raise ValueError("z_right must be greater than z_left")

    if r_top <= r_bottom:
        raise ValueError("r_top must be greater than r_bottom")

    if r_bottom < 0.0:
        raise ValueError("radial coordinate must satisfy r >= 0")

    # ========================================================
    # 1-D GLL data
    # ========================================================

    xi, wz = gll_nodes_weights(order_z)

    eta, wr = gll_nodes_weights(order_r)

    Dz_ref = differentiation_matrix(xi)

    Dr_ref = differentiation_matrix(eta)

    hz = z_right - z_left

    hr = r_top - r_bottom

    Dz_1d = 2.0 / hz * Dz_ref

    Dr_1d = 2.0 / hr * Dr_ref

    # ========================================================
    # Physical nodes
    # ========================================================

    z_center = 0.5 * (z_left + z_right)

    r_center = 0.5 * (r_bottom + r_top)

    z_nodes = z_center + 0.5 * hz * xi

    r_nodes = r_center + 0.5 * hr * eta

    Z, R = np.meshgrid(
        z_nodes,
        r_nodes,
        indexing="xy",
    )

    coordinates = np.column_stack(
        (
            Z.ravel(),
            R.ravel(),
        )
    )

    nz_local = order_z + 1

    nr_local = order_r + 1

    ndof = nz_local * nr_local

    # ========================================================
    # Tensor derivative operators
    #
    # z varies fastest.
    # ========================================================

    Dz = np.kron(
        np.eye(nr_local),
        Dz_1d,
    )

    Dr = np.kron(
        Dr_1d,
        np.eye(nz_local),
    )

    # ========================================================
    # Physical quadrature
    # ========================================================

    quadrature_weights = (
        np.kron(
            wr,
            wz,
        )
        * hz
        * hr
        / 4.0
    )

    radial_coordinate = coordinates[:, 1]

    # Axisymmetric volume weight:
    #
    #     r dz dr
    #
    # 2*pi is omitted because it cancels from the equations.
    axisymmetric_weights = quadrature_weights * radial_coordinate

    mass = np.diag(axisymmetric_weights)

    # ========================================================
    # Coefficient
    # ========================================================

    kappa = np.asarray(
        coefficient_function(
            coordinates[:, 0],
            coordinates[:, 1],
        )
    )

    if kappa.ndim == 0:
        kappa = np.full(
            ndof,
            kappa,
        )

    if kappa.shape != (ndof,):
        raise ValueError(
            "coefficient_function must return a scalar or one value per SEM node"
        )

    weighted_coefficient = np.diag(axisymmetric_weights * kappa)

    # ========================================================
    # Weak-form stiffness
    # ========================================================

    stiffness = Dz.T @ weighted_coefficient @ Dz + Dr.T @ weighted_coefficient @ Dr

    return (
        coordinates,
        mass,
        stiffness,
    )


def assemble_axisymmetric_2d(
    num_elements_z: int,
    num_elements_r: int,
    order_z: int,
    order_r: int,
    z_left: float,
    z_right: float,
    r_bottom: float,
    r_top: float,
    coefficient_function,
    source_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Assemble the global axisymmetric SEM system for

        -div_axi(kappa grad(phi)) = f

    on a structured (z, r) mesh.
    """

    if num_elements_z < 1:
        raise ValueError("num_elements_z must be at least 1")

    if num_elements_r < 1:
        raise ValueError("num_elements_r must be at least 1")

    nz_global = num_elements_z * order_z + 1

    nr_global = num_elements_r * order_r + 1

    ndof = nz_global * nr_global

    sample_coefficient = np.asarray(
        coefficient_function(
            np.array([z_left]),
            np.array([max(r_bottom, 0.0)]),
        )
    )

    sample_source = np.asarray(
        source_function(
            np.array([z_left]),
            np.array([max(r_bottom, 0.0)]),
        )
    )

    dtype = np.result_type(
        sample_coefficient,
        sample_source,
        float,
    )

    coordinates = np.empty(
        (ndof, 2),
        dtype=float,
    )

    stiffness = np.zeros(
        (ndof, ndof),
        dtype=dtype,
    )

    rhs = np.zeros(
        ndof,
        dtype=dtype,
    )

    hz = (z_right - z_left) / num_elements_z

    hr = (r_top - r_bottom) / num_elements_r

    # ========================================================
    # Element loop
    # ========================================================

    for er in range(num_elements_r):
        ra = r_bottom + er * hr

        rb = ra + hr

        for ez in range(num_elements_z):
            za = z_left + ez * hz

            zb = za + hz

            (
                coordinates_local,
                mass_local,
                stiffness_local,
            ) = axisymmetric_element_2d(
                order_z=order_z,
                order_r=order_r,
                z_left=za,
                z_right=zb,
                r_bottom=ra,
                r_top=rb,
                coefficient_function=coefficient_function,
            )

            nz_local = order_z + 1

            nr_local = order_r + 1

            global_indices = np.empty(
                nz_local * nr_local,
                dtype=int,
            )

            for j in range(nr_local):
                J = er * order_r + j

                for i in range(nz_local):
                    I = ez * order_z + i

                    local_index = j * nz_local + i

                    global_index = J * nz_global + I

                    global_indices[local_index] = global_index

                    coordinates[global_index] = coordinates_local[local_index]

            source_local = np.asarray(
                source_function(
                    coordinates_local[:, 0],
                    coordinates_local[:, 1],
                )
            )

            if source_local.ndim == 0:
                source_local = np.full(
                    coordinates_local.shape[0],
                    source_local,
                )

            rhs_local = mass_local @ source_local

            rhs[global_indices] += rhs_local

            stiffness[
                np.ix_(
                    global_indices,
                    global_indices,
                )
            ] += stiffness_local

    return (
        coordinates,
        stiffness,
        rhs,
        (
            nz_global,
            nr_global,
        ),
    )
