import numpy as np

from lbm.electrostatics.sem.differentiation import (
    differentiation_matrix,
)
from lbm.electrostatics.sem.gll import (
    gll_nodes_weights,
)


def variable_coefficient_element_2d(
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    coefficient_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Construct the local SEM stiffness matrix for

        -div(kappa grad(phi))

    on one rectangular spectral element.

    The coefficient kappa(x, y) is evaluated at the GLL nodes.

    Returns
    -------
    coordinates : ndarray, shape (ndof, 2)
        Physical GLL coordinates.

    mass : ndarray, shape (ndof, ndof)
        Standard 2-D quadrature mass matrix.

    stiffness : ndarray, shape (ndof, ndof)
        Variable-coefficient stiffness matrix.
    """

    if x_right <= x_left:
        raise ValueError("x_right must be greater than x_left")

    if y_top <= y_bottom:
        raise ValueError("y_top must be greater than y_bottom")

    # ========================================================
    # 1-D GLL data
    # ========================================================

    xi, wx = gll_nodes_weights(order_x)

    eta, wy = gll_nodes_weights(order_y)

    Dx_ref = differentiation_matrix(xi)

    Dy_ref = differentiation_matrix(eta)

    hx = x_right - x_left

    hy = y_top - y_bottom

    # Physical differentiation matrices:
    #
    # d/dx = 2/hx d/dxi
    # d/dy = 2/hy d/deta
    Dx_1d = 2.0 / hx * Dx_ref

    Dy_1d = 2.0 / hy * Dy_ref

    # ========================================================
    # Physical coordinates
    # ========================================================

    x_center = 0.5 * (x_left + x_right)

    y_center = 0.5 * (y_bottom + y_top)

    x_nodes = x_center + 0.5 * hx * xi

    y_nodes = y_center + 0.5 * hy * eta

    X, Y = np.meshgrid(
        x_nodes,
        y_nodes,
        indexing="xy",
    )

    coordinates = np.column_stack(
        (
            X.ravel(),
            Y.ravel(),
        )
    )

    nx_local = order_x + 1

    ny_local = order_y + 1

    ndof = nx_local * ny_local

    # ========================================================
    # Tensor-product derivative operators
    #
    # Flattening convention:
    # x varies fastest.
    # ========================================================

    Dx = np.kron(
        np.eye(ny_local),
        Dx_1d,
    )

    Dy = np.kron(
        Dy_1d,
        np.eye(nx_local),
    )

    # ========================================================
    # Physical quadrature weights
    #
    # Jacobian:
    #
    # J = hx hy / 4
    # ========================================================

    weights_2d = (
        np.kron(
            wy,
            wx,
        )
        * hx
        * hy
        / 4.0
    )

    mass = np.diag(weights_2d)

    # ========================================================
    # Variable coefficient
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

    weighted_coefficient = np.diag(weights_2d * kappa)

    # ========================================================
    # Weak-form stiffness matrix
    #
    # K =
    #   Dx^T W_kappa Dx
    # + Dy^T W_kappa Dy
    # ========================================================

    stiffness = Dx.T @ weighted_coefficient @ Dx + Dy.T @ weighted_coefficient @ Dy

    return (
        coordinates,
        mass,
        stiffness,
    )


def assemble_variable_coefficient_2d(
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    coefficient_function,
    source_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Assemble the global system for

        -div(kappa grad(phi)) = f

    on a structured rectangular SEM mesh.
    """

    if num_elements_x < 1:
        raise ValueError("num_elements_x must be at least 1")

    if num_elements_y < 1:
        raise ValueError("num_elements_y must be at least 1")

    nx_global = num_elements_x * order_x + 1

    ny_global = num_elements_y * order_y + 1

    ndof = nx_global * ny_global

    # Infer numerical type already here.
    sample_coefficient = np.asarray(
        coefficient_function(
            np.array([x_left]),
            np.array([y_bottom]),
        )
    )

    sample_source = np.asarray(
        source_function(
            np.array([x_left]),
            np.array([y_bottom]),
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

    hx = (x_right - x_left) / num_elements_x

    hy = (y_top - y_bottom) / num_elements_y

    # ========================================================
    # Element assembly
    # ========================================================

    for ey in range(num_elements_y):
        ya = y_bottom + ey * hy

        yb = ya + hy

        for ex in range(num_elements_x):
            xa = x_left + ex * hx

            xb = xa + hx

            (
                coordinates_local,
                mass_local,
                stiffness_local,
            ) = variable_coefficient_element_2d(
                order_x=order_x,
                order_y=order_y,
                x_left=xa,
                x_right=xb,
                y_bottom=ya,
                y_top=yb,
                coefficient_function=coefficient_function,
            )

            nx_local = order_x + 1

            ny_local = order_y + 1

            global_indices = np.empty(
                nx_local * ny_local,
                dtype=int,
            )

            for j in range(ny_local):
                J = ey * order_y + j

                for i in range(nx_local):
                    I = ex * order_x + i

                    local_index = j * nx_local + i

                    global_index = J * nx_global + I

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
            nx_global,
            ny_global,
        ),
    )


def solve_variable_coefficient_dirichlet_2d(
    num_elements_x: int,
    num_elements_y: int,
    order_x: int,
    order_y: int,
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    coefficient_function,
    source_function,
    boundary_function,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Solve

        -div(kappa grad(phi)) = f

    with Dirichlet boundary conditions.
    """

    (
        coordinates,
        stiffness,
        rhs,
        shape,
    ) = assemble_variable_coefficient_2d(
        num_elements_x=num_elements_x,
        num_elements_y=num_elements_y,
        order_x=order_x,
        order_y=order_y,
        x_left=x_left,
        x_right=x_right,
        y_bottom=y_bottom,
        y_top=y_top,
        coefficient_function=coefficient_function,
        source_function=source_function,
    )

    nx_global, ny_global = shape

    ndof = coordinates.shape[0]

    boundary_mask = np.zeros(
        ndof,
        dtype=bool,
    )

    for j in range(ny_global):
        for i in range(nx_global):
            index = j * nx_global + i

            if i == 0 or i == nx_global - 1 or j == 0 or j == ny_global - 1:
                boundary_mask[index] = True

    boundary = np.where(boundary_mask)[0]

    interior = np.where(~boundary_mask)[0]

    x = coordinates[:, 0]
    y = coordinates[:, 1]

    boundary_values = np.asarray(
        boundary_function(
            x[boundary],
            y[boundary],
        )
    )

    dtype = np.result_type(
        stiffness,
        rhs,
        boundary_values,
    )

    phi = np.zeros(
        ndof,
        dtype=dtype,
    )

    phi[boundary] = boundary_values

    rhs_interior = (
        rhs[interior]
        - stiffness[
            np.ix_(
                interior,
                boundary,
            )
        ]
        @ phi[boundary]
    )

    phi[interior] = np.linalg.solve(
        stiffness[
            np.ix_(
                interior,
                interior,
            )
        ],
        rhs_interior,
    )

    return (
        coordinates,
        phi,
        shape,
    )
