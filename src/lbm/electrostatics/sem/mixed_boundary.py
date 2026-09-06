import numpy as np

from lbm.electrostatics.sem.gll import gll_nodes_weights
from lbm.electrostatics.sem.variable_coefficient import (
    assemble_variable_coefficient_2d,
)


def solve_mixed_boundary_2d(
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
    boundary_conditions: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Solve

        -div(kappa grad(phi)) = f

    on a structured rectangular SEM mesh.

    Boundary-condition format
    -------------------------

    Dirichlet:

        ("dirichlet", value_function)

    with

        phi = value_function(x, y)

    Neumann:

        ("neumann", flux_function)

    with

        n . kappa grad(phi) = flux_function(x, y)

    Robin:

        ("robin", beta_function, rhs_function)

    with

        n . kappa grad(phi)
        + beta phi
        = rhs_function(x, y)

    The coefficient and boundary data may be real or complex.
    """

    required_sides = {
        "left",
        "right",
        "bottom",
        "top",
    }

    if set(boundary_conditions) != required_sides:
        raise ValueError(
            "boundary_conditions must contain exactly "
            "'left', 'right', 'bottom', and 'top'"
        )

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

    def global_index(
        i: int,
        j: int,
    ) -> int:
        return j * nx_global + i

    # ========================================================
    # Edge quadrature data
    # ========================================================

    _, weights_x_ref = gll_nodes_weights(order_x)

    _, weights_y_ref = gll_nodes_weights(order_y)

    hx = (x_right - x_left) / num_elements_x

    hy = (y_top - y_bottom) / num_elements_y

    weights_x = 0.5 * hx * weights_x_ref

    weights_y = 0.5 * hy * weights_y_ref

    # ========================================================
    # Utility for scalar/vectorized BC functions
    # ========================================================

    def evaluate_function(
        function,
        x: float,
        y: float,
    ):
        value = function(
            np.array([x]),
            np.array([y]),
        )

        return np.asarray(value).reshape(-1)[0]

    # ========================================================
    # Neumann and Robin contributions
    # ========================================================

    def add_vertical_boundary(
        side: str,
        i_global: int,
    ) -> None:

        bc = boundary_conditions[side]

        bc_type = bc[0]

        if bc_type == "dirichlet":
            return

        if bc_type not in (
            "neumann",
            "robin",
        ):
            raise ValueError(f"Unknown boundary type '{bc_type}' on side '{side}'")

        for ey in range(num_elements_y):
            for j_local in range(order_y + 1):
                j_global = ey * order_y + j_local

                index = global_index(
                    i_global,
                    j_global,
                )

                x = coordinates[
                    index,
                    0,
                ]

                y = coordinates[
                    index,
                    1,
                ]

                weight = weights_y[j_local]

                if bc_type == "neumann":
                    flux_function = bc[1]

                    flux = evaluate_function(
                        flux_function,
                        x,
                        y,
                    )

                    rhs[index] += weight * flux

                else:
                    beta_function = bc[1]
                    rhs_function = bc[2]

                    beta = evaluate_function(
                        beta_function,
                        x,
                        y,
                    )

                    robin_rhs = evaluate_function(
                        rhs_function,
                        x,
                        y,
                    )

                    stiffness[
                        index,
                        index,
                    ] += weight * beta

                    rhs[index] += weight * robin_rhs

    def add_horizontal_boundary(
        side: str,
        j_global: int,
    ) -> None:

        bc = boundary_conditions[side]

        bc_type = bc[0]

        if bc_type == "dirichlet":
            return

        if bc_type not in (
            "neumann",
            "robin",
        ):
            raise ValueError(f"Unknown boundary type '{bc_type}' on side '{side}'")

        for ex in range(num_elements_x):
            for i_local in range(order_x + 1):
                i_global = ex * order_x + i_local

                index = global_index(
                    i_global,
                    j_global,
                )

                x = coordinates[
                    index,
                    0,
                ]

                y = coordinates[
                    index,
                    1,
                ]

                weight = weights_x[i_local]

                if bc_type == "neumann":
                    flux_function = bc[1]

                    flux = evaluate_function(
                        flux_function,
                        x,
                        y,
                    )

                    rhs[index] += weight * flux

                else:
                    beta_function = bc[1]
                    rhs_function = bc[2]

                    beta = evaluate_function(
                        beta_function,
                        x,
                        y,
                    )

                    robin_rhs = evaluate_function(
                        rhs_function,
                        x,
                        y,
                    )

                    stiffness[
                        index,
                        index,
                    ] += weight * beta

                    rhs[index] += weight * robin_rhs

    add_vertical_boundary(
        "left",
        0,
    )

    add_vertical_boundary(
        "right",
        nx_global - 1,
    )

    add_horizontal_boundary(
        "bottom",
        0,
    )

    add_horizontal_boundary(
        "top",
        ny_global - 1,
    )

    # ========================================================
    # Collect strong Dirichlet conditions
    # ========================================================

    dirichlet_values = {}

    def add_dirichlet_node(
        index: int,
        function,
    ) -> None:

        x = coordinates[
            index,
            0,
        ]

        y = coordinates[
            index,
            1,
        ]

        value = evaluate_function(
            function,
            x,
            y,
        )

        if index in dirichlet_values:
            if not np.allclose(
                dirichlet_values[index],
                value,
                atol=1.0e-12,
                rtol=1.0e-12,
            ):
                raise ValueError("Inconsistent Dirichlet values at a corner")

        else:
            dirichlet_values[index] = value

    for side, i_global in (
        ("left", 0),
        ("right", nx_global - 1),
    ):
        bc = boundary_conditions[side]

        if bc[0] == "dirichlet":
            function = bc[1]

            for j in range(ny_global):
                add_dirichlet_node(
                    global_index(
                        i_global,
                        j,
                    ),
                    function,
                )

    for side, j_global in (
        ("bottom", 0),
        ("top", ny_global - 1),
    ):
        bc = boundary_conditions[side]

        if bc[0] == "dirichlet":
            function = bc[1]

            for i in range(nx_global):
                add_dirichlet_node(
                    global_index(
                        i,
                        j_global,
                    ),
                    function,
                )

    # A pure Neumann problem is singular.
    #
    # A Robin boundary with nonzero beta can also fix the
    # reference, so Dirichlet is not strictly required.
    has_robin_anchor = False

    for bc in boundary_conditions.values():
        if bc[0] == "robin":
            has_robin_anchor = True
            break

    if len(dirichlet_values) == 0 and not has_robin_anchor:
        raise ValueError(
            "The problem has no potential reference: "
            "use at least one Dirichlet or Robin boundary"
        )

    # ========================================================
    # Solve
    # ========================================================

    if len(dirichlet_values) > 0:
        boundary = np.array(
            sorted(dirichlet_values.keys()),
            dtype=int,
        )

        boundary_values = np.array([dirichlet_values[i] for i in boundary])

        interior_mask = np.ones(
            ndof,
            dtype=bool,
        )

        interior_mask[boundary] = False

        interior = np.where(interior_mask)[0]

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

    else:
        # Robin-only anchoring case.
        phi = np.linalg.solve(
            stiffness,
            rhs,
        )

    return (
        coordinates,
        phi,
        shape,
    )
