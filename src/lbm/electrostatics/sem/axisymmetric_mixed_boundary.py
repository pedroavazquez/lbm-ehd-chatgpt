import numpy as np

from lbm.electrostatics.sem.axisymmetric import (
    assemble_axisymmetric_2d,
)
from lbm.electrostatics.sem.gll import gll_nodes_weights


def solve_axisymmetric_mixed_boundary_2d(
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
    boundary_conditions: dict,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int],
]:
    """
    Solve

        -div_axi(kappa grad(phi)) = f

    on a structured axisymmetric (z, r) spectral-element mesh.

    Boundary-condition format
    -------------------------

    Dirichlet:
        ("dirichlet", value_function)

    Neumann:
        ("neumann", flux_function)

    where
        n . kappa grad(phi) = flux_function(z, r)

    Robin:
        ("robin", beta_function, rhs_function)

    where
        n . kappa grad(phi)
        + beta phi
        = rhs_function(z, r)

    Boundary names
    --------------
    "left"   : z = z_left
    "right"  : z = z_right
    "axis"   : r = r_bottom
    "outer"  : r = r_top

    For a physical symmetry axis, use r_bottom = 0 and
    ("neumann", zero_flux) on "axis".
    """

    required_sides = {
        "left",
        "right",
        "axis",
        "outer",
    }

    if set(boundary_conditions) != required_sides:
        raise ValueError(
            "boundary_conditions must contain exactly "
            "'left', 'right', 'axis', and 'outer'"
        )

    (
        coordinates,
        stiffness,
        rhs,
        shape,
    ) = assemble_axisymmetric_2d(
        num_elements_z=num_elements_z,
        num_elements_r=num_elements_r,
        order_z=order_z,
        order_r=order_r,
        z_left=z_left,
        z_right=z_right,
        r_bottom=r_bottom,
        r_top=r_top,
        coefficient_function=coefficient_function,
        source_function=source_function,
    )

    nz_global, nr_global = shape
    ndof = coordinates.shape[0]

    def global_index(
        i: int,
        j: int,
    ) -> int:
        return j * nz_global + i

    def evaluate_function(
        function,
        z: float,
        r: float,
    ):
        value = function(
            np.array([z]),
            np.array([r]),
        )

        return np.asarray(value).reshape(-1)[0]

    # ========================================================
    # Edge quadrature
    # ========================================================

    _, weights_z_ref = gll_nodes_weights(order_z)

    _, weights_r_ref = gll_nodes_weights(order_r)

    hz = (z_right - z_left) / num_elements_z

    hr = (r_top - r_bottom) / num_elements_r

    weights_z = 0.5 * hz * weights_z_ref

    weights_r = 0.5 * hr * weights_r_ref

    # ========================================================
    # Neumann and Robin contributions
    # ========================================================

    def add_axial_boundary(
        side: str,
        i_global: int,
    ) -> None:
        """
        Boundary z = constant.

        Edge integration variable is r, so the cylindrical
        boundary weight is

            r dr.
        """

        bc = boundary_conditions[side]

        bc_type = bc[0]

        if bc_type == "dirichlet":
            return

        if bc_type not in (
            "neumann",
            "robin",
        ):
            raise ValueError(f"Unknown boundary type '{bc_type}' on side '{side}'")

        for er in range(num_elements_r):
            for j_local in range(order_r + 1):
                j_global = er * order_r + j_local

                index = global_index(
                    i_global,
                    j_global,
                )

                z = coordinates[
                    index,
                    0,
                ]

                r = coordinates[
                    index,
                    1,
                ]

                weight = weights_r[j_local] * r

                if bc_type == "neumann":
                    flux = evaluate_function(
                        bc[1],
                        z,
                        r,
                    )

                    rhs[index] += weight * flux

                else:
                    beta = evaluate_function(
                        bc[1],
                        z,
                        r,
                    )

                    robin_rhs = evaluate_function(
                        bc[2],
                        z,
                        r,
                    )

                    stiffness[
                        index,
                        index,
                    ] += weight * beta

                    rhs[index] += weight * robin_rhs

    def add_radial_boundary(
        side: str,
        j_global: int,
    ) -> None:
        """
        Boundary r = constant.

        Edge integration variable is z, and cylindrical
        weighting contributes the constant boundary radius r.
        """

        bc = boundary_conditions[side]

        bc_type = bc[0]

        if bc_type == "dirichlet":
            return

        if bc_type not in (
            "neumann",
            "robin",
        ):
            raise ValueError(f"Unknown boundary type '{bc_type}' on side '{side}'")

        for ez in range(num_elements_z):
            for i_local in range(order_z + 1):
                i_global = ez * order_z + i_local

                index = global_index(
                    i_global,
                    j_global,
                )

                z = coordinates[
                    index,
                    0,
                ]

                r = coordinates[
                    index,
                    1,
                ]

                weight = weights_z[i_local] * r

                if bc_type == "neumann":
                    flux = evaluate_function(
                        bc[1],
                        z,
                        r,
                    )

                    rhs[index] += weight * flux

                else:
                    beta = evaluate_function(
                        bc[1],
                        z,
                        r,
                    )

                    robin_rhs = evaluate_function(
                        bc[2],
                        z,
                        r,
                    )

                    stiffness[
                        index,
                        index,
                    ] += weight * beta

                    rhs[index] += weight * robin_rhs

    add_axial_boundary(
        "left",
        0,
    )

    add_axial_boundary(
        "right",
        nz_global - 1,
    )

    add_radial_boundary(
        "axis",
        0,
    )

    add_radial_boundary(
        "outer",
        nr_global - 1,
    )

    # ========================================================
    # Collect Dirichlet conditions
    # ========================================================

    dirichlet_values = {}

    def add_dirichlet_node(
        index: int,
        function,
    ) -> None:

        z = coordinates[
            index,
            0,
        ]

        r = coordinates[
            index,
            1,
        ]

        value = evaluate_function(
            function,
            z,
            r,
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
        ("right", nz_global - 1),
    ):
        bc = boundary_conditions[side]

        if bc[0] == "dirichlet":
            for j in range(nr_global):
                add_dirichlet_node(
                    global_index(
                        i_global,
                        j,
                    ),
                    bc[1],
                )

    for side, j_global in (
        ("axis", 0),
        ("outer", nr_global - 1),
    ):
        bc = boundary_conditions[side]

        if bc[0] == "dirichlet":
            for i in range(nz_global):
                add_dirichlet_node(
                    global_index(
                        i,
                        j_global,
                    ),
                    bc[1],
                )

    has_robin_anchor = any(bc[0] == "robin" for bc in boundary_conditions.values())

    if len(dirichlet_values) == 0 and not has_robin_anchor:
        raise ValueError(
            "The problem has no potential reference: "
            "use at least one Dirichlet or Robin boundary"
        )

    # ========================================================
    # Solve reduced system
    # ========================================================

    if len(dirichlet_values) > 0:
        boundary = np.array(
            sorted(dirichlet_values.keys()),
            dtype=int,
        )

        boundary_values = np.array([dirichlet_values[i] for i in boundary])

        free_mask = np.ones(
            ndof,
            dtype=bool,
        )

        free_mask[boundary] = False

        free = np.where(free_mask)[0]

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

        rhs_free = (
            rhs[free]
            - stiffness[
                np.ix_(
                    free,
                    boundary,
                )
            ]
            @ phi[boundary]
        )

        phi[free] = np.linalg.solve(
            stiffness[
                np.ix_(
                    free,
                    free,
                )
            ],
            rhs_free,
        )

    else:
        phi = np.linalg.solve(
            stiffness,
            rhs,
        )

    return (
        coordinates,
        phi,
        shape,
    )
