import numpy as np

from lbm.electrostatics.sem.axisymmetric_mixed_boundary import (
    solve_axisymmetric_mixed_boundary_2d,
)


def constant_coefficient(
    z,
    r,
):
    return np.ones_like(z)


def zero_source(
    z,
    r,
):
    return np.zeros_like(z)


def zero_flux(
    z,
    r,
):
    return np.zeros_like(z)


def test_axisymmetric_symmetry_axis():
    """
    Exact solution:

        phi(z,r) = 1 - z

    It is independent of r, so

        dphi/dr = 0

    at the axis and outer radial boundary.

    Boundary conditions:

        phi = 1 at z = 0
        phi = 0 at z = 1

        zero flux at r = 0
        zero flux at r = 1.
    """

    def left_value(
        z,
        r,
    ):
        return np.ones_like(z)

    def right_value(
        z,
        r,
    ):
        return np.zeros_like(z)

    bc = {
        "left": (
            "dirichlet",
            left_value,
        ),
        "right": (
            "dirichlet",
            right_value,
        ),
        "axis": (
            "neumann",
            zero_flux,
        ),
        "outer": (
            "neumann",
            zero_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_axisymmetric_mixed_boundary_2d(
        num_elements_z=3,
        num_elements_r=2,
        order_z=5,
        order_r=5,
        z_left=0.0,
        z_right=1.0,
        r_bottom=0.0,
        r_top=1.0,
        coefficient_function=constant_coefficient,
        source_function=zero_source,
        boundary_conditions=bc,
    )

    exact = 1.0 - coordinates[:, 0]

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-11,
    )


def test_axisymmetric_nonzero_outer_flux():
    """
    Exact solution:

        phi(z,r) = r^2

    For kappa = 1,

        div_axi(grad phi) = 4,

    therefore

        -div_axi(grad phi) = -4.

    At r = 1:

        n.grad(phi) = dphi/dr = 2.

    At r = 0:

        dphi/dr = 0.

    Use Dirichlet in z only to fix the reference.
    """

    def source(
        z,
        r,
    ):
        return -4.0 * np.ones_like(z)

    def exact(
        z,
        r,
    ):
        return r**2

    def outer_flux(
        z,
        r,
    ):
        return 2.0 * np.ones_like(z)

    bc = {
        "left": (
            "dirichlet",
            exact,
        ),
        "right": (
            "dirichlet",
            exact,
        ),
        "axis": (
            "neumann",
            zero_flux,
        ),
        "outer": (
            "neumann",
            outer_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_axisymmetric_mixed_boundary_2d(
        num_elements_z=2,
        num_elements_r=3,
        order_z=5,
        order_r=5,
        z_left=0.0,
        z_right=1.0,
        r_bottom=0.0,
        r_top=1.0,
        coefficient_function=constant_coefficient,
        source_function=source,
        boundary_conditions=bc,
    )

    phi_exact = exact(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    assert np.allclose(
        phi,
        phi_exact,
        atol=1.0e-11,
    )


def test_axisymmetric_complex_ac():
    """
    Uniform complex coefficient with electrode-like axial
    Dirichlet conditions and insulating radial boundaries.
    """

    sigma = 2.0
    epsilon = 3.0
    omega = 4.0

    kappa0 = sigma + 1.0j * omega * epsilon

    phi0 = 1.0 - 0.3j

    def coefficient(
        z,
        r,
    ):
        return kappa0 * np.ones_like(
            z,
            dtype=complex,
        )

    def source(
        z,
        r,
    ):
        return np.zeros_like(
            z,
            dtype=complex,
        )

    def left_value(
        z,
        r,
    ):
        return phi0 * np.ones_like(
            z,
            dtype=complex,
        )

    def right_value(
        z,
        r,
    ):
        return np.zeros_like(
            z,
            dtype=complex,
        )

    def zero_complex_flux(
        z,
        r,
    ):
        return np.zeros_like(
            z,
            dtype=complex,
        )

    bc = {
        "left": (
            "dirichlet",
            left_value,
        ),
        "right": (
            "dirichlet",
            right_value,
        ),
        "axis": (
            "neumann",
            zero_complex_flux,
        ),
        "outer": (
            "neumann",
            zero_complex_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_axisymmetric_mixed_boundary_2d(
        num_elements_z=3,
        num_elements_r=2,
        order_z=6,
        order_r=6,
        z_left=0.0,
        z_right=1.0,
        r_bottom=0.0,
        r_top=1.0,
        coefficient_function=coefficient,
        source_function=source,
        boundary_conditions=bc,
    )

    exact = phi0 * (1.0 - coordinates[:, 0])

    assert np.iscomplexobj(phi)

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-11,
    )
