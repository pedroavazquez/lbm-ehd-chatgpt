import numpy as np

from lbm.electrostatics.sem.mixed_boundary import (
    solve_mixed_boundary_2d,
)


def test_complex_electrode_insulating_walls():
    """
    Complex AC potential with:

        phi = phi0 at x = 0
        phi = 0    at x = 1

    and insulating top/bottom:

        n . kappa grad(phi) = 0.

    For uniform kappa,

        phi(x) = phi0 (1 - x).
    """

    sigma = 2.0
    epsilon = 3.0
    omega = 5.0

    kappa0 = sigma + 1.0j * omega * epsilon

    phi0 = 1.0 + 0.5j

    def coefficient(
        x,
        y,
    ):
        return kappa0 * np.ones_like(
            x,
            dtype=complex,
        )

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(
            x,
            dtype=complex,
        )

    def left_potential(
        x,
        y,
    ):
        return phi0 * np.ones_like(
            x,
            dtype=complex,
        )

    def right_potential(
        x,
        y,
    ):
        return np.zeros_like(
            x,
            dtype=complex,
        )

    def zero_flux(
        x,
        y,
    ):
        return np.zeros_like(
            x,
            dtype=complex,
        )

    bc = {
        "left": (
            "dirichlet",
            left_potential,
        ),
        "right": (
            "dirichlet",
            right_potential,
        ),
        "bottom": (
            "neumann",
            zero_flux,
        ),
        "top": (
            "neumann",
            zero_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_mixed_boundary_2d(
        num_elements_x=3,
        num_elements_y=2,
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
        source_function=zero_source,
        boundary_conditions=bc,
    )

    exact = phi0 * (1.0 - coordinates[:, 0])

    assert np.iscomplexobj(phi)

    assert np.allclose(
        phi,
        exact,
        atol=1.0e-11,
    )


def test_nonzero_neumann_flux():
    """
    Exact solution:

        phi(x,y) = x + y

    with

        -laplacian(phi) = 0.

    Use Dirichlet on left/right and Neumann on bottom/top.

    Since grad(phi) = (1,1), with kappa = 1:

        bottom normal = (0,-1) -> flux = -1
        top normal    = (0,+1) -> flux = +1
    """

    def coefficient(
        x,
        y,
    ):
        return np.ones_like(x)

    def source(
        x,
        y,
    ):
        return np.zeros_like(x)

    def exact(
        x,
        y,
    ):
        return x + y

    def bottom_flux(
        x,
        y,
    ):
        return -np.ones_like(x)

    def top_flux(
        x,
        y,
    ):
        return np.ones_like(x)

    bc = {
        "left": (
            "dirichlet",
            exact,
        ),
        "right": (
            "dirichlet",
            exact,
        ),
        "bottom": (
            "neumann",
            bottom_flux,
        ),
        "top": (
            "neumann",
            top_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_mixed_boundary_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
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


def test_robin_boundary_condition():
    """
    Exact solution:

        phi(x,y) = 1 + x

    with

        -laplacian(phi) = 0.

    Boundary conditions:

        left:
            phi = 1

        top/bottom:
            zero normal flux

        right:
            n.grad(phi) + beta phi = g

    At x = 1:

        n.grad(phi) = 1
        phi = 2

    Choose beta = 3, therefore

        g = 1 + 3*2 = 7.
    """

    def coefficient(
        x,
        y,
    ):
        return np.ones_like(x)

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(x)

    def exact(
        x,
        y,
    ):
        return 1.0 + x

    def left_value(
        x,
        y,
    ):
        return np.ones_like(x)

    def zero_flux(
        x,
        y,
    ):
        return np.zeros_like(x)

    def beta(
        x,
        y,
    ):
        return 3.0 * np.ones_like(x)

    def robin_rhs(
        x,
        y,
    ):
        return 7.0 * np.ones_like(x)

    bc = {
        "left": (
            "dirichlet",
            left_value,
        ),
        "right": (
            "robin",
            beta,
            robin_rhs,
        ),
        "bottom": (
            "neumann",
            zero_flux,
        ),
        "top": (
            "neumann",
            zero_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_mixed_boundary_2d(
        num_elements_x=3,
        num_elements_y=2,
        order_x=5,
        order_y=5,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
        source_function=zero_source,
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


def test_complex_robin_boundary():
    """
    Complex-valued Robin condition.

    Exact solution:

        phi(x,y) = phi0 (1 + x)

    for uniform complex kappa.

    At x = 1:

        n . kappa grad(phi)
            = kappa phi0

        phi = 2 phi0.

    Thus for complex beta:

        g_R =
            kappa phi0
            + beta (2 phi0).
    """

    sigma = 2.0
    epsilon = 1.5
    omega = 4.0

    kappa0 = sigma + 1.0j * omega * epsilon

    beta0 = 3.0 + 0.7j

    phi0 = 1.0 - 0.2j

    def coefficient(
        x,
        y,
    ):
        return kappa0 * np.ones_like(
            x,
            dtype=complex,
        )

    def zero_source(
        x,
        y,
    ):
        return np.zeros_like(
            x,
            dtype=complex,
        )

    def exact(
        x,
        y,
    ):
        return phi0 * (1.0 + x)

    def left_value(
        x,
        y,
    ):
        return phi0 * np.ones_like(
            x,
            dtype=complex,
        )

    def zero_flux(
        x,
        y,
    ):
        return np.zeros_like(
            x,
            dtype=complex,
        )

    def beta(
        x,
        y,
    ):
        return beta0 * np.ones_like(
            x,
            dtype=complex,
        )

    def robin_rhs(
        x,
        y,
    ):
        value = kappa0 * phi0 + beta0 * (2.0 * phi0)

        return value * np.ones_like(
            x,
            dtype=complex,
        )

    bc = {
        "left": (
            "dirichlet",
            left_value,
        ),
        "right": (
            "robin",
            beta,
            robin_rhs,
        ),
        "bottom": (
            "neumann",
            zero_flux,
        ),
        "top": (
            "neumann",
            zero_flux,
        ),
    }

    (
        coordinates,
        phi,
        _,
    ) = solve_mixed_boundary_2d(
        num_elements_x=2,
        num_elements_y=2,
        order_x=6,
        order_y=6,
        x_left=0.0,
        x_right=1.0,
        y_bottom=0.0,
        y_top=1.0,
        coefficient_function=coefficient,
        source_function=zero_source,
        boundary_conditions=bc,
    )

    phi_exact = exact(
        coordinates[:, 0],
        coordinates[:, 1],
    )

    assert np.iscomplexobj(phi)

    assert np.allclose(
        phi,
        phi_exact,
        atol=1.0e-11,
    )
