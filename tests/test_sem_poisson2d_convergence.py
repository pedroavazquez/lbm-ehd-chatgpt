import numpy as np
import pytest

from lbm.electrostatics.sem.poisson2d_multielement import (
    solve_poisson_dirichlet_multielement_2d,
)


def exact_solution(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def source_function(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def zero_boundary(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    return np.zeros_like(x)


def relative_l2_error(
    numerical: np.ndarray,
    exact: np.ndarray,
) -> float:
    return np.sqrt(np.sum((numerical - exact) ** 2) / np.sum(exact**2))


@pytest.mark.slow
def test_sem_poisson2d_h_convergence():
    """
    h-refinement study for the 2-D Poisson problem
    at fixed polynomial order p = 4.
    """

    order = 4

    num_elements_values = np.array([1, 2, 4, 8])

    errors = []

    for num_elements in num_elements_values:
        (
            coordinates,
            phi,
            _,
        ) = solve_poisson_dirichlet_multielement_2d(
            num_elements_x=int(num_elements),
            num_elements_y=int(num_elements),
            order_x=order,
            order_y=order,
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            source_function=source_function,
            boundary_function=zero_boundary,
        )

        phi_exact = exact_solution(
            coordinates[:, 0],
            coordinates[:, 1],
        )

        error = relative_l2_error(
            phi,
            phi_exact,
        )

        errors.append(error)

    errors = np.asarray(errors)

    # Since h ~ 1 / Ne,
    #
    # doubling Ne halves h:
    #
    # p_obs = log(e_h / e_h/2) / log(2)
    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)

    print()
    print("2-D SEM h-convergence, p = 4")
    print("--------------------------------")
    print(" Ne       error          order")
    print("--------------------------------")

    for j, ne in enumerate(num_elements_values):
        if j == 0:
            print(f"{ne:3d}   {errors[j]:.6e}      ---")

        else:
            print(f"{ne:3d}   {errors[j]:.6e}   {orders[j - 1]:7.4f}")

    # Error should decrease monotonically.
    assert np.all(errors[1:] < errors[:-1])

    # With fourth-order elements we expect high-order
    # algebraic convergence. Keep the threshold conservative.
    assert orders[-1] > 3.5


@pytest.mark.slow
def test_sem_poisson2d_p_convergence():
    """
    p-refinement study on a fixed 2 x 2 element mesh.

    For the analytic exact solution, the error should
    decrease spectrally with polynomial order.
    """

    num_elements_x = 2
    num_elements_y = 2

    polynomial_orders = np.array(
        [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
        ]
    )

    errors = []

    for order in polynomial_orders:
        (
            coordinates,
            phi,
            _,
        ) = solve_poisson_dirichlet_multielement_2d(
            num_elements_x=num_elements_x,
            num_elements_y=num_elements_y,
            order_x=int(order),
            order_y=int(order),
            x_left=0.0,
            x_right=1.0,
            y_bottom=0.0,
            y_top=1.0,
            source_function=source_function,
            boundary_function=zero_boundary,
        )

        phi_exact = exact_solution(
            coordinates[:, 0],
            coordinates[:, 1],
        )

        error = relative_l2_error(
            phi,
            phi_exact,
        )

        errors.append(error)

    errors = np.asarray(errors)

    print()
    print("2-D SEM p-convergence, 2 x 2 elements")
    print("--------------------------------------")
    print(" p        error")
    print("--------------------------------------")

    for order, error in zip(
        polynomial_orders,
        errors,
    ):
        print(f"{order:2d}   {error:.6e}")

    # Spectral convergence should reduce the error
    # by many orders of magnitude.
    assert errors[-1] < errors[0]

    assert errors[-1] < 1.0e-5 * errors[0]

    # Enforce monotonic decrease before machine precision
    # starts to dominate.
    assert np.all(errors[1:7] < errors[:6])
