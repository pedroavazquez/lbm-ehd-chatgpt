import numpy as np
import pytest

from lbm.electrostatics.sem.poisson1d_multielement import (
    solve_poisson_dirichlet_multielement,
)


def exact_solution(x):
    return np.sin(np.pi * x)


def source_function(x):
    return np.pi**2 * np.sin(np.pi * x)


def relative_l2_error(
    numerical,
    exact,
):
    return np.sqrt(np.sum((numerical - exact) ** 2) / np.sum(exact**2))


@pytest.mark.slow
def test_sem_h_convergence():
    """
    h-refinement study for fixed polynomial order p = 4.
    """

    order = 4

    num_elements_values = np.array([1, 2, 4, 8, 16])

    errors = []

    for num_elements in num_elements_values:
        x, phi = solve_poisson_dirichlet_multielement(
            num_elements=int(num_elements),
            order=order,
            x_left=0.0,
            x_right=1.0,
            source_function=source_function,
            phi_left=0.0,
            phi_right=0.0,
        )

        phi_exact = exact_solution(x)

        error = relative_l2_error(
            phi,
            phi_exact,
        )

        errors.append(error)

    errors = np.asarray(errors)

    # Element size:
    #
    # h = 1 / Ne
    #
    # Doubling Ne halves h.
    orders = np.log(errors[:-1] / errors[1:]) / np.log(2.0)

    print()
    print("SEM h-convergence, p = 4")
    print("--------------------------------")
    print(" Ne       error          order")
    print("--------------------------------")

    for j, ne in enumerate(num_elements_values):
        if j == 0:
            print(f"{ne:3d}   {errors[j]:.6e}      ---")

        else:
            print(f"{ne:3d}   {errors[j]:.6e}   {orders[j - 1]:7.4f}")

    assert np.all(errors[1:] < errors[:-1])

    # For p = 4 we expect high-order algebraic convergence.
    # Keep the threshold slightly conservative.
    assert orders[-1] > 3.5


@pytest.mark.slow
def test_sem_p_convergence():
    """
    p-refinement study at fixed number of elements.

    For the analytic solution sin(pi x), the error should
    decrease spectrally/exponentially with polynomial order.
    """

    num_elements = 2

    orders = np.array(
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
            11,
            12,
        ]
    )

    errors = []

    for order in orders:
        x, phi = solve_poisson_dirichlet_multielement(
            num_elements=num_elements,
            order=int(order),
            x_left=0.0,
            x_right=1.0,
            source_function=source_function,
            phi_left=0.0,
            phi_right=0.0,
        )

        phi_exact = exact_solution(x)

        error = relative_l2_error(
            phi,
            phi_exact,
        )

        errors.append(error)

    errors = np.asarray(errors)

    print()
    print("SEM p-convergence, Ne = 2")
    print("-----------------------------")
    print(" p        error")
    print("-----------------------------")

    for order, error in zip(
        orders,
        errors,
    ):
        print(f"{order:2d}   {error:.6e}")

    # The error should decrease substantially as p increases.
    assert errors[-1] < errors[0]

    # Spectral convergence should reduce the error by several
    # orders of magnitude over this range.
    assert errors[-1] < 1.0e-6 * errors[0]

    # Over the practically useful range, enforce mostly
    # monotonic decrease.
    assert np.all(errors[1:8] < errors[:7])
