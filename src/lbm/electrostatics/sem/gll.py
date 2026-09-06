import numpy as np
from numpy.polynomial.legendre import Legendre


def gll_nodes_weights(order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Lobatto-Legendre nodes and quadrature weights.

    Parameters
    ----------
    order : int
        Polynomial order p. The element contains p + 1 nodes.

    Returns
    -------
    nodes : ndarray, shape (p + 1,)
        GLL nodes in [-1, 1].

    weights : ndarray, shape (p + 1,)
        GLL quadrature weights.
    """

    if order < 1:
        raise ValueError("order must be at least 1")

    p = order

    # Legendre polynomial P_p.
    P = Legendre.basis(p)

    # Interior GLL nodes are the roots of P'_p.
    if p == 1:
        interior = np.array(
            [],
            dtype=float,
        )

    else:
        interior = P.deriv().roots()

        # The roots are theoretically real. Depending on the
        # polynomial root algorithm, NumPy may return a complex
        # dtype with zero or roundoff-level imaginary parts.
        interior = np.real_if_close(
            interior,
            tol=1000,
        )

        if np.iscomplexobj(interior):
            raise RuntimeError("Unexpected complex GLL roots")

        interior = np.asarray(
            interior,
            dtype=float,
        )

    nodes = np.concatenate(
        (
            np.array([-1.0]),
            interior,
            np.array([1.0]),
        )
    )

    # Ensure monotonically increasing ordering.
    nodes.sort()

    # GLL weights:
    #
    # w_i = 2 / [p(p+1) P_p(x_i)^2]
    weights = 2.0 / (p * (p + 1) * P(nodes) ** 2)

    return nodes, weights
