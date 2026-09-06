import numpy as np


def differentiation_matrix(nodes: np.ndarray) -> np.ndarray:
    """
    Construct the nodal Lagrange differentiation matrix.

    Parameters
    ----------
    nodes : ndarray, shape (n,)
        Distinct interpolation nodes.

    Returns
    -------
    D : ndarray, shape (n, n)
        Differentiation matrix satisfying

            f'(x_i) = sum_j D[i, j] f(x_j)

        for polynomials of degree <= n - 1.
    """

    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    n = nodes.size

    if n < 2:
        raise ValueError("at least two nodes are required")

    # Barycentric weights:
    #
    # lambda_j =
    # 1 / product_{k != j} (x_j - x_k)
    barycentric_weights = np.ones(
        n,
        dtype=float,
    )

    for j in range(n):
        differences = nodes[j] - np.delete(
            nodes,
            j,
        )

        barycentric_weights[j] = 1.0 / np.prod(differences)

    D = np.empty(
        (n, n),
        dtype=float,
    )

    # Off-diagonal entries:
    #
    # D_ij =
    #
    # lambda_j /
    # [lambda_i (x_i - x_j)]
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = barycentric_weights[j] / (
                    barycentric_weights[i] * (nodes[i] - nodes[j])
                )

    # Diagonal entries follow from
    #
    # sum_j D_ij = 0
    #
    # because the derivative of a constant is zero.
    for i in range(n):
        D[i, i] = -np.sum(D[i, np.arange(n) != i])

    return D
