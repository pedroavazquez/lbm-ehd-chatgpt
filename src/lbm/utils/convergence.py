import numpy as np


def relative_change(
    new: np.ndarray,
    old: np.ndarray,
    atol: float = 1.0e-30,
) -> float:
    """
    Relative L2 change between two fields.

    Parameters
    ----------
    new, old : ndarray
        New and previous field values.

    atol : float
        Small denominator protection.

    Returns
    -------
    float
        ||new - old||_2 / max(||new||_2, atol)
    """

    numerator = np.linalg.norm(new - old)
    denominator = max(np.linalg.norm(new), atol)

    return numerator / denominator
