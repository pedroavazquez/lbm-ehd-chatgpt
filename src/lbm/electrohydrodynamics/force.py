import numpy as np


def electric_force_density(
    charge_density: np.ndarray,
    electric_field: np.ndarray,
) -> np.ndarray:
    """
    Compute the Coulomb electric force density

        F_e = rho_e E.

    Parameters
    ----------
    charge_density : ndarray, shape (nx, ny)
        Free electric charge density.

    electric_field : ndarray, shape (2, nx, ny)
        Electric field.

    Returns
    -------
    force : ndarray, shape (2, nx, ny)
        Electric body-force density.
    """

    if electric_field.ndim != 3:
        raise ValueError("electric_field must have shape (2, nx, ny)")

    if electric_field.shape[0] != 2:
        raise ValueError("electric_field must have two components")

    if charge_density.shape != electric_field.shape[1:]:
        raise ValueError(
            "charge_density and electric_field must use the same spatial grid"
        )

    return electric_field * charge_density[None, :, :]
