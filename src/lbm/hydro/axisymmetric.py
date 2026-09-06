import numpy as np

from lbm.hydro.equilibrium import equilibrium
from lbm.lattices.d2q9 import CS2, C, Q


def radial_coordinates(nr: int) -> np.ndarray:
    """
    Radial coordinates of fluid nodes.

    We use a cell-centred radial grid:

        r_j = j + 1/2

    so that the symmetry axis lies halfway below the
    first fluid node.
    """

    return np.arange(nr, dtype=np.float64) + 0.5


def axisymmetric_source(
    rho: np.ndarray,
    u: np.ndarray,
    tau: float,
    external_force: np.ndarray | None = None,
    rho0: float = 1.0,
) -> np.ndarray:
    """
    Axisymmetric source term of the improved Li et al.
    BGK formulation for non-swirling axisymmetric flow.

    Coordinates
    -----------
    axis 0 : z
    axis 1 : r

    Parameters
    ----------
    rho : ndarray, shape (nz, nr)
        Physical density.

    u : ndarray, shape (2, nz, nr)
        Physical velocity:
            u[0] = u_z
            u[1] = u_r

    tau : float
        Standard lattice BGK relaxation time.
        Must satisfy tau > 0.5.

    external_force : ndarray, shape (2, nz, nr), optional
        External force density.

    rho0 : float
        Reference density used for the constant dynamic
        viscosity in the incompressible formulation.

    Returns
    -------
    source : ndarray, shape (9, nz, nr)
        Axisymmetric lattice source S_i.
    """

    if tau <= 0.5:
        raise ValueError("tau must be greater than 0.5")

    nz, nr = rho.shape

    r = radial_coordinates(nr)

    # --------------------------------------------------------
    # Kinematic and dynamic viscosity
    #
    # Li's tau_a = tau_LBM - 1/2
    #
    # nu = cs^2 tau_a
    # --------------------------------------------------------

    tau_a = tau - 0.5

    nu = CS2 * tau_a

    mu0 = rho0 * nu

    # --------------------------------------------------------
    # External force density
    # --------------------------------------------------------

    if external_force is None:
        force_ext = np.zeros(
            (2, nz, nr),
            dtype=np.float64,
        )

    else:
        force_ext = external_force

    # --------------------------------------------------------
    # Axisymmetric force F_i
    #
    # F_z = F_ext,z
    #
    # F_r = F_ext,r - 2 mu u_r / r^2
    #
    # The second term is the cylindrical viscous contribution.
    # --------------------------------------------------------

    force_total = force_ext.copy()

    force_total[1] -= 2.0 * mu0 * u[1] / r[None, :] ** 2

    # --------------------------------------------------------
    # Standard equilibrium distribution
    # --------------------------------------------------------

    feq = equilibrium(
        rho,
        u,
    )

    # --------------------------------------------------------
    # Source:
    #
    # S_i =
    #
    # [
    #   (c_i-u).F / (rho cs^2)
    #   - u_r/r
    # ] f_i^eq
    # --------------------------------------------------------

    source = np.empty_like(feq)

    for i in range(Q):
        ci_minus_u = C[i, :, None, None] - u

        force_projection = np.sum(
            ci_minus_u * force_total,
            axis=0,
        )

        source[i] = feq[i] * (force_projection / (rho * CS2) - u[1] / r[None, :])

    return source


def axisymmetric_macroscopic(
    f: np.ndarray,
    tau: float,
    external_force: np.ndarray | None = None,
    rho0: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover physical density and velocity from the transformed
    distribution functions of the Li axisymmetric scheme.

    Coordinates
    -----------
    u[0] = u_z
    u[1] = u_r
    """

    if tau <= 0.5:
        raise ValueError("tau must be greater than 0.5")

    nz, nr = f.shape[1:]

    r = radial_coordinates(nr)

    tau_a = tau - 0.5

    nu = CS2 * tau_a

    mu0 = rho0 * nu

    # --------------------------------------------------------
    # Zeroth and first moments of transformed populations
    # --------------------------------------------------------

    rho_hat = np.sum(
        f,
        axis=0,
    )

    momentum = np.einsum(
        "izr,ia->azr",
        f,
        C,
    )

    # --------------------------------------------------------
    # External force correction
    # --------------------------------------------------------

    if external_force is not None:
        momentum += 0.5 * external_force

    # --------------------------------------------------------
    # Velocity.
    #
    # Axial:
    #
    #   u_z = m_z / rho_hat
    #
    # Radial:
    #
    #   u_r =
    #       m_r /
    #       (rho_hat + mu0/r^2)
    # --------------------------------------------------------

    u = np.empty(
        (2, nz, nr),
        dtype=np.float64,
    )

    u[0] = momentum[0] / rho_hat

    u[1] = momentum[1] / (rho_hat + mu0 / r[None, :] ** 2)

    # --------------------------------------------------------
    # Density:
    #
    # rho =
    #
    #   sum_i f_i
    #   -----------------
    #   1 + u_r/(2r)
    # --------------------------------------------------------

    rho = rho_hat / (1.0 + 0.5 * u[1] / r[None, :])

    return rho, u


def collide_axisymmetric(
    f: np.ndarray,
    tau: float,
    external_force: np.ndarray | None = None,
    rho0: float = 1.0,
) -> np.ndarray:
    """
    General non-swirling axisymmetric BGK collision step
    following Li et al. (2010).
    """

    rho, u = axisymmetric_macroscopic(
        f,
        tau=tau,
        external_force=external_force,
        rho0=rho0,
    )

    feq = equilibrium(
        rho,
        u,
    )

    source = axisymmetric_source(
        rho,
        u,
        tau=tau,
        external_force=external_force,
        rho0=rho0,
    )

    nr = rho.shape[1]

    r = radial_coordinates(nr)

    tau_a = tau - 0.5

    f_post = np.empty_like(f)

    # --------------------------------------------------------
    # Direction-dependent relaxation:
    #
    # omega_i =
    #
    # (1 + tau_a c_ir/r) / tau
    #
    # where tau is our conventional lattice relaxation time.
    # --------------------------------------------------------

    for i in range(Q):
        c_ir = C[i, 1]

        omega = (1.0 + tau_a * c_ir / r[None, :]) / tau

        f_post[i] = f[i] - omega * (f[i] - feq[i]) + (1.0 - 0.5 * omega) * source[i]

    return f_post


def initialize_axisymmetric(
    rho: np.ndarray,
    u: np.ndarray,
    tau: float,
    external_force: np.ndarray | None = None,
    rho0: float = 1.0,
) -> np.ndarray:
    """
    Initialize the transformed axisymmetric populations
    consistently with prescribed physical rho and u.
    """

    feq = equilibrium(
        rho,
        u,
    )

    source = axisymmetric_source(
        rho,
        u,
        tau=tau,
        external_force=external_force,
        rho0=rho0,
    )

    # At initial local equilibrium Omega_i = 0, therefore
    #
    #   f_hat = f_eq - 1/2 S
    #
    # in lattice units.
    return feq - 0.5 * source
