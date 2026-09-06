import numpy as np


def manufactured_velocity(
    z: np.ndarray,
    r: np.ndarray,
    amplitude: float,
    radius: float,
    wavenumber: float,
) -> np.ndarray:
    """
    Exact divergence-free, regular axisymmetric manufactured flow.

    The generating Stokes streamfunction is

        Psi = A r^2 (1 - r^2/R^2)^2 sin(k z)

    giving

        u_z = (1/r) dPsi/dr
        u_r = -(1/r) dPsi/dz

    The field satisfies

        u_r = 0                    at r = 0
        du_z/dr = 0                at r = 0

        u_z = 0
        u_r = 0                    at r = R

    and is periodic in z.

    Returns
    -------
    u : ndarray, shape (2, nz, nr)

        u[0] = u_z
        u[1] = u_r
    """

    Z, Rcoord = np.meshgrid(
        z,
        r,
        indexing="ij",
    )

    A = amplitude
    R = radius
    k = wavenumber

    # --------------------------------------------------------
    # Radial functions
    #
    # u_z = A p(r) sin(kz)
    #
    # u_r = -A k q(r) cos(kz)
    # --------------------------------------------------------

    p = 2.0 - 8.0 * Rcoord**2 / R**2 + 6.0 * Rcoord**4 / R**4

    q = Rcoord - 2.0 * Rcoord**3 / R**2 + Rcoord**5 / R**4

    uz = A * p * np.sin(k * Z)

    ur = -A * k * q * np.cos(k * Z)

    return np.stack(
        (
            uz,
            ur,
        ),
        axis=0,
    )


def manufactured_acceleration(
    z: np.ndarray,
    r: np.ndarray,
    amplitude: float,
    radius: float,
    wavenumber: float,
    nu: float,
) -> np.ndarray:
    """
    Exact body acceleration required to sustain the
    manufactured steady axisymmetric flow.

    We choose constant pressure, so

        a = (u . grad)u - nu Laplacian_axi(u)

    where the vector Laplacian in cylindrical coordinates is

        (Lap u)_z
            = d2uz/dz2
            + d2uz/dr2
            + (1/r) duz/dr

        (Lap u)_r
            = d2ur/dz2
            + d2ur/dr2
            + (1/r) dur/dr
            - ur/r^2.
    """

    Z, Rcoord = np.meshgrid(
        z,
        r,
        indexing="ij",
    )

    A = amplitude
    R = radius
    k = wavenumber

    sin_kz = np.sin(k * Z)

    cos_kz = np.cos(k * Z)

    # ========================================================
    # Radial functions
    # ========================================================

    p = 2.0 - 8.0 * Rcoord**2 / R**2 + 6.0 * Rcoord**4 / R**4

    dp = -16.0 * Rcoord / R**2 + 24.0 * Rcoord**3 / R**4

    ddp = -16.0 / R**2 + 72.0 * Rcoord**2 / R**4

    q = Rcoord - 2.0 * Rcoord**3 / R**2 + Rcoord**5 / R**4

    dq = 1.0 - 6.0 * Rcoord**2 / R**2 + 5.0 * Rcoord**4 / R**4

    ddq = -12.0 * Rcoord / R**2 + 20.0 * Rcoord**3 / R**4

    # ========================================================
    # Velocity
    # ========================================================

    uz = A * p * sin_kz

    ur = -A * k * q * cos_kz

    # ========================================================
    # First derivatives
    # ========================================================

    duz_dz = A * k * p * cos_kz

    duz_dr = A * dp * sin_kz

    dur_dz = A * k**2 * q * sin_kz

    dur_dr = -A * k * dq * cos_kz

    # ========================================================
    # Second derivatives
    # ========================================================

    d2uz_dz2 = -A * k**2 * p * sin_kz

    d2uz_dr2 = A * ddp * sin_kz

    d2ur_dz2 = A * k**3 * q * cos_kz

    d2ur_dr2 = -A * k * ddq * cos_kz

    # ========================================================
    # Convective acceleration
    # ========================================================

    conv_z = uz * duz_dz + ur * duz_dr

    conv_r = uz * dur_dz + ur * dur_dr

    # ========================================================
    # Cylindrical vector Laplacian
    # ========================================================

    lap_z = d2uz_dz2 + d2uz_dr2 + duz_dr / Rcoord

    lap_r = d2ur_dz2 + d2ur_dr2 + dur_dr / Rcoord - ur / Rcoord**2

    # ========================================================
    # Required acceleration
    # ========================================================

    acceleration_z = conv_z - nu * lap_z

    acceleration_r = conv_r - nu * lap_r

    return np.stack(
        (
            acceleration_z,
            acceleration_r,
        ),
        axis=0,
    )


def manufactured_divergence(
    z: np.ndarray,
    r: np.ndarray,
    amplitude: float,
    radius: float,
    wavenumber: float,
) -> np.ndarray:
    """
    Cylindrical divergence

        div u =
            du_z/dz
            + du_r/dr
            + u_r/r.

    It should be zero to roundoff.
    """

    Z, Rcoord = np.meshgrid(
        z,
        r,
        indexing="ij",
    )

    A = amplitude
    R = radius
    k = wavenumber

    p = 2.0 - 8.0 * Rcoord**2 / R**2 + 6.0 * Rcoord**4 / R**4

    q = Rcoord - 2.0 * Rcoord**3 / R**2 + Rcoord**5 / R**4

    dq = 1.0 - 6.0 * Rcoord**2 / R**2 + 5.0 * Rcoord**4 / R**4

    duz_dz = A * k * p * np.cos(k * Z)

    ur = -A * k * q * np.cos(k * Z)

    dur_dr = -A * k * dq * np.cos(k * Z)

    return duz_dz + dur_dr + ur / Rcoord
