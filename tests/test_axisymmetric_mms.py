import numpy as np

from lbm.verification.axisymmetric_mms import (
    manufactured_divergence,
    manufactured_velocity,
)


def test_manufactured_flow_is_divergence_free():

    nz = 32
    nr = 16

    radius = float(nr)

    length = 2.0 * radius

    dz = length / nz

    z = (np.arange(nz) + 0.5) * dz

    r = np.arange(nr) + 0.5

    k = 2.0 * np.pi / length

    divergence = manufactured_divergence(
        z=z,
        r=r,
        amplitude=0.01,
        radius=radius,
        wavenumber=k,
    )

    assert np.max(np.abs(divergence)) < 1.0e-14


def test_manufactured_velocity_zero_at_outer_wall():

    nz = 16

    radius = 10.0

    length = 20.0

    z = (np.arange(nz) + 0.5) * (length / nz)

    # Evaluate exactly at the wall.
    r = np.array([radius])

    k = 2.0 * np.pi / length

    u = manufactured_velocity(
        z=z,
        r=r,
        amplitude=0.01,
        radius=radius,
        wavenumber=k,
    )

    assert np.allclose(
        u,
        0.0,
        atol=1.0e-14,
    )


def test_manufactured_velocity_regular_at_axis():

    nz = 16

    radius = 1.0
    length = 2.0

    amplitude = 0.01

    k = 2.0 * np.pi / length

    z = (np.arange(nz) + 0.5) * (length / nz)

    # Evaluate very close to the axis on both sides.
    eps = 1.0e-7

    r = np.array(
        [
            -eps,
            eps,
        ]
    )

    u = manufactured_velocity(
        z=z,
        r=r,
        amplitude=amplitude,
        radius=radius,
        wavenumber=k,
    )

    # u_z must be even in r.
    assert np.allclose(
        u[0, :, 0],
        u[0, :, 1],
        atol=1.0e-12,
    )

    # u_r must be odd in r.
    assert np.allclose(
        u[1, :, 0],
        -u[1, :, 1],
        atol=1.0e-12,
    )
