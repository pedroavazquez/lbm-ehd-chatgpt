import numpy as np

from lbm.hydro.axisymmetric import (
    axisymmetric_macroscopic,
    axisymmetric_source,
    initialize_axisymmetric,
)


def test_axisymmetric_source_zero_at_rest():

    nz, nr = 6, 8

    rho = np.ones((nz, nr))

    u = np.zeros((2, nz, nr))

    source = axisymmetric_source(
        rho,
        u,
        tau=0.8,
    )

    assert np.allclose(
        source,
        0.0,
    )


def test_source_zeroth_moment():

    nz, nr = 6, 8

    rho = np.ones((nz, nr))

    u = np.zeros((2, nz, nr))

    u[1] = 1.0e-3

    source = axisymmetric_source(
        rho,
        u,
        tau=0.8,
    )

    r = np.arange(nr) + 0.5

    zeroth_moment = np.sum(
        source,
        axis=0,
    )

    expected = -rho * u[1] / r[None, :]

    assert np.allclose(
        zeroth_moment,
        expected,
    )


def test_axisymmetric_initialization_roundtrip():

    nz, nr = 7, 9

    tau = 0.8

    rho_expected = 1.0 * np.ones((nz, nr))

    u_expected = np.zeros((2, nz, nr))

    u_expected[0] = 0.02
    u_expected[1] = 1.0e-3

    f = initialize_axisymmetric(
        rho_expected,
        u_expected,
        tau=tau,
    )

    rho, u = axisymmetric_macroscopic(
        f,
        tau=tau,
    )

    assert np.allclose(
        rho,
        rho_expected,
    )

    assert np.allclose(
        u,
        u_expected,
    )
