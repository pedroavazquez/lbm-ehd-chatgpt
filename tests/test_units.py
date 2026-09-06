import numpy as np

from lbm.lattices.d2q9 import CS2
from lbm.units import LatticeScaling


def test_dx():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=100,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    assert np.isclose(
        scaling.dx,
        0.01,
    )


def test_lattice_viscosity():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=100,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    expected = CS2 * (0.8 - 0.5)

    assert np.isclose(
        scaling.nu_lattice,
        expected,
    )


def test_viscosity_scaling():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=100,
        nu_phys=2.0e-3,
        tau=0.8,
    )

    nu_recovered = scaling.nu_phys * scaling.dt / scaling.dx**2

    assert np.isclose(
        nu_recovered,
        scaling.nu_lattice,
    )


def test_velocity_roundtrip():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=64,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    u_phys = 0.025

    u_lattice = scaling.velocity_to_lattice(u_phys)

    recovered = scaling.velocity_to_physical(u_lattice)

    assert np.isclose(
        recovered,
        u_phys,
    )


def test_acceleration_roundtrip():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=64,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    a_phys = 0.0012

    a_lattice = scaling.acceleration_to_lattice(a_phys)

    recovered = scaling.acceleration_to_physical(a_lattice)

    assert np.isclose(
        recovered,
        a_phys,
    )


def test_time_roundtrip():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=32,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    t_phys = 2.5

    t_lattice = scaling.time_to_lattice(t_phys)

    recovered = scaling.time_to_physical(t_lattice)

    assert np.isclose(
        recovered,
        t_phys,
    )


def test_frequency_roundtrip():

    scaling = LatticeScaling(
        length_phys=1.0,
        n_length=32,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    omega_phys = 7.5

    omega_lattice = scaling.frequency_to_lattice(omega_phys)

    recovered = scaling.frequency_to_physical(omega_lattice)

    assert np.isclose(
        recovered,
        omega_phys,
    )


def test_diffusive_scaling():

    coarse = LatticeScaling(
        length_phys=1.0,
        n_length=32,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    fine = LatticeScaling(
        length_phys=1.0,
        n_length=64,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    # Doubling resolution halves dx.
    assert np.isclose(
        fine.dx,
        coarse.dx / 2.0,
    )

    # Diffusive scaling:
    #
    # dt ~ dx^2
    #
    # therefore dt decreases by factor 4.
    assert np.isclose(
        fine.dt,
        coarse.dt / 4.0,
    )


def test_lattice_velocity_scales_linearly_with_dx():

    u_phys = 0.02

    coarse = LatticeScaling(
        length_phys=1.0,
        n_length=32,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    fine = LatticeScaling(
        length_phys=1.0,
        n_length=64,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    u_coarse = coarse.velocity_to_lattice(u_phys)

    u_fine = fine.velocity_to_lattice(u_phys)

    # Since dt ~ dx^2:
    #
    # u_LB ~ dt/dx ~ dx.
    assert np.isclose(
        u_fine,
        u_coarse / 2.0,
    )


def test_reynolds_number_is_resolution_independent():

    u_phys = 0.03

    coarse = LatticeScaling(
        length_phys=1.0,
        n_length=32,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    fine = LatticeScaling(
        length_phys=1.0,
        n_length=128,
        nu_phys=1.0e-3,
        tau=0.8,
    )

    re_coarse = coarse.reynolds_number(u_phys)

    re_fine = fine.reynolds_number(u_phys)

    assert np.isclose(
        re_coarse,
        re_fine,
    )
