import numpy as np

from lbm.boundary.axisymmetric import (
    stream_axisymmetric_pipe,
)
from lbm.lattices.d2q9 import Q


def test_axis_normal_reflection():

    nz, nr = 6, 5

    f = np.zeros((Q, nz, nr))

    f[4, 3, 0] = 1.0

    f_streamed = stream_axisymmetric_pipe(f)

    assert f_streamed[2, 3, 0] == 1.0


def test_axis_specular_diagonal_reflection():

    nz, nr = 6, 5

    f = np.zeros((Q, nz, nr))

    # (-z, -r)
    f[7, 3, 0] = 1.0

    f_streamed = stream_axisymmetric_pipe(f)

    # axial component unchanged:
    #
    # (-z, -r) -> (-z, +r)
    #
    # therefore 7 -> 6.
    assert f_streamed[6, 3, 0] == 1.0


def test_axis_other_diagonal():

    nz, nr = 6, 5

    f = np.zeros((Q, nz, nr))

    # (+z, -r)
    f[8, 3, 0] = 1.0

    f_streamed = stream_axisymmetric_pipe(f)

    # (+z, -r) -> (+z, +r)
    #
    # 8 -> 5
    assert f_streamed[5, 3, 0] == 1.0


def test_outer_wall_bounceback():

    nz, nr = 6, 5

    f = np.zeros((Q, nz, nr))

    f[5, 3, nr - 1] = 1.0

    f_streamed = stream_axisymmetric_pipe(f)

    # Full opposite direction:
    #
    # 5 -> 7
    assert (
        f_streamed[
            7,
            3,
            nr - 1,
        ]
        == 1.0
    )


def test_axial_periodicity():

    nz, nr = 6, 5

    f = np.zeros((Q, nz, nr))

    f[1, nz - 1, 2] = 1.0

    f_streamed = stream_axisymmetric_pipe(f)

    assert f_streamed[1, 0, 2] == 1.0


def test_axisymmetric_streaming_preserves_mass():

    nz, nr = 8, 7

    rng = np.random.default_rng(1234)

    f = rng.random((Q, nz, nr))

    mass_before = np.sum(f)

    f_streamed = stream_axisymmetric_pipe(f)

    mass_after = np.sum(f_streamed)

    assert np.isclose(
        mass_after,
        mass_before,
    )
