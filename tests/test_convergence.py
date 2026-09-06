import numpy as np

from lbm.utils.convergence import relative_change


def test_relative_change_zero():
    a = np.ones((4, 4))

    assert relative_change(a, a) == 0.0


def test_relative_change_known_value():
    old = np.ones((2, 2))
    new = 2.0 * np.ones((2, 2))

    # ||new-old|| = 2
    # ||new||     = 4
    #
    # relative change = 0.5
    assert np.isclose(
        relative_change(new, old),
        0.5,
    )
