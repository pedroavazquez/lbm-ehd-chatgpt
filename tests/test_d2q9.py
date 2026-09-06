import numpy as np

from lbm.lattices.d2q9 import CS2, CS4, OPPOSITE, C, D, Q, W


def test_dimensions():
    assert C.shape == (Q, D)
    assert W.shape == (Q,)
    assert OPPOSITE.shape == (Q,)


def test_weights_sum_to_one():
    assert np.isclose(np.sum(W), 1.0)


def test_first_order_isotropy():
    first_moment = np.einsum("i,ia->a", W, C)

    expected = np.zeros(D)

    assert np.allclose(first_moment, expected)


def test_second_order_isotropy():
    second_moment = np.einsum("i,ia,ib->ab", W, C, C)

    expected = CS2 * np.eye(D)

    assert np.allclose(second_moment, expected)


def test_third_order_isotropy():
    third_moment = np.einsum("i,ia,ib,ic->abc", W, C, C, C)

    expected = np.zeros((D, D, D))

    assert np.allclose(third_moment, expected)


def test_fourth_order_isotropy():
    fourth_moment = np.einsum(
        "i,ia,ib,ic,id->abcd",
        W,
        C,
        C,
        C,
        C,
    )

    delta = np.eye(D)

    expected = CS4 * (
        np.einsum("ab,cd->abcd", delta, delta)
        + np.einsum("ac,bd->abcd", delta, delta)
        + np.einsum("ad,bc->abcd", delta, delta)
    )

    assert np.allclose(fourth_moment, expected)


def test_opposite_velocities():
    for i in range(Q):
        assert np.array_equal(C[OPPOSITE[i]], -C[i])


def test_opposite_is_involution():
    for i in range(Q):
        assert OPPOSITE[OPPOSITE[i]] == i
