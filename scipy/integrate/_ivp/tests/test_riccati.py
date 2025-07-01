import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_raises

from scipy.integrate.ipv.riccati import solve_riccati

# FIXME use pytest like in test_rk.py?

def test_real_input():
    a = np.array([[1, 1], [2, 1]])
    b = np.array([[1], [1]])
    q = np.array([[2, 1], [1, 1]])
    r = np.array([[1]])
    pt = np.array([[1, 1], [1, 1]])
    tspan = [0, 5]
    t_eval = np.linspace(0, 5, 10)

    t, p = solve_riccati(a, b, q, r, pt, tspan, t_eval=t_eval)

    assert len(t) == len(t_eval)
    for Pi in p:
        assert_array_almost_equal(Pi, Pi.conj().T)
        eigvals = np.linalg.eigvalsh(Pi)
        assert np.all(eigvals >= -1e-10)


def test_complex_input():
    a = np.array([[1 + 1j, 0], [0, 2 - 1j]])
    b = np.array([[1], [1j]])
    q = np.eye(2)
    r = np.eye(1)
    pt = np.eye(2)
    tspan = [0, 5]
    t_eval = np.linspace(0, 5, 10)

    t, p = solve_riccati(a, b, q, r, pt, tspan, t_eval=t_eval)
    assert len(t) == len(t_eval)
    for Pi in p:
        assert_array_almost_equal(Pi, Pi.conj().T)


def test_nonsymmetric_inputs():
    a = np.eye(2)
    b = np.eye(2)
    q = np.array([[2, 5], [0, 1]])  # Not symmetric
    r = np.eye(2)
    pt = np.array([[3, -1], [4, 2]])  # Not symmetric
    tspan = [0, 1]
    t_eval = np.linspace(0, 1, 5)

    t, p = solve_riccati(a, b, q, r, pt, tspan, t_eval=t_eval)
    for Pi in p:
        assert_array_almost_equal(Pi, Pi.T)


def test_invalid_dimensions():
    a = np.eye(2)
    b = np.eye(3)
    q = np.eye(2)
    r = np.eye(2)
    pt = np.eye(2)
    tspan = [0, 1]

    with pytest.raises(ValueError):
        solve_riccati(a, b, q, r, pt, tspan)


def test_invalid_tspan():
    a = b = q = r = pt = np.eye(2)
    with pytest.raises(ValueError):
        solve_riccati(a, b, q, r, pt, [10, 0])


def test_non_positive_r():
    a = np.eye(2)
    b = np.eye(2)
    q = np.eye(2)
    r = np.array([[0, 1], [1, -1]])
    pt = np.eye(2)
    tspan = [0, 1]

    with pytest.raises(np.linalg.LinAlgError):
        solve_riccati(a, b, q, r, pt, tspan)


def test_non_symmetric_q():
    a = np.eye(2)
    b = np.ones((2, 1))
    q = np.array([[1, 2], [3, 4]])
    r = np.eye(1)
    pt = np.eye(2)
    tspan = [0, 5]

    with pytest.raises(ValueError):
        solve_riccati(a, b, q, r, pt, tspan)
