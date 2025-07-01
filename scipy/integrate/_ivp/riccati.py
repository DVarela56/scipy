import numpy as np
from numpy.linalg import norm
from scipy._lib._util import _asarray_validated
from scipy.linalg import expm, solve


def solve_riccati(A, B, Q, R, PT, t_span, t_eval=None, method="davison-maki"):
    # TODO docstrings Bernardo
    # TODO implement t_eval
    
    a, b, q, r, pt, tspan, t_eval, m = _riccati_validate_args(A, B, Q, R, PT, t_span, t_eval, method)
    t0, tf = tspan
    h = t_eval[1] - t_eval[0]

    r_inv = np.linalg.inv(r)

    # Construct Hamiltonian matrix
    H = np.block([
        [a, -b @ r_inv @ b.T],
        [-q, -a.T]
    ])
    
    # Initialize the solution matrixes
    p = []
    t = []
    
    # p_next = P(k+1)
    p_next = pt.copy()
    # t_next = t(k+1)
    t_next = tf
    
    theta = expm(-h * H)
    
    if norm(theta, ord=1) > 1e8: # TODO change by tol_exp
        raise ValueError("Exponential matrix norm is too large, reduce step size h")

    theta_11 = theta[:m, :m]
    theta_12 = theta[:m, m:]
    theta_21 = theta[m:, :m]
    theta_22 = theta[m:, m:]    

    
    while t_next - h >= t0 - 1e-8:
        U = theta_11 + theta_12 @ p_next
        V = theta_21 + theta_22 @ p_next
        p_k = solve(U.T, V.T).T # V @ inv(U)
        
        t_next -= h
        t.append(t_next)
        p.append(p_k)
        p_next = p_k
    
    # Reverse the lists to have them in increasing order of time
    p.reverse()
    t.reverse()
    return p, t

def _riccati_validate_args(a, b, q, r, pt, tspan, t_eval, method):
    """
    Validate input arguments for the continuous-time Riccati Differential
    Equation (RDE) solver.

    This helper function ensures all inputs to the `solve_continuous_rde`
    function are valid.
    It performs a series of checks including:
        - Finite-valued array inputs;
        - Shape consistency between system and cost matrices;
        - Symmetry (Hermitian) of Q, R, and PT;
        - Positive definiteness of R;
        - Positive semi-definiteness of Q and PT;
        - Proper definition of the time span vector;
        - Verification of the cross-weight matrix S.

    Parameters
    ----------
        a : (M, M) array_like
            System matrix.
        b : (M, N) array_like
            Input matrix.
        q : (M, M) array_like
            State cost matrix.
        r : (N, N) array_like
            Control cost matrix. Must be symmetric and positive definite.
        s : (M, N) array_like or None
            Optional cross-weight matrix. If None, assumed to be zero.
        pt : (M, M) array_like
            Terminal condition matrix. Must be symmetric and positive semi-definite.
        tspan : (2,) array_like
            Time interval [t0, tf] with t0 < tf.
        # TODO Add t_eval and method parameters to docstring

    Returns
    -------
    a, b, q, r, pt, tspan : ndarray
        Regularized input data.
    m : int
        shape of the problem.

    """
    if method != "davison-maki":
        raise ValueError(f"Method {method} not implemented for solve_riccati")

    a = np.atleast_2d(_asarray_validated(a, check_finite=True))
    b = np.atleast_2d(_asarray_validated(b, check_finite=True))
    q = np.atleast_2d(_asarray_validated(q, check_finite=True))
    r = np.atleast_2d(_asarray_validated(r, check_finite=True))
    pt = np.atleast_2d(_asarray_validated(pt, check_finite=True))
    tspan = np.atleast_1d(_asarray_validated(tspan, check_finite=True))
    if t_eval is None:
        t_eval = np.linspace(tspan[0], tspan[1], 100)
    else:
        raise NotImplementedError("t_eval is not implemented yet, please use default t_eval")
    # Shape consistency checks
    m, n = b.shape
    if m != a.shape[0]:
        raise ValueError("Matrix a and b should have the same number of rows.")
    if m != q.shape[0]:
        raise ValueError("Matrix a and q should have the same shape.")
    if n != r.shape[0]:
        raise ValueError("Matrix b and r should have the same number of cols.")
    if m != pt.shape[0]:
        raise ValueError("Matrix pt and b should have the same number of rows.")

    # Check tspan size
    if tspan.shape != (2,):
        raise ValueError("tspan must be a 1D array-like of length 2 (e.g., [t0, tf])")

    if tspan[0] >= tspan[1]:
        raise ValueError("tspan[0] < tspan[1] is required")

    # Check if the data matrices q, r, pt are (sufficiently) hermitian
    for ind, mat in enumerate((q, r, pt)):
        if norm(mat - mat.conj().T, 1) > np.spacing(norm(mat, 1))*100:
            raise ValueError(f"Matrix {'qrpt'[ind]} should be symmetric/hermitian.")

    # Check if r is positive definite
    eigvals = np.linalg.eigvalsh(r)
    if not np.all(eigvals > 0):
        raise ValueError("R must be positive definite.")

    # Check if q, pt are positive semi-definite
    eigvals = np.linalg.eigvalsh(q)
    if not np.all(eigvals >= 0):
        raise ValueError("Q must be positive semi-definite.")

    eigvals = np.linalg.eigvalsh(pt)
    if not np.all(eigvals >= 0):
        raise ValueError("PT must be positive semi-definite.")

    # Check tspan size
    if tspan.shape != (2,):
        raise ValueError("tspan must be a 1D array-like of length 2 (e.g., [t0, tf])")

    return a, b, q, r, pt, tspan, t_eval, m
