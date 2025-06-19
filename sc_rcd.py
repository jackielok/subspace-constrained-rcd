#!/usr/bin/env python3

import numpy as np
import scipy as sp

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(script_dir, "RPCholesky")))
from rpcholesky import rpcholesky

def project_solution_space(x, A, b, I, F):
    """
    Project onto subspace constraint A[I,:] @ x = b[I] given current x and residual r[I] = (Ax - b)[I]
    Modifies x in place
    """
    if not sp.sparse.issparse(F):
        w = sp.linalg.solve_triangular(F[I,:], A[I,:] @ x - b[I], lower=True)  # F is lower triangular factor
        alpha = sp.linalg.solve_triangular(F[I,:].T, w, lower=False)
    else:
        X = np.tril(F[I,:].toarray(), k=0)  # enforce lower triangular structure of Cholesky factor
        w = sp.sparse.linalg.spsolve_triangular(X, A[I,:] @ x - b[I], lower=True)  # F is lower triangular factor
        alpha = sp.sparse.linalg.spsolve_triangular(X.T, w, lower=False)
        # w = sp.sparse.linalg.spsolve(F[I,:], A[I,:] @ x - b[I])  # F is lower triangular factor
        # alpha = sp.sparse.linalg.spsolve(F[I,:].T, w)
    x[I] -= alpha

def compute_B_matrix(I, F):
    ### Compute B = pinv(A[I,I]) @ A[I,:] using partial Cholesky factors
    n, d = F.shape
    indices = [i for i in range(n) if i not in I]  # [n] \ I
    B = np.zeros((d, n), order='F')  # column-ordering
    if not sp.sparse.issparse(F):
        B[:,indices] = sp.linalg.solve_triangular(F[I,:].T, F[indices,:].T, lower=False)
    else:
        X = np.tril(F[I,:].toarray(), k=0)  # enforce lower triangular structure of Cholesky factor
        B[:,indices] = sp.sparse.linalg.spsolve_triangular(X.T, F[indices,:].toarray().T, lower=False)
        # B[:,indices] = sp.sparse.linalg.spsolve(F[I,:].T, F[indices,:].T).toarray()
        B = sp.sparse.csc_matrix(B)
        # Eliminate entries that are numerically zero
        B.data[np.abs(B.data) < 10 * np.finfo(float).eps] = 0
        B.eliminate_zeros()
    # B[:,I] = np.eye(d)
    return B

def sc_rcd_update(x, AJ, AresJJ, r, I, F, J, B, method="direct", implicitA=False):
    # If implicitA is True: A is not stored in memory, AJ = A[:,J]
    # Otherwise, A is stored in memory, AJ = Ares[:,J]
    # AresJJ = Ares[J,J]
    sparse_A = sp.sparse.issparse(AJ)
    if method == "direct":
        try:
            if not sparse_A:
                alpha = sp.linalg.solve(AresJJ, r[J], assume_a="pos")
            else:
                solve_A = sp.sparse.linalg.factorized(AresJJ)  # preprocess / factor once
                alpha = solve_A(r[J])                          # fast solve
                # alpha = sp.sparse.linalg.spsolve(AresJJ, r[J])
        except (sp.linalg.LinAlgError, RuntimeError) as e:
            print(f"Exception {e} caught: finding min-norm least-squares solution instead")
            # Deal with singularity issues, can be much slower or lead to instability
            if not sparse_A:
                alpha = sp.linalg.lstsq(AresJJ, r[J])[0]
            else:
                alpha = sp.sparse.linalg.lsqr(AresJJ, r[J], atol=1e-6, btol=1e-6)[0]
                # alpha = sp.sparse.linalg.lsmr(AresJJ, r[J])[0]
    else:
        ### Inexact projection using inner iterative solver (works quite well if blocks are well-conditioned)
        # Solve using vanilla CG
        # alpha = sp.sparse.linalg.cg(AresJJ, r[J], rtol=5e-2)[0]
        # print(f"Ares[J,J] condition number: {np.linalg.cond(AresJJ)}")

        # Solve with Jacobi-preconditioned CG (normalize the diagonal)
        D_inv = 1.0 / AresJJ.diagonal()
        M = sp.sparse.linalg.LinearOperator(
            dtype=AresJJ.dtype,
            shape=AresJJ.shape,
            matvec=lambda x: D_inv * x
        )
        alpha = sp.sparse.linalg.cg(AresJJ, r[J], M=M, rtol=5e-2)[0]
        # print(f"Ares[J,J] (diagonal-normalized) condition number: {np.linalg.cond((AresJJ * np.sqrt(D_inv)).T * np.sqrt(D_inv))}")

    if I is not None:
        beta = B[:,J] @ alpha
        x[J] -= alpha
        x[I] += beta
        if implicitA:
            r -= AJ @ alpha - F @ (F[J,:].T @ alpha)
        else:
            r -= AJ @ alpha  # Argument AJ is AresJ
    else:
        x[J] -= alpha
        r -= AJ @ alpha

def sc_rcd_helper(A, b, I, F, l, n_iter=10, with_replace=False):
    """
    Subspace-constrained randomized coordinate descent (SC-RCD):
    Returns approximate solution x of Ax = b and residual vector r = Ax - b given psd matrix A, measurements vector b
    Subspace constraint is specified in the form of a set of indices I and a partial Cholesky factor F defining the Nystrom approximation A<I> = FF^T
    Each iteration projects onto a block of l additional coordinates
    """
    rng = np.random.default_rng()
    n = A.shape[0]
    
    if I is not None:
        d = len(I)
        indices = [i for i in range(n) if i not in I]  # [n] \ I

        ### Initialize
        # Solve for initial iterate satisfying A[I,:]x = b[I] using partial Cholesky factors
        x = np.zeros(n)
        project_solution_space(x, A, b, I, F)
        # Compute B = pinv(A[I,I]) @ A[I,:] using partial Cholesky factors
        B = compute_B_matrix(I, F)
        # Initialize residuals
        r = A @ x - b  # residual vector
        Ares = A - F @ F.T  # residual matrix

        diags = np.diag(Ares)
        probs = np.maximum(diags, 0)
        probs /= np.sum(probs[indices])

        ### Iterate
        for k in range(n_iter):
            # Sample block
            J = rng.choice(indices, size=l, p=probs[indices], replace=with_replace)
            # Remove duplicates
            if l > 1 and with_replace:
                J = np.unique(J)  

            sc_rcd_update(x, Ares[:,J], r, I, J, B[:,J])

    # Standard randomized coordinate descent (no subspace constraint)
    else:
        ### Initialize
        x = np.zeros(n)
        r = A @ x - b  # residual vector
        B = None

        diags = np.diag(A)
        probs = np.maximum(diags, 0)
        probs /= np.sum(probs)

        ### Iterate
        for k in range(n_iter):
            # Sample block
            J = rng.choice(range(n), size=l, p=probs, replace=with_replace)
            # Remove duplicates
            if l > 1 and with_replace:
                J = np.unique(J)  

            sc_rcd_update(x, Ares[:,J], r, I, J, B[:,J])

    return (x, r)

def sc_rcd(A, b, d, l, n_iter=10, with_replace=False):
    """
    Subspace-constrained randomized coordinate descent (SC-RCD):
    Returns approximate solution x of Ax = b and residual vector r = Ax - b given psd matrix A, measurements vector b
    Generates subspace constraint corresponding to a rank-d Nystrom approximation using RPCholesky
    Each iteration projects onto a block of l additional coordinates
    """
    if d > 0:
        Ahat = rpcholesky(A, d)
        x, r = sc_rcd_helper(A, b, I=Ahat.get_indices(), F=Ahat.get_left_factor(), l=l, n_iter=n_iter, with_replace=with_replace)
    else:
        Ahat = None
        x, r = sc_rcd_helper(A, b, I=None, F=None, l=l, n_iter=n_iter, with_replace=with_replace)
    return (x, r, Ahat)
