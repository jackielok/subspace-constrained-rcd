#!/usr/bin/env python3

import numpy as np
import scipy as sp
from timeit import default_timer as timer

import sc_rcd

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(script_dir, "RPCholesky")))
import matrix

class SCRCD():
    def __init__(self, A, b, x=None, **kwargs):
        self.A = A
        self.b = b
        self.n = A.shape[0]
        self.implicitA = True if isinstance(A, matrix.FunctionMatrix) else False
        self.x = x.copy() if x is not None else np.zeros(self.n)
        self.n_iter = 0
        self.rng = kwargs["rng"] if ("rng" in kwargs) else np.random.default_rng()
        self.l = kwargs["l"] if ("l" in kwargs) else 1  # block size for coordinate descent
        self.update_times = []
        self.num_queries = []
        self.rnorms = []
        self.rnorms_iters = []
        if "I" in kwargs and "F" in kwargs:
            self.set_chol(I=kwargs["I"], F=kwargs["F"])  # indices/Cholesky factor for Nystrom approx
        else:
            self.set_chol(I=None, F=None)  # default None for no subspace constraint
        self.x_init = self.x.copy()

    def set_chol(self, I, F):
        ### Computes RPCholesky approximation of A (if indices I is not empty),
        ### and initializes using partial Cholesky factors
        if (I is not None) and (len(I) == 0):
            I = None
            F = None
    
        self.I = I
        self.F = F
            
        # For subspace-constrained randomized coordinate descent
        if I is not None:
            self.indices = [i for i in range(self.n) if i not in I]
            if self.implicitA:
                self.diags = np.maximum(self.A.diag() - np.sum(F * F, axis=1), 0)
            else:
                # Explicitly compute residual matrix instead of computing on demand
                if not sp.sparse.issparse(self.A.matrix):
                    self.Ares = np.zeros(self.A.shape, order='F')  # column-ordering
                    self.Ares[:,:] = self.A.matrix - F @ F.T
                else:
                    self.Ares = self.A.matrix - F @ F.T
                self.Ares = matrix.PSDMatrix(self.Ares)
                self.diags = np.maximum(self.Ares.diag(), 0)
            self.probs = self.diags / np.sum(self.diags[self.indices])
            
            ts = timer()
            qs = self.A.num_queries()

            # Compute auxiliary matrix B = pinv(A[I,I]) @ A[I,:] using partial Cholesky factors
            self.B = sc_rcd.compute_B_matrix(I, F)
            # Compute initial iterate satisfying A[I,:]x = b[I] using partial Cholesky factors
            sc_rcd.project_solution_space(self.x, self.A, self.b, I, F)
            # Compute residual vector r = A @ x - b
            self.update_r()
            
            te = timer()
            qe = self.A.num_queries()
            self.update_times.append(te - ts)
            self.num_queries.append(qe - qs)
            self.update_rnorms()
        
        # For randomized coordinate descent with no subspace constraint
        else:    
            self.indices = [i for i in range(self.n)]
            if self.implicitA:
                self.diags = np.maximum(self.A.diag(), 0)
            else:
                # Explicitly compute residual matrix instead of computing on demand
                self.Ares = self.A
                self.diags = np.maximum(self.Ares.diag(), 0)
            self.probs = self.diags / np.sum(self.diags)
            
            ts = timer()
            qs = self.A.num_queries()

            self.B = None
            # Compute residual vector
            self.update_r()
            
            te = timer()
            qe = self.A.num_queries()
            self.update_times.append(te - ts)
            self.num_queries.append(qe - qs)
            self.update_rnorms()

    def set_l(self, l):
        self.l = l

    def update_r(self):
        self.r = self.A @ self.x - self.b

    def update_rnorms(self):
        self.rnorms.append(np.linalg.norm(self.r))
        self.rnorms_iters.append(self.n_iter)

    def sample_block(self, uniform=False, with_replace=False):
        ### Samples a block of coordinates with/without replacement
        if uniform:
            J = self.rng.choice(self.indices, size=self.l, replace=with_replace)
        else:
            J = self.rng.choice(self.indices, size=self.l, p=self.probs[self.indices], replace=with_replace)
        # Remove duplicates
        if self.l > 1 and with_replace:
            J = np.unique(J)  
        return J

    def update(self, J, method="direct", update_stats=True):
        ### Performs a singular iteration of SC-RCD using the given set of indices J
        if update_stats:
            ts = timer()
            qs = self.A.num_queries()

        if self.implicitA:
            # Entries in A and Ares are evaluated
            if self.I is not None:
                AresJJ = self.A[np.ix_(J,J)] - self.F[J,:] @ self.F[J,:].T
                if len(J) == 1:
                    # Need to be careful with shape of matrix to avoid flattening if the list J has a single index
                    sc_rcd.sc_rcd_update(self.x, self.A[:,J].reshape(self.n, 1), AresJJ, self.r, self.I, self.F, J, self.B, method, self.implicitA)
                else:
                    sc_rcd.sc_rcd_update(self.x, self.A[:,J], AresJJ, self.r, self.I, self.F, J, self.B, method, self.implicitA)
            else:  # No subspace constraint
                if len(J) == 1:
                    sc_rcd.sc_rcd_update(self.x, self.A[:,J].reshape(self.n, 1), self.A[np.ix_(J,J)], self.r, self.I, self.F, J, self.B, method, self.implicitA)
                else:
                    sc_rcd.sc_rcd_update(self.x, self.A[:,J], self.A[np.ix_(J,J)], self.r, self.I, self.F, J, self.B, method, self.implicitA)
        else:
            # Input A and Ares is stored in memory
            if len(J) == 1:
                sc_rcd.sc_rcd_update(self.x, self.Ares[:,J].reshape(self.n, 1), self.Ares[np.ix_(J,J)], self.r, self.I, self.F, J, self.B, method, self.implicitA)
            else:
                sc_rcd.sc_rcd_update(self.x, self.Ares[:,J], self.Ares[np.ix_(J,J)], self.r, self.I, self.F, J, self.B, method, self.implicitA)
        
        self.n_iter += 1
        if update_stats:
            te = timer()
            qe = self.A.num_queries()
            self.update_times.append(te - ts)
            self.num_queries.append(qe - qs)
            self.update_rnorms()

class PCG():
    def __init__(self, A, b, x=None, precinv=None, **kwargs):
        self.A = A
        self.b = b
        self.n = A.shape[0]
        self.x = x.copy() if x is not None else np.zeros(self.n)
        self.x_init = self.x.copy()
        self.n_iter = 0
        self.l = self.n
        # Preconditioner P: precinv is a function such that precinv(r) returns solution to Pz = r
        self.precinv = precinv if precinv is not None else PCG.identity
        self.update_times = []
        self.num_queries = []
        self.rnorms = []
        self.rnorms_iters = []

        ts = timer()
        qs = self.A.num_queries()

        # Initialize residual
        self.r = self.b - self.A @ self.x  # track actual residual for comparison purposes
        self.r_iter = self.r.copy()        # residual vector maintained throughout CG
        self.update_rnorms()
        # Initialize iterates
        self.z = self.precinv(self.r)
        self.p = self.z

        te = timer()
        qe = self.A.num_queries()
        self.update_times.append(te - ts)
        self.num_queries.append(qe - qs)

    @staticmethod
    def identity(x):
        return x
    
    def update_r(self):
        self.r = self.b - self.A @ self.x

    def update_rnorms(self):
        self.rnorms.append(np.linalg.norm(self.r))
        self.rnorms_iters.append(self.n_iter)

    def rpcholesky_pinv(self, F, lamb):
        ### Forms RPCholesky preconditioner given pivoted partial Cholesky factor F coresponding to
        ### Nystrom approximation of input kernel matrix and regularization parameter lamb > 0
        assert lamb > 0, "Positive regularization parameter lamb required"
        self.F = F
        self.lamb = lamb

        ts = timer()

        self.U, self.S, _ = np.linalg.svd(F, full_matrices=False)
        self.d = 1/(self.S**2 + self.lamb) - 1/self.lamb
        self.precinv = lambda x: self.U @ (self.d * (self.U.T @ x)) + x/self.lamb
        # Initialize iterates
        self.z = self.precinv(self.r_iter)
        self.p = self.z

        te = timer()
        self.update_times[-1] += (te - ts)  # update time taken for latest step

    def update(self):
        ### Performs a single iteration of PCG
        ts = timer()
        qs = self.A.num_queries()
        
        self.v = self.A @ self.p
        self.zr = np.dot(self.z, self.r_iter)
        self.mu = self.zr / np.dot(self.p, self.v)
        self.x += self.mu * self.p
        self.r_iter -= self.mu * self.v
        self.z = self.precinv(self.r_iter)  # preconditioner solve
        self.tau = np.dot(self.z, self.r_iter) / self.zr
        self.p = self.z + self.tau * self.p

        te = timer()
        qe = self.A.num_queries()
        self.update_times.append(te - ts)
        self.num_queries.append(qe - qs)
        self.n_iter += 1
        # Update actual residual vector and corresponding norm for comparison purposes
        self.update_r()
        self.update_rnorms()
