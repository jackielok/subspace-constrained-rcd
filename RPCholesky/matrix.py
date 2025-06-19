#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod
import numbers
from kernels import MaternKernel, MaternKernel_vec, MaternKernel_mtx, GaussianKernel, GaussianKernel_vec, GaussianKernel_mtx, LaplaceKernel, LaplaceKernel_vec, LaplaceKernel_mtx
from functools import partial
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

class AbstractPSDMatrix(ABC):

    def __init__(self, **kwargs):
        self.queries = 0
        self.count_queries = kwargs['count_queries'] if ('count_queries' in kwargs) else True
        self.dpp_stuff = None

    @abstractmethod
    def _diag_helper(self, *args):
        pass

    @abstractmethod 
    def _getitem_helper(self, *args):
        pass

    def __getitem__(self, *args):
        to_return = self._getitem_helper(*args)
        if isinstance(to_return, np.ndarray):
            self.queries += to_return.size
        else:
            self.queries += 1
        return to_return

    def diag(self, *args):
        to_return = self._diag_helper(*args)
        if isinstance(to_return, np.matrix):
            to_return = np.array(to_return).ravel()
        if isinstance(to_return, np.ndarray):
            self.queries += to_return.size
        else:
            self.queries += 1
        return to_return
    
    def __call__(self, *args):
        return self.__getitem__(*args)

    def trace(self):
        return sum(self.diag())

    def num_queries(self):
        return self.queries

    def reset(self):
        self.queries = 0
        self.dpp_stuff = None

    def to_matrix(self):
        return PSDMatrix(self[:,:])

    def _clean_index_input(self, args):
        idx = args[0]
        if len(idx) == 1:
            idx = [idx,idx]
        else:
            idx = list(idx)
            assert len(idx) == 2

        for j in range(2):
            if isinstance(idx[j], np.ndarray):
                idx[j] = idx[j].ravel().tolist()
            elif isinstance(idx[j], slice):
                idx[j] = list(range(self.shape[0]))[idx[j]]
            elif isinstance(idx[j], list):
                pass
            elif isinstance(idx[j], numbers.Integral):
                idx[j] = [idx[j]]
            else:
                raise RuntimeError("Indexing not implemented with index of type {}".format(type(idx)))

        return idx
        
class PSDMatrix(AbstractPSDMatrix):

    def __init__(self, A, **kwargs):
        super().__init__(**kwargs)
        self.matrix = A
        self.shape = A.shape

    def _diag_helper(self, *args):
        if len(args) > 0:
            X = list(args[0])
        else:
            X = range(self.matrix.shape[0])
        return self.matrix[X,X]

    def _getitem_helper(self, *args):
        idx = self._clean_index_input(args)
        
        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self.matrix[idx[0][0], idx[1][0]]
        else:
            mtx = self.matrix[np.ix_(idx[0],idx[1])]
            if len(idx[0]) == 1: return mtx[0,:]
            elif len(idx[1]) == 1: return mtx[:,0]
            else: return mtx

    def __add__(self, other):
        if isinstance(other, PSDMatrix):
            return PSDMatrix(self.matrix + other.matrix)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'PSDMatrix' and '{}'".format(type(other).__name__))
        
    def __sub__(self, other):
        if isinstance(other, PSDMatrix):
            return PSDMatrix(self.matrix - other.matrix)
        else:
            raise TypeError("Unsupported operand type(s) for -: 'PSDMatrix' and '{}'".format(type(other).__name__))

    def __matmul__(self, other):
        return self.matrix @ other

    def __rmatmul__(self, other):
        return (self @ other.T).T

class FunctionMatrix(AbstractPSDMatrix):

    def __init__(self, n, **kwargs):
        self.shape = (n,n)
        super().__init__(**kwargs)

    @abstractmethod
    def _function(self,i,j):
        pass
    
    @abstractmethod
    def _function_vec(self,vec_i,vec_j):
        pass

    @abstractmethod
    def _function_mtx(self,vec_i,vec_j):
        pass
    
    # Allows for A @ x when A is given as an implicit kernel matrix
    def __matmul__(self, other):
        if other.ndim == 1:
            b = np.zeros(self.shape[0])
        else:
            b = np.zeros((self.shape[0], other.shape[1]))
        j = 0
        while j < self.shape[0]:
            m = min(1000, self.shape[0] - j)
            b[j:j+m] = self[j:j+m,:] @ other
            j += m
        return b

    def __rmatmul__(self, other):
        return (self @ other.T).T

    def _diag_helper(self, *args):
        if len(args) > 0:
            idx = list(args[0])
        else:
            idx = range(self.shape[0])
        return self._function_vec(idx,idx)
    
    def _getitem_helper(self, *args):
        idx = self._clean_index_input(args)
        
        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self._function(idx[0][0], idx[1][0])
        else:
            mtx = self._function_mtx(idx[0],idx[1])
            if len(idx[0]) == 1: return mtx[0,:]
            elif len(idx[1]) == 1: return mtx[:,0]
            else: return mtx

    def __add__(self, other):
        if isinstance(other, FunctionMatrix):
            return SumFunctionMatrix(self, other)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'FunctionMatrix' and '{}'".format(type(other).__name__))

class SumFunctionMatrix(FunctionMatrix):

    def __init__(self, M1, M2, **kwargs):
        assert M1.shape == M2.shape, "Matrices must have the same shape"
        super().__init__(M1.shape[0], **kwargs)
        self.M1 = M1
        self.M2 = M2

    def _function(self,i,j):
        return self.M1._function(i,j) + self.M2._function(i,j)

    def _function_vec(self,vec_i,vec_j):
        return self.M1._function_vec(vec_i,vec_j) + self.M2._function_vec(vec_i,vec_j)

    def _function_mtx(self,vec_i,vec_j):
        return self.M1._function_mtx(vec_i,vec_j) + self.M2._function_mtx(vec_i,vec_j)

class DiagonalFunctionMatrix(FunctionMatrix):

    def __init__(self, diag, **kwargs):
        super().__init__(len(diag), **kwargs)
        self.d = np.array(diag)

    def _function(self,i,j):
        return self.d[i] if i == j else 0
    
    def _function_vec(self,vec_i,vec_j):
        result = np.zeros(len(vec_i))
        vec_i = np.array(vec_i)
        vec_j = np.array(vec_j)
        mask = (vec_i == vec_j)
        result[mask] = self.d[vec_i[mask]]
        return result

    def _function_mtx(self,vec_i,vec_j):
        result = np.zeros((len(vec_i), len(vec_j)))
        vec_i = np.array(vec_i)
        vec_j = np.array(vec_j)
        mask = (vec_i[:,None] == vec_j[None,:])
        result[mask.any(axis=1)] = mask[mask.any(axis=1)] * self.d[vec_i[mask.any(axis=1)][:,None]]
        return result
    
    def __matmul__(self, other):
        return np.array([self.d[i] * other[i] for i in range(other.shape[0])])
    
    def __rmatmul__(self, other):
        return (self @ other.T).T

class KernelMatrix(FunctionMatrix):
    
    @staticmethod
    def median_trick(X, kernel):
        if kernel in ["gaussian", "matern"]:
            dists = euclidean_distances(X,X)
        elif kernel in "laplace":
            dists = manhattan_distances(X,X)
        else:
            raise RuntimeError(f"Median trick is not implement for kernel {kernel}")

        return np.median(dists)
            
    @staticmethod
    def kernel_from_input(kernel, bandwidth = 1.0, extra_stability = False, **kwargs):
        if isinstance(kernel, str):
            if kernel == 'gaussian':
                return partial(GaussianKernel, bandwidth=bandwidth, extra_stability=extra_stability), partial(GaussianKernel_vec, bandwidth=bandwidth, extra_stability=extra_stability), partial(GaussianKernel_mtx, bandwidth=bandwidth, extra_stability=extra_stability)
            elif kernel == 'matern':
                return partial(MaternKernel, bandwidth=bandwidth,nu=kwargs["nu"], extra_stability=extra_stability), partial(MaternKernel_vec, bandwidth=bandwidth,nu=kwargs["nu"], extra_stability=extra_stability), partial(MaternKernel_mtx, bandwidth=bandwidth,nu=kwargs["nu"], extra_stability=extra_stability)
            elif kernel == 'laplace':
                return partial(LaplaceKernel, bandwidth=bandwidth, extra_stability=extra_stability), partial(LaplaceKernel_vec, bandwidth=bandwidth, extra_stability=extra_stability), partial(LaplaceKernel_mtx, bandwidth=bandwidth, extra_stability=extra_stability)
                
            else:
                raise RuntimeError("Kernel name {} not recognized".format(kernel))
        else:
            return kernel
    
    def __init__(self, X, kernel = "gaussian", bandwidth = 1.0, **kwargs):
        super().__init__(X.shape[0],**kwargs)
        if bandwidth == "median":
            self.bandwidth = KernelMatrix.median_trick(X, kernel)
        elif bandwidth == "approx_median":
            idx = np.random.choice(X.shape[0], size = min(X.shape[0],1000), replace=False)
            self.bandwidth = KernelMatrix.median_trick(X[idx,:], kernel)
        else:
            self.bandwidth = bandwidth
        self.data = X
        kernel, kernel_vec, kernel_mtx = KernelMatrix.kernel_from_input(kernel, bandwidth = self.bandwidth, **kwargs)
        self.kernel = kernel
        self.kernel_vec = kernel_vec
        self.kernel_mtx = kernel_mtx        
        
    def _function(self, i, j):
        return self.kernel(self.data[i,:], self.data[j,:])
    
    def _function_vec(self,vec_i,vec_j):
        return self.kernel_vec(self.data[vec_i,:], self.data[vec_j,:])
    
    def _function_mtx(self,vec_i,vec_j):
        return self.kernel_mtx(self.data[vec_i,:], self.data[vec_j,:])

    def out_of_sample(self, Xtest, vec):
        return self.kernel_mtx(Xtest, self.data[vec, :])

class NonsymmetricKernelMatrix(object):
    
    def __init__(self, X, Y, kernel = "gaussian", bandwidth = 1.0, **kwargs):
        self.X = X
        self.Y = Y
        self.shape = (X.shape[0], Y.shape[0])
        self.kernel, self.kernel_vec, self.kernel_mtx = KernelMatrix.kernel_from_input(kernel, bandwidth = bandwidth, **kwargs)
        
    def _function(self, i, j):
        return self.kernel(self.X[i,:], self.Y[j,:])
    
    def _function_vec(self,vec_i,vec_j):
        return self.kernel_vec(self.X[vec_i,:], self.Y[vec_j,:])
    
    def _function_mtx(self,vec_i,vec_j):
        return self.kernel_mtx(self.X[vec_i,:], self.Y[vec_j,:])

    # Allows for A @ x when A is given as an implicit kernel matrix
    def __matmul__(self, other):
        if other.ndim == 1:
            b = np.zeros(self.shape[0])
        else:
            b = np.zeros((self.shape[0], other.shape[1]))
        j = 0
        while j < self.shape[0]:
            m = min(2000, self.shape[0] - j)
            b[j:j+m] = self[j:j+m,:] @ other
            j += m
        return b

    def __rmatmul__(self, other):
        return (self @ other.T).T

    def _getitem_helper(self, *args):
        idx = args[0]
        if len(idx) == 1:
            idx = [idx,idx]
        else:
            idx = list(idx)

        for j in range(2):
            if isinstance(idx[j], np.ndarray):
                idx[j] = idx[j].ravel().tolist()
            elif isinstance(idx[j], slice):
                idx[j] = list(range(self.shape[j]))[idx[j]]
            elif isinstance(idx[j], list):
                pass
            elif isinstance(idx[j], numbers.Integral):
                idx[j] = [idx[j]]
            else:
                raise RuntimeError("Indexing not implemented with index of type {}".format(type(idx)))

        if len(idx[0]) == 1 and len(idx[1]) == 1:
            return self._function(idx[0], idx[1])
        else:
            mtx = self._function_mtx(idx[0],idx[1])
            if len(idx[0]) == 1: return mtx[0,:]
            elif len(idx[1]) == 1: return mtx[:,0]
            else: return mtx
            
    def __getitem__(self, *args):
        return self._getitem_helper(*args)
