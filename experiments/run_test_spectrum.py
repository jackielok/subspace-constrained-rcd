#!/usr/bin/env python3
### Computes spectrum of residual/preconditioned matrices coming from kernel ridge regression
### Running on a single set of parameters (dataset, block size, approximation rank)
### Allows for parallelization over the selected dataset

import numpy as np
import scipy as sp
import pandas as pd
import test_helper
import argparse

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "RPCholesky")))
from rpcholesky import rpcholesky

#################### Inputs ####################

### Locations for input data and saving output data/figures
script_dir = os.path.dirname(os.path.abspath(__file__))
data_loc = os.path.join(script_dir, "..", "data/preprocessed") + os.sep
out_loc = "./output/"

# List of selected datasets
selected_datasets = list(test_helper.datasets.keys())

n = 100000                 # number of datapoints to sample
kernel = "gaussian"        # supported kernels: {"gaussian", "matern", "laplace"}
bandwidth = 3              # kernel bandwidth parameter
implicit_kernel = True     # evaluate entries implicitly (True) or store array in memory
lamb = 1e-9 * n            # regularization parameter
d = 1000                   # approximation rank for RPCholesky
top_eigvals = 20000        # returns leading 1 <= top_eigvals < n eigenvalues
save_data = True

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Define functions ####################

def run_test_spectrum(
    dataset_loc,    # dataset location and file name (preprocessed .mat file)
    dataset_id,     # string to identify dataset
    n = n,
    kernel = kernel,
    bandwidth = bandwidth,
    implicit_kernel = implicit_kernel,
    lamb = lamb,
    d = d,
    top_eigvals = top_eigvals,
    save_data = save_data,
    out_loc = out_loc,
    rng = rng,
):
    print("========================================")
    print(f"Loading dataset {dataset_id} (n = {n}, d = {d})...")
    
    Xtrain, ytrain, scaler, K, D = test_helper.load_data(dataset_loc=dataset_loc, n=n, implicit_kernel=implicit_kernel, lamb=lamb, kernel=kernel, bandwidth=bandwidth, rng=rng)
    
    print(f"Dataset {dataset_id} loaded.")

    print("========================================")
    print(f"Running RPCholesky: d = {d}...")

    ### Compute low-rank approximations
    # Use the same random approximation in each sample
    # Low rank approximation for regularized kernel
    Khat = rpcholesky(K + D, d, rng=rng, b=int(np.ceil(d / 10)))  # default: b = "auto" (outcome is random)
    I = Khat.get_indices()
    F = Khat.get_left_factor()
    print(f"Selected indices: I = {I}")

    # Low rank approximation for kernel only (for CG preconditioner)
    Khat2 = rpcholesky(K, d, rng=rng, b=int(np.ceil(d / 10)))  # default: b = "auto" (outcome is random)
    I2 = Khat2.get_indices()
    F2 = Khat2.get_left_factor()
    print(f"Selected indices: I2 = {I2}")

    print("RPCholesky completed.")

    ### Compute eigenvalues
    print("========================================")
    print(f"Computing eigenvalues...")
    
    if not implicit_kernel:
        ### Compute using explicitly stored matrices
        KD = np.array(K[:,:], order='C')
        KD[range(n), range(n)] += lamb

        # Original matrix
        KD_evals = sp.sparse.linalg.eigsh(KD, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]
        # KD_evals = sp.linalg.eigh(KD, eigvals_only=True, subset_by_index=[n-top_eigvals, n-1])[::-1]

        # Residual matrix (SCRCD)
        KDres = KD - F @ F.T
        KDres_evals = sp.sparse.linalg.eigsh(KDres, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]
        # KDres_evals = sp.linalg.eigh(KDres, eigvals_only=True, subset_by_index=[n-top_eigvals, n-1])
        # Filter eigenvalues that are zero up to machine tolerance
        KDres_evals[KDres_evals < 100 * np.finfo(float).eps] = 0
        KDres_evals = KDres_evals[::-1]

        # Preconditioned matrix (PCG)
        U, S, _ = np.linalg.svd(F2, full_matrices=False)
        sqrtlamb = np.sqrt(lamb)
        tmp = 1/np.sqrt(S**2 + lamb) - 1/sqrtlamb
        KDprec = U @ (tmp[:,None] * (U.T @ KD)) + (KD / sqrtlamb)  # left multiply
        KDprec = (U @ (tmp[:,None] * (U.T @ KDprec.T)) + (KDprec.T / sqrtlamb)).T  # right multiply
        KDprec = lamb * KDprec
        KDprec_evals = sp.sparse.linalg.eigsh(KDprec, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]
        # KDprec_evals = sp.linalg.eigh(KDprec, eigvals_only=True, subset_by_index=[n-top_eigvals, n-1])[::-1]

    else:
        ### Compute using operator A
        # Original matrix
        KD = sp.sparse.linalg.LinearOperator((n, n), matvec=lambda x: K @ x + lamb * x)
        KD_evals = sp.sparse.linalg.eigsh(KD, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]
        
        # Residual matrix (SCRCD)
        KDres = sp.sparse.linalg.LinearOperator((n, n), matvec=lambda x: K @ x + lamb * x - F @ (F.T @ x))
        KDres_evals = sp.sparse.linalg.eigsh(KDres, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]

        # Preconditioned matrix (PCG)
        U, S, _ = np.linalg.svd(F2, full_matrices=False)
        sqrtlamb = np.sqrt(lamb)
        tmp = 1/np.sqrt(S**2 + lamb) - 1/sqrtlamb
        precinvsqrt = lambda x: U @ (tmp * (U.T @ x)) + x/sqrtlamb
        def KDprec_matvec(x):
            u = precinvsqrt(x)
            return lamb * precinvsqrt(K @ u + lamb * u)
        KDprec = sp.sparse.linalg.LinearOperator((n, n), matvec=KDprec_matvec)
        KDprec_evals = sp.sparse.linalg.eigsh(KDprec, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]

    ### Save data
    if save_data:
        print("========================================")
        print(f"Saving data to {out_loc + dataset_id}...")
    
        pd.DataFrame({
            "KD": KD_evals,
            "KDresidual": KDres_evals,
            "KDpreconditioned": KDprec_evals
        }).to_csv(f"{out_loc + dataset_id}_eigvals.csv", index=False)

        print("Data saved.")

    print("========================================")
    print(f"Dataset {dataset_id} completed (n = {n}, d = {d}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Positional (required) arguments
    parser.add_argument("job_idx", type=int, help="Index of current job (selected dataset)")

    # Optional arguments
    parser.add_argument("--selected_datasets", nargs="+", type=str, default=selected_datasets, help="List of selected datasets (space-separated)")
    parser.add_argument("--n", type=int, default=n, help="Number of datapoints to sample")
    parser.add_argument("--d", type=int, default=d, help="Approximation rank for RPCholesky")
    parser.add_argument("--kernel", type=str, default=kernel, help="Kernel function used")
    parser.add_argument("--bandwidth", type=float, default=bandwidth, help="Kernel bandwidth parameter")
    parser.add_argument("--implicit_kernel", type=lambda x: x.lower() == "true", default=implicit_kernel, help="evaluate entries implicitly (True) or store array in memory (False)")
    parser.add_argument("--lamb", type=float, default=lamb, help="Regularization parameter")
    parser.add_argument("--top_eigvals", type=int, default=top_eigvals, help="Number of leading eigenvalues to compute")

    args = parser.parse_args()

    dataset_id = args.selected_datasets[args.job_idx]
    dataset_loc = data_loc + test_helper.datasets[dataset_id]
    
    run_test_spectrum(
        dataset_loc = dataset_loc,
        dataset_id = dataset_id,
        n = args.n,
        d = args.d,
        kernel = args.kernel,
        bandwidth = args.bandwidth,
        implicit_kernel = args.implicit_kernel,
        lamb = args.lamb,
        top_eigvals = args.top_eigvals
    )
