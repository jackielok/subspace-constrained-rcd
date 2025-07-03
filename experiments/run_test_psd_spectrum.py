#!/usr/bin/env python3
### Computes spectrum of residual/preconditioned matrices coming from solving a general psd linear system
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
data_loc = os.path.join(script_dir, "..", "data") + os.sep
out_loc = "./output/"

# List of selected datasets
selected_datasets = ["simlowrank400"]

n = 2**13                  # size of input matrix 
d = 500                    # approximation rank for RPCholesky
top_eigvals = 8000         # returns leading 1 <= top_eigvals < n eigenvalues
save_data = True

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Define functions ####################

def run_test_psd_spectrum(
    dataset_loc,    # dataset location and file name (preprocessed .mat file)
    dataset_id,     # string to identify dataset
    n = n,
    d = d,
    top_eigvals = top_eigvals,
    save_data = save_data,
    out_loc = out_loc,
    rng = rng,
):
    print("========================================")
    print(f"Loading dataset {dataset_id} (n = {n}, d = {d})...")
    
    A, b = test_helper.load_data_psd(dataset_loc=dataset_loc, n=n, rng=rng)
    n = A.shape[0]
    
    print(f"Dataset {dataset_id} loaded.")

    print("========================================")
    print(f"Running RPCholesky: d = {d}...")

    ### Compute low-rank approximations
    Ahat = rpcholesky(A, d, rng=rng, b=int(np.ceil(d / 10)))  # default: b = "auto" (outcome is random)
    I = Ahat.get_indices()
    F = Ahat.get_left_factor()

    print("RPCholesky completed.")

    ### Initialize solvers
    print("========================================")
    print(f"Initializing solvers...")
    
    ### Compute eigenvalues
    print("========================================")
    print(f"Computing eigenvalues...")
    
    # Compute using explicitly stored matrices
    A = np.array(A[:,:], order='C')
    
    # Original matrix
    # Compute top eigenvalues only
    if top_eigvals < n:
        A_evals = sp.sparse.linalg.eigsh(A, k=top_eigvals, which="LM", return_eigenvectors=False)[::-1]
    else:
        A_evals = sp.linalg.eigh(A, eigvals_only=True)[::-1]
    
    # Residual matrix (SCRCD)
    Ares = A - F @ F.T
    if top_eigvals < n:
        Ares_evals = sp.sparse.linalg.eigsh(Ares, k=top_eigvals, which="LM", return_eigenvectors=False)
    else:
        Ares_evals = sp.linalg.eigh(Ares, eigvals_only=True)
    # Filter eigenvalues that are zero up to machine tolerance
    Ares_evals[Ares_evals < 100 * np.finfo(float).eps] = 0
    Ares_evals = Ares_evals[::-1]

    ### Save data
    if save_data:
        print("========================================")
        print(f"Saving data to {out_loc + dataset_id}...")
    
        pd.DataFrame({
            "A": A_evals,
            "Aresidual": Ares_evals,
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
    parser.add_argument("--top_eigvals", type=int, default=top_eigvals, help="Number of leading eigenvalues to compute")

    args = parser.parse_args()

    dataset_id = args.selected_datasets[args.job_idx]
    dataset_loc = data_loc + test_helper.datasets_psd[dataset_id]
    
    run_test_psd_spectrum(
        dataset_loc = dataset_loc,
        dataset_id = dataset_id,
        n = args.n,
        d = args.d,
        top_eigvals = args.top_eigvals
    )
