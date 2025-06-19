#!/usr/bin/env python3
### For solving a general psd linear system
### Allows for parallelization over different approximation ranks and over a number of samples

import numpy as np
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

# Selected dataset
selected_dataset = "simlowrank400"

n = 2**13                  # number of datapoints to sample
l = 500                    # block size for coordinate descent
method = "direct"          # method for solving inner projection for block CD in ["direct", "cg"]
n_iter_cd = 3500           # number of CD iterations
rel_rnorm_tol = -1         # terminate if relative residual norm falls below threshold
save_data = True

d_list = list(np.arange(0, 1001, step=100))  # list of approximation ranks for RPCholesky to run over
n_samples = 100            # number of samples to average over

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Define functions ####################

def run_test_psd_varyd(
    dataset_loc,    # dataset location and file name (preprocessed .mat file)
    dataset_id,     # string to identify dataset
    d,              # approximation rank for RPCholesky
    n = n,
    l = l,
    method = method,
    n_iter_cd = n_iter_cd,
    rel_rnorm_tol = rel_rnorm_tol,
    save_data = save_data,
    out_loc = out_loc,
    n_samples = n_samples,
    sample_num = 1,
    rng = rng,
):
    print("========================================")
    if n_samples > 1:
        print(f"Loading dataset {dataset_id} (n = {n}, d = {d}, l = {l}), sample_num = {sample_num}...")
    else:
        print(f"Loading dataset {dataset_id} (n = {n}, d = {d}, l = {l})...")
    
    A, b = test_helper.load_data_psd(dataset_loc=dataset_loc, rng=rng)
    n = A.shape[0]

    print(f"Dataset {dataset_id} loaded.")

    print("========================================")
    print(f"Running RPCholesky: d = {d}...")

    ### Compute low-rank approximations
    # Use a different random approximation in each sample
    Ahat = rpcholesky(A, d, rng=None, b=int(np.ceil(d / 10)))  # default: b = "auto" (outcome is random)
    I = Ahat.get_indices()
    F = Ahat.get_left_factor()

    print("RPCholesky completed.")

    ### Run solvers
    print("========================================")
    print(f"Running solvers: l = {l}...")

    # SCRCD (diagonal sampling)
    A_scrcd, A_scrcd_plotdata = test_helper.run_scrcd(A, b, l=l, I=I, F=F, x_init=None, uniform=False, n_iter=n_iter_cd, method=method, rel_rnorm_tol=rel_rnorm_tol)

    # SCRCD (uniform sampling)
    A_scrcd2, A_scrcd2_plotdata = test_helper.run_scrcd(A, b, l=l, I=I, F=F, x_init=A_scrcd.x_init, uniform=True, n_iter=n_iter_cd, method=method, rel_rnorm_tol=rel_rnorm_tol)

    plotdata_list = [
        {"data": A_scrcd_plotdata,
        "label": "SCRCD (diagonal)",
        "id": "SCRCD"},
        {"data": A_scrcd2_plotdata,
        "label": "SCRCD (uniform)",
        "id": "SCRCD2"},
    ]

    ### Save data
    if save_data:
        print("========================================")
        print(f"Saving data to {out_loc + dataset_id}...")
        for plotdata in plotdata_list:
            if n_samples > 1:
                plotdata["data"].to_csv(f"{out_loc + dataset_id}_{plotdata['id']}_l{l}_varyd{d}_{sample_num}.csv", index=False)
            else:
                plotdata["data"].to_csv(f"{out_loc + dataset_id}_{plotdata['id']}_l{l}_varyd{d}.csv", index=False)
        print("Data saved.")

    print("========================================")
    print(f"Dataset {dataset_id} completed (n = {n}, d = {d}, l = {l}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Positional (required) arguments
    parser.add_argument("job_idx", type=int, help="Index of current job (sample number)")
    parser.add_argument("d_list_idx", type=int, help="Index for approximation rank in d_list")

    # Optional arguments
    parser.add_argument("--selected_dataset", type=str, default=selected_dataset, help="Selected dataset")
    parser.add_argument("--n", type=int, default=n, help="Number of datapoints to sample")
    parser.add_argument("--l", type=int, default=l, help="Block size for coordinate descent")
    parser.add_argument("--method", type=str, default=method, help="Method for solving inner projection for block CD")
    parser.add_argument("--n_iter_cd", type=int, default=n_iter_cd, help="Number of CD iterations")
    parser.add_argument("--rel_rnorm_tol", type=float, default=rel_rnorm_tol, help="Terminate if relative residual norm (wrt initial iterate) falls below threshold")
    parser.add_argument("--n_samples", type=int, default=n_samples, help="Number of samples to average over")

    parser.add_argument("--d_list", nargs="+", type=int, default=d_list, help="List of approximation ranks (space-separated)")

    args = parser.parse_args()
    
    d = args.d_list[args.d_list_idx]
    dataset_loc = data_loc + test_helper.datasets_psd[args.selected_dataset]

    run_test_psd_varyd(
        dataset_loc = dataset_loc,
        dataset_id = args.selected_dataset,
        n = args.n,
        d = d,
        l = args.l,
        method = args.method,
        rel_rnorm_tol = args.rel_rnorm_tol,
        n_iter_cd = args.n_iter_cd,
        n_samples = args.n_samples,
        sample_num = args.job_idx
    )
