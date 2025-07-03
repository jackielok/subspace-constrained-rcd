#!/usr/bin/env python3
### For solving a positive definite linear system coming from kernel ridge regression
### Allows for parallelization over different block sizes and over a number of samples

import numpy as np
import math
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

# Selected dataset
selected_dataset = "codrna"

n = 15000                  # number of datapoints to sample
kernel = "gaussian"        # supported kernels: {"gaussian", "matern", "laplace"}
bandwidth = 3              # kernel bandwidth parameter
implicit_kernel = False    # evaluate entries implicitly (True) or store array in memory
lamb = 1e-7 * n            # regularization parameter
d = 1000                   # approximation rank for RPCholesky
method = "direct"          # method for solving inner projection for block CD in ["direct", "cg"]
rel_rnorm_tol = -1         # terminate if relative residual norm falls below threshold
save_data = True

l_list = [1, 20, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1875]    # list of block sizes to run over
n_epochs = 30     # calculate number of iterations to match the number of epochs
n_iter_cd_list = [math.ceil(n_epochs * (n / l)) for l in l_list]
n_samples = 100   # number of samples to average over

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Define functions ####################

def run_test_varyl(
    dataset_loc,    # dataset location and file name (preprocessed .mat file)
    dataset_id,     # string to identify dataset
    l,              # block size for coordinate descent
    n_iter_cd,      # number of iterations for cd
    n = n,
    kernel = kernel,
    bandwidth = bandwidth,
    implicit_kernel = implicit_kernel,
    lamb = lamb,
    d = d,
    method = method,
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

    print("RPCholesky completed.")

    ### Run solvers
    print("========================================")
    print(f"Running solvers: l = {l}...")

    # SCRCD (diagonal sampling)
    K_scrcd, K_scrcd_plotdata = test_helper.run_scrcd(K + D, ytrain, l=l, I=I, F=F, x_init=None, uniform=False, n_iter=n_iter_cd, method=method, rel_rnorm_tol=rel_rnorm_tol)
    
    # SCRCD (uniform sampling)
    K_scrcd2, K_scrcd2_plotdata = test_helper.run_scrcd(K + D, ytrain, l=l, I=I, F=F, x_init=K_scrcd.x_init, uniform=True, n_iter=n_iter_cd, method=method, rel_rnorm_tol=rel_rnorm_tol)
    
    # RCD (diagonal sampling)
    K_rcd, K_rcd_plotdata = test_helper.run_scrcd(K + D, ytrain, l=l, I=None, F=None, x_init=K_scrcd.x_init, uniform=False, n_iter=n_iter_cd, method=method, rel_rnorm_tol=rel_rnorm_tol)

    plotdata_list = [
        {"data": K_scrcd_plotdata,
        "label": "SCRCD (diagonal)",
        "id": "SCRCD"},
        {"data": K_scrcd2_plotdata,
        "label": "SCRCD (uniform)",
        "id": "SCRCD2"},
        {"data": K_rcd_plotdata,
        "label": "RCD",
        "id": "RCD"}
    ]

    ### Save data
    if save_data:
        print("========================================")
        print(f"Saving data to {out_loc + dataset_id}...")
        for plotdata in plotdata_list:
            if n_samples > 1:
                plotdata["data"].to_csv(f"{out_loc + dataset_id}_{plotdata['id']}_d{d}_varyl{l}_{sample_num}.csv", index=False)
            else:
                plotdata["data"].to_csv(f"{out_loc + dataset_id}_{plotdata['id']}_d{d}_varyl{l}.csv", index=False)
        print("Data saved.")

    print("========================================")
    print(f"Dataset {dataset_id} completed (n = {n}, d = {d}, l = {l}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    

    # Positional (required) arguments
    parser.add_argument("job_idx", type=int, help="Index of current job (sample number)")
    parser.add_argument("l_list_idx", type=int, help="Index for block size in l_list")

    # Optional arguments
    parser.add_argument("--selected_dataset", type=str, default=selected_dataset, help="Selected dataset")
    parser.add_argument("--n", type=int, default=n, help="Number of datapoints to sample")
    parser.add_argument("--d", type=int, default=d, help="Approximation rank for RPCholesky")
    parser.add_argument("--kernel", type=str, default=kernel, help="Kernel function used")
    parser.add_argument("--bandwidth", type=float, default=bandwidth, help="Kernel bandwidth parameter")
    parser.add_argument("--implicit_kernel", type=lambda x: x.lower() == "true", default=implicit_kernel, help="evaluate entries implicitly (True) or store array in memory (False)")
    parser.add_argument("--lamb", type=float, default=lamb, help="Regularization parameter")
    parser.add_argument("--method", type=str, default=method, help="Method for solving inner projection for block CD")
    parser.add_argument("--rel_rnorm_tol", type=float, default=rel_rnorm_tol, help="Terminate if relative residual norm (wrt initial iterate) falls below threshold")
    parser.add_argument("--n_samples", type=int, default=n_samples, help="Number of samples to average over")

    parser.add_argument("--l_list", nargs="+", type=int, default=l_list, help="List of block sizes for coordinate descent (space-separated)")
    parser.add_argument("--n_epochs", type=float, default=n_epochs, help="Number of epochs to perform")

    args = parser.parse_args()

    n_iter_cd_list = [math.ceil(args.n_epochs * (args.n / l)) for l in args.l_list]
    
    l = args.l_list[args.l_list_idx]
    n_iter_cd = n_iter_cd_list[args.l_list_idx]
    dataset_loc = data_loc + test_helper.datasets[args.selected_dataset]
    
    run_test_varyl(
        dataset_loc = dataset_loc,
        dataset_id = args.selected_dataset,
        n = args.n,
        d = args.d,
        l = l,
        kernel = args.kernel,
        bandwidth = args.bandwidth,
        implicit_kernel = args.implicit_kernel,
        lamb = args.lamb,
        method = args.method,
        rel_rnorm_tol = args.rel_rnorm_tol,
        n_iter_cd = n_iter_cd,
        n_samples = args.n_samples,
        sample_num = args.job_idx
    )
