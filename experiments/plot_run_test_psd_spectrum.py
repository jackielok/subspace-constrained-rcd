#!/usr/bin/env python3
### Plots output of run_test_psd_spectrum.py

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import run_test_psd_spectrum
import argparse

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Inputs ####################

figs_loc = "./figures/"

solver_id_list = ["A", "Aresidual"]
color_list = ["tab:green", "tab:blue"]
label_list = [r"Input $A$", r"Residual $A^{\bigcirc}$"]
linestyle_list = [(0, (5, 1)), "solid"]
linewidth_list = [1.75, 1.75]

#################### Define functions ####################

def plot_run_test_psd_spectrum(
    dataset_id,
    out_loc = run_test_psd_spectrum.out_loc,
    solver_id_list = solver_id_list,
    color_list = color_list,
    label_list = label_list,
    linestyle_list = linestyle_list,
    linewidth_list = linewidth_list,
    figs_loc = figs_loc,
):
    ### Save plots
    print("========================================")
    print(f"Saving figures to {figs_loc + dataset_id}...")

    # Load data
    plotdata_loc = f"{out_loc + dataset_id}"
    plotdata = pd.read_csv(f"{plotdata_loc}_eigvals.csv")
    
    fig, ax = plt.subplots(figsize=(6,4))
    for idx, solver_id in enumerate(solver_id_list):
        eigvals = plotdata[solver_id] 
        eigvals = eigvals[eigvals > 0]  # plot non-zero eigenvalues only
        dcondnum = np.sum(eigvals) / eigvals.iloc[-1]  # Demmel condition number
        
        ax.plot(np.arange(1, len(eigvals) + 1), eigvals,
                color=color_list[idx],
                label=f"{label_list[idx]} ({dcondnum:.2e})",
                linestyle=linestyle_list[idx],
                linewidth=linewidth_list[idx],
                zorder=len(solver_id_list) - idx);

    plt.xlabel("Eigenvalue index");
    plt.ylabel("Eigenvalue");
    plt.yscale("log");
    plt.tight_layout();
    plt.legend(frameon=False);

    if figs_loc is not None:
        plt.savefig(f"{figs_loc + dataset_id}_eigvals.pdf", dpi=300);

    print("Figures saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    

    # Optional arguments
    parser.add_argument("--selected_dataset", type=str, default=run_test_psd_spectrum.selected_datasets[0], help="Selected dataset")
    
    args = parser.parse_args()
    
    plot_run_test_psd_spectrum(
        dataset_id = args.selected_dataset
    )
