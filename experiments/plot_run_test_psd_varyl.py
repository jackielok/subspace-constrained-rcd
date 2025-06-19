#!/usr/bin/env python3
### Plots averaged outputs of run_test_psd_varyl.py

import numpy as np
from matplotlib import pyplot as plt
import plot_helper
import run_test_psd_varyl
import argparse

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Inputs ####################

figs_loc = "./figures/"

create_summary_data = True  # flag for whether summary dataset is created from individual sample outputs or not

solver_id_list = ["RCD", "CDPP", "SCRCD"]
color_list = ["tab:green", "tab:olive", "tab:blue"]
label_list = ["RCD", "CD++", r"SCRCD ($d = 500$)"]
linestyle_list = ["dashed", (5, (10, 3)), "solid"]
linewidth_list = [1.75, 1.75, 1.75]

#################### Define functions ####################

def plot_run_test_psd_varyl(
    dataset_id = run_test_psd_varyl.selected_dataset,
    out_loc = run_test_psd_varyl.out_loc,
    l_list = run_test_psd_varyl.l_list,
    d = run_test_psd_varyl.d,
    solver_id_list = solver_id_list,
    color_list = color_list,
    label_list = label_list,
    linestyle_list = linestyle_list,
    linewidth_list = linewidth_list,
    figs_loc = figs_loc,
    n_samples = run_test_psd_varyl.n_samples,
    create_summary_data = create_summary_data,
    legend_loc = "best",
):  
    ### Save plots
    print("========================================")
    print(f"Saving figures to {figs_loc + dataset_id}...")

    fig, ax = plt.subplots(figsize=(6,4));

    # Load all data and create summary statistics data if not yet created
    if n_samples > 1 and create_summary_data:
        for solver_id in solver_id_list:
            for l in l_list:
                plotdata_list = []
                for i in range(n_samples):
                    plotdata_loc = f"{out_loc + dataset_id}_{solver_id}_d{d}_varyl{l}_{i}"
                    plotdata = plot_helper.load_plotdata(plotdata_loc)
                    plotdata_list.append(plotdata)
            
                plotdata_loc = f"{out_loc + dataset_id}_{solver_id}_d{d}_varyl{l}"
                plot_helper.average_plotdata(plotdata_list, plotdata_loc=plotdata_loc)

    # Load summary data
    for idx, solver_id in enumerate(solver_id_list):
        rel_rnorms_list = [] 
        rel_rnorms_lwr_list = []
        rel_rnorms_upr_list = []        
        for l in l_list:
            plotdata_loc = f"{out_loc + dataset_id}_{solver_id}_d{d}_varyl{l}"
            plotdata = plot_helper.load_plotdata(plotdata_loc)
            rel_rnorms_list.append(plotdata["rel_residual_norms"].values[-1])
            if n_samples > 1:
                rel_rnorms_lwr_list.append(plotdata["rel_residual_norms_q0.2"].values[-1])
                rel_rnorms_upr_list.append(plotdata["rel_residual_norms_q0.8"].values[-1])
    
        if n_samples > 1:
            plot_helper.plot_rnorms_varyl(l_list, [rel_rnorms_list, rel_rnorms_lwr_list, rel_rnorms_upr_list], ax=ax,
                                          color=color_list[idx],
                                          marker=".",
                                          plot_interval=True,
                                          label=label_list[idx],
                                          linestyle=linestyle_list[idx],
                                          linewidth=linewidth_list[idx],
                                          legend_loc=legend_loc)
        else:
            plot_helper.plot_rnorms_varyl(l_list, rel_rnorms_list, ax=ax,
                                          color=color_list[idx],
                                          marker=".",
                                          plot_interval=False,
                                          label=label_list[idx],
                                          linestyle=linestyle_list[idx],
                                          linewidth=linewidth_list[idx],
                                          legend_loc=legend_loc)

    ax.legend(frameon=False);
    plt.tight_layout();    
    plt.savefig(f"{figs_loc + dataset_id}_varyl.pdf", dpi=300);

    print("Figures saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--selected_dataset", type=str, default=run_test_psd_varyl.selected_dataset, help="Selected dataset")
    parser.add_argument("--l_list", type=int, nargs="+", default=run_test_psd_varyl.l_list, help="List of block sizes for coordinate descent (space-separated)")
    parser.add_argument("--d", type=int, default=run_test_psd_varyl.d, help="Approximation rank for RPCholesky")
    parser.add_argument("--n_samples", type=int, default=run_test_psd_varyl.n_samples, help="Number of samples to average over")
    parser.add_argument("--legend_loc", type=str, default="best", help="Location for legend, given as a string, (x, y) coordinates, or None for no legend")
    parser.add_argument("--data_created", action="store_true", help="Whether figures can be produced using summary data that has already been created, or if not create the summary data")

    args = parser.parse_args()

    legend_loc = args.legend_loc if args.legend_loc != "None" else None
    if legend_loc[0] == "(":
        # Convert input string "(x, y)" to tuple of coordinates (x, y)
        legend_loc = legend_loc.strip("()")  # remove the parentheses
        x, y = legend_loc.split(",")         # split the string by comma
        x, y = float(x), float(y)
        legend_loc = (x, y)

    plot_run_test_psd_varyl(
        dataset_id = args.selected_dataset,
        l_list = args.l_list,
        d = args.d,
        n_samples = args.n_samples,
        create_summary_data = not args.data_created,
        legend_loc = legend_loc
    )
