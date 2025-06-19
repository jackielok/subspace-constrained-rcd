#!/usr/bin/env python3
### Plots averaged outputs of run_test_psd.py

import numpy as np
from matplotlib import pyplot as plt
import plot_helper
import run_test_psd
import argparse

# rng = np.random.default_rng()
rng = np.random.default_rng(seed=123)

#################### Inputs ####################

figs_loc = "./figures/"

create_summary_data = True  # flag for whether summary dataset is created from individual sample outputs or not
plot_rel_rnorm_tol = -1     # trims summary dataset for relative norms (wrt initial iterate) smaller than this threshold
plot_rel_rnorm_b_tol = -1   # trims summary dataset for relative norms (wrt b) smaller than this threshold
plot_max_epoch = None       # plots up to plot_max_epoch number of epochs only if specified

# Plot SCRCD with diagonal sampling only
solver_id_list = ["CG", "RCD", "CDPP", "SCRCD"]
color_list = ["tab:brown", "tab:green", "tab:olive", "tab:blue"]
label_list = ["CG", "RCD", "CD++", r"SCRCD ($d=500$)"]
linestyle_list = ["solid", "dashed", (5, (10, 3)), "solid"]
linewidth_list = [1.5, 1.5, 1.5, 2.25]

# solver_id_list = ["CG", "RCD", "CDPP", "SCRCD", "SCRCD2"]
# color_list = ["tab:brown", "tab:green", "tab:olive", "tab:blue", "blue"]
# label_list = ["CG", "RCD", "CD++", "SCRCD (diag)", "SCRCD (unif)"]
# linestyle_list = ["solid", "dashed", (5, (10, 3)), "solid", (0, (3, 1, 1, 1))]
# linewidth_list = [1.5, 1.5, 1.5, 2.25, 2.25]

#################### Define functions ####################

def plot_run_test_psd(
    dataset_id = run_test_psd.selected_dataset,
    out_loc = run_test_psd.out_loc,
    solver_id_list = solver_id_list,
    color_list = color_list,
    label_list = label_list,
    linestyle_list = linestyle_list,
    linewidth_list = linewidth_list,
    figs_loc = figs_loc,
    n_samples = run_test_psd.n_samples,
    create_summary_data = create_summary_data,
    plot_rel_rnorm_tol = plot_rel_rnorm_tol,
    plot_rel_rnorm_b_tol = plot_rel_rnorm_b_tol,
    plot_max_epoch = plot_max_epoch,
    legend_loc = "best",
):
    ### Save plots
    print("========================================")
    print(f"Saving figures to {figs_loc + dataset_id}...")

    # Load all data and create summary statistics data if not yet created    
    if n_samples > 1 and create_summary_data:
        for solver_id in solver_id_list:
            plotdata_list = []
            for i in range(n_samples):
                plotdata_loc = f"{out_loc + dataset_id}_{solver_id}_{i}"
                plotdata = plot_helper.load_plotdata(plotdata_loc)
                plotdata_list.append(plotdata)
            
            plotdata_loc = f"{out_loc + dataset_id}_{solver_id}"
            plot_helper.average_plotdata(plotdata_list, plotdata_loc=plotdata_loc)

    # Load summary data
    plotdata_list = []
    for idx, solver_id in enumerate(solver_id_list):
        plotdata_loc = f"{out_loc + dataset_id}_{solver_id}"
        plotdata = plot_helper.load_plotdata(plotdata_loc, rel_rnorm_tol=plot_rel_rnorm_tol, rel_rnorm_b_tol=plot_rel_rnorm_b_tol, max_epoch=plot_max_epoch)
        plotdata_list.append({
            "data": plotdata,
            "color": color_list[idx],
            "label": label_list[idx],
            "linestyle": linestyle_list[idx],
            "linewidth": linewidth_list[idx],
            "id": solver_id
        })

    # Plot figures
    plot_helper.plot_list_rnorms_all(plotdata_list, plot_interval=(n_samples > 1), fig_loc=f"{figs_loc + dataset_id}", legend_loc=legend_loc)
    plot_helper.plot_list_rnorms(plotdata_list, plot_by="epoch", plot_interval=(n_samples > 1), fig_loc=f"{figs_loc + dataset_id}" + "_epoch", legend_loc=legend_loc)
    plot_helper.plot_list_rnorms(plotdata_list, plot_by="time", plot_interval=(n_samples > 1), fig_loc=f"{figs_loc + dataset_id}" + "_time", legend_loc=legend_loc)
    # plot_helper.plot_list_rnorms(plotdata_list, plot_by="eval", plot_interval=(n_samples > 1), fig_loc=f"{figs_loc + dataset_id}" + "_eval", legend_loc=legend_loc)
    plot_helper.plot_list_rnorms_relb(plotdata_list, plot_by="epoch", plot_interval=(n_samples > 1), fig_loc=f"{figs_loc + dataset_id}" + "_epoch_relb", legend_loc=legend_loc)
    plot_helper.plot_list_rnorms_relb(plotdata_list, plot_by="time", plot_interval=(n_samples > 1), fig_loc=f"{figs_loc + dataset_id}" + "_time_relb", legend_loc=legend_loc)
    
    print("Figures saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("--selected_dataset", type=str, default=run_test_psd.selected_dataset, help="Selected dataset")
    parser.add_argument("--plot_rel_rnorm_tol", type=float, default=plot_rel_rnorm_tol, help="Threshold for truncating dataset if relative residual norm (wrt initial iterat) falls below this threshold")
    parser.add_argument("--plot_rel_rnorm_b_tol", type=float, default=plot_rel_rnorm_b_tol, help="Threshold for truncating dataset if relative residual norm (wrt b) falls below this threshold")
    parser.add_argument("--plot_max_epoch", type=int, default=plot_max_epoch, help="Max number of epochs to plot up to")
    parser.add_argument("--legend_loc", type=str, default="best", help="Location for legend, given as a string, (x, y) coordinates, or None for no legend")
    parser.add_argument("--data_created", action="store_true", help="Whether figures can be produced using summary data that has already been created, or if not create the summary data")
    parser.add_argument("--n_samples", type=int, default=run_test_psd.n_samples, help="Number of samples to average over")

    args = parser.parse_args()

    if args.plot_max_epoch is not None:
        plot_max_epoch = int(args.plot_max_epoch) if args.plot_max_epoch != "None" else None

    legend_loc = args.legend_loc if args.legend_loc != "None" else None
    if legend_loc[0] == "(":
        # Convert input string "(x, y)" to tuple of coordinates (x, y)
        legend_loc = legend_loc.strip("()")  # remove the parentheses
        x, y = legend_loc.split(",")         # split the string by comma
        x, y = float(x), float(y)
        legend_loc = (x, y)
    
    plot_run_test_psd(
        dataset_id = args.selected_dataset,
        create_summary_data = not args.data_created,
        n_samples = args.n_samples,
        plot_rel_rnorm_tol = args.plot_rel_rnorm_tol,
        plot_rel_rnorm_b_tol = args.plot_rel_rnorm_b_tol,
        plot_max_epoch = plot_max_epoch,
        legend_loc = legend_loc
    )
