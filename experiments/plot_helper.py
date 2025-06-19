#!/usr/bin/env python3

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#################### Inputs ####################

import matplotlib.pylab as pylab
params = {
    "legend.fontsize": "large",
    "figure.figsize": (6, 4),
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
pylab.rcParams.update(params)

#################### Define functions ####################

def output_plotdata(solver, plotdata_loc=None):
    ### Create DataFrame with solver output
    data = pd.DataFrame({
        "iterations": solver.rnorms_iters,
        "epochs": np.array(solver.rnorms_iters) * (solver.l / solver.n),
        "residual_norms": solver.rnorms,
        "rel_residual_norms_b": solver.rnorms / np.linalg.norm(solver.b),
        "cum_update_times": np.cumsum(solver.update_times),
        # "cum_num_queries": np.cumsum(solver.num_queries),
    })

    if plotdata_loc is not None:
        data.to_csv(f"{plotdata_loc}.csv", index=False)
    
    return data

def average_plotdata(plotdata_list, plotdata_loc=None):
    ### Computes summary statistics given a list of output DataFrames
    data = pd.DataFrame({
        "iterations": plotdata_list[0]["iterations"].values,
        "epochs": plotdata_list[0]["epochs"].values,
        "residual_norms": np.median([plotdata["residual_norms"].values for plotdata in plotdata_list], axis=0),
        "residual_norms_mean": np.mean([plotdata["residual_norms"].values for plotdata in plotdata_list], axis=0),
        "residual_norms_q0.2": np.quantile([plotdata["residual_norms"].values for plotdata in plotdata_list], axis=0, q=0.2),
        "residual_norms_q0.8": np.quantile([plotdata["residual_norms"].values for plotdata in plotdata_list], axis=0, q=0.8),
        "rel_residual_norms_b": np.median([plotdata["rel_residual_norms_b"].values for plotdata in plotdata_list], axis=0),
        "rel_residual_norms_b_mean": np.mean([plotdata["rel_residual_norms_b"].values for plotdata in plotdata_list], axis=0),
        "rel_residual_norms_b_q0.2": np.quantile([plotdata["rel_residual_norms_b"].values for plotdata in plotdata_list], axis=0, q=0.2),
        "rel_residual_norms_b_q0.8": np.quantile([plotdata["rel_residual_norms_b"].values for plotdata in plotdata_list], axis=0, q=0.8),
        "cum_update_times": np.median([plotdata["cum_update_times"].values for plotdata in plotdata_list], axis=0),
        # "cum_num_queries": np.median([plotdata["cum_num_queries"].values for plotdata in plotdata_list], axis=0),
    })

    if plotdata_loc is not None:
        data.to_csv(f"{plotdata_loc}.csv", index=False)
    
    return data

def load_plotdata(plotdata_loc, rel_rnorm_tol=None, rel_rnorm_b_tol=None, max_epoch=None):
    ### Loads and processes output DataFrame
    data = pd.read_csv(f"{plotdata_loc}.csv")
    data["rel_residual_norms"] = data["residual_norms"] / data["residual_norms"][0]

    if "residual_norms_mean" in data.columns:
        data["rel_residual_norms_mean"] = data["residual_norms_mean"] / data["residual_norms"][0]
        data["rel_residual_norms_q0.2"] = data["residual_norms_q0.2"] / data["residual_norms"][0]
        data["rel_residual_norms_q0.8"] = data["residual_norms_q0.8"] / data["residual_norms"][0]

    # If rel_rnorm_tol is specified, removes rows after the first point the median relative residual norm
    # (with respect to initial iterate x0) is less than rel_rnorm_tol
    if rel_rnorm_tol is not None:
        if np.any(data["rel_residual_norms"].values < rel_rnorm_tol):
            idx = np.argmax(data["rel_residual_norms"].values < rel_rnorm_tol)
            data = data.iloc[0:idx+1]

    # If rel_rnorm_b_tol is specified, removes rows after the first point the median relative residual norm
    # (with respect to b) is less than rel_rnorm_tol
    if rel_rnorm_b_tol is not None:
        if np.any(data["rel_residual_norms_b"].values < rel_rnorm_b_tol):
            idx = np.argmax(data["rel_residual_norms_b"].values < rel_rnorm_b_tol)
            data = data.iloc[0:idx+1]

    # If max_epoch is specified, removes subsequent rows after max_epoch number of epochs
    if max_epoch is not None:
        if np.any(data["epochs"] >= max_epoch):
            idx = np.argmax(data["epochs"].values >= max_epoch)
            data = data.iloc[0:idx+1]

    return data

def plot_rnorms_iter(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to x0) against the number of iterations
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4));
        plot_rnorms_iter(plotdata, ax=ax, label=label, **kwargs)
        plt.xlabel("Iterations");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["iterations"], plotdata["rel_residual_norms"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["iterations"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["iterations"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_epoch(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to x0) against the number of epochs
    if ax is None:
        fig, ax = plt.subpots(figsize=(6,4));
        plot_rnorms_epoch(plotdata, ax=ax, label=label, **kwargs)
        plt.xlabel("Iterations");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["epochs"], plotdata["rel_residual_norms"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["epochs"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["epochs"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_time(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to x0) against the time elapsed
    if ax is None:
        fig, ax = plt.subpots(figsize=(6,4));
        plot_rnorms_time(plotdata, ax=ax, label=label, **kwargs);
        plt.xlabel("Time (seconds)");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["cum_update_times"], plotdata["rel_residual_norms"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["cum_update_times"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["cum_update_times"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_eval(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to x0) against the number of entry evaluations
    ### The count is only correct if the matrix A is a FunctionMatrix (e.g. representing a kernel matrix) not stored in memory
    if ax is None:
        fig, ax = plt.subpots(figsize=(6,4));
        plot_rnorms_eval(plotdata, ax=ax, label=label, **kwargs)
        plt.xlabel("Entry evaluations");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["cum_num_queries"], plotdata["rel_residual_norms"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["cum_num_queries"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["cum_num_queries"], plotdata["rel_residual_norms_q0.2"], plotdata["rel_residual_norms_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_relb_iter(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to b) against the number of iterations
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4));
        plot_rnorms_relb_iter(plotdata, ax=ax, label=label, **kwargs)
        plt.xlabel("Iterations");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["iterations"], plotdata["rel_residual_norms_b"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["iterations"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["iterations"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_relb_epoch(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to b) against the number of epochs
    if ax is None:
        fig, ax = plt.subpots(figsize=(6,4));
        plot_rnorms_relb_epoch(plotdata, ax=ax, label=label, **kwargs)
        plt.xlabel("Iterations");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["epochs"], plotdata["rel_residual_norms_b"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["epochs"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["epochs"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_relb_time(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to b) against the time elapsed
    if ax is None:
        fig, ax = plt.subpots(figsize=(6,4));
        plot_rnorms_relb_time(plotdata, ax=ax, label=label, **kwargs);
        plt.xlabel("Time (seconds)");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["cum_update_times"], plotdata["rel_residual_norms_b"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["cum_update_times"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["cum_update_times"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0);

def plot_rnorms_relb_eval(plotdata, ax=None, plot_interval=False, label=None, **kwargs):
    ### Plots relative residual norms (relative to b) against the number of entry evaluations
    ### The count is only correct if the matrix A is a FunctionMatrix (e.g. representing a kernel matrix) not stored in memory
    if ax is None:
        fig, ax = plt.subpots(figsize=(6,4));
        plot_rnorms_relb_eval(plotdata, ax=ax, label=label, **kwargs)
        plt.xlabel("Entry evaluations");
        plt.ylabel("Relative residual norm");
        plt.yscale("log");
        plt.tight_layout();
    else:
        ax.plot(plotdata["cum_num_queries"], plotdata["rel_residual_norms_b"], label=label, **kwargs);
        if plot_interval:
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(plotdata["cum_num_queries"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(plotdata["cum_num_queries"], plotdata["rel_residual_norms_b_q0.2"], plotdata["rel_residual_norms_b_q0.8"], alpha=0.25, linewidth=0);

def plot_list_rnorms(plotdata_list, plot_by="iter", ax=None, plot_interval=False, fig_loc=None, legend_loc="best", **kwargs):
    ### Plots a list of output DataFrames sequentially (selected plot, relative residual norms relative to x0)
    ### plot_by: {"iter", "epoch", "time", "eval"}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4));
        plot_list_rnorms(plotdata_list, plot_by, ax=ax, plot_interval=plot_interval, fig_loc=fig_loc, legend_loc=legend_loc, **kwargs)
        plt.tight_layout();
        if fig_loc is not None:
            plt.savefig(f"{fig_loc}.pdf", dpi=300);

    else:
        if plot_by == "iter":
            for plotdata in plotdata_list:
                plot_rnorms_iter(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Iterations");
        elif plot_by == "epoch":
            for plotdata in plotdata_list:
                plot_rnorms_epoch(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Epochs");
        elif plot_by == "time":
            for plotdata in plotdata_list:
                plot_rnorms_time(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Time (seconds)");
        elif plot_by == "eval":
            for plotdata in plotdata_list:
                plot_rnorms_eval(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Entry evaluations");
        
        ax.set_ylabel("Relative residual norm");
        ax.set_yscale("log");
        if legend_loc is not None:
            ax.legend(loc=legend_loc, frameon=False);

def plot_list_rnorms_relb(plotdata_list, plot_by="iter", ax=None, plot_interval=False, fig_loc=None, legend_loc="best", **kwargs):
    ### Plots a list of output DataFrames sequentially (selected plot, relative residual norms relative to b)
    ### plot_by: {"iter", "epoch", "time", "eval"}
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4));
        plot_list_rnorms_relb(plotdata_list, plot_by, ax=ax, plot_interval=plot_interval, fig_loc=fig_loc, legend_loc=legend_loc, **kwargs)
        plt.tight_layout();
        if fig_loc is not None:
            plt.savefig(f"{fig_loc}.pdf", dpi=300);

    else:
        if plot_by == "iter":
            for plotdata in plotdata_list:
                plot_rnorms_relb_iter(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Iterations");
        elif plot_by == "epoch":
            for plotdata in plotdata_list:
                plot_rnorms_relb_epoch(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Epochs");
        elif plot_by == "time":
            for plotdata in plotdata_list:
                plot_rnorms_relb_time(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Time (seconds)");
        elif plot_by == "eval":
            for plotdata in plotdata_list:
                plot_rnorms_relb_eval(plotdata["data"], plot_interval=plot_interval, label=plotdata["label"], color=plotdata["color"], linestyle=plotdata["linestyle"], linewidth=plotdata["linewidth"], ax=ax, **kwargs)
                ax.set_xlabel("Entry evaluations");
        
        ax.set_ylabel("Relative residual norm");
        ax.set_yscale("log");
        if legend_loc is not None:
            ax.legend(loc=legend_loc, frameon=False);

def plot_list_rnorms_all(plotdata_list, plot_interval=False, fig_loc=None, legend_loc="best", **kwargs):
    ### Plots a list of output DataFrames sequentially (all plots, relative residual norms relative to x0)
    fig, axes = plt.subplots(2, 2, figsize=(12,8));

    plot_list_rnorms(plotdata_list, plot_by="epoch", ax=axes[0,0], plot_interval=plot_interval, legend_loc=legend_loc, **kwargs)
    plot_list_rnorms(plotdata_list, plot_by="time", ax=axes[0,1], plot_interval=plot_interval, legend_loc=legend_loc, **kwargs)
    plot_list_rnorms_relb(plotdata_list, plot_by="epoch", ax=axes[1,0], plot_interval=plot_interval, legend_loc=legend_loc, **kwargs)
    plot_list_rnorms_relb(plotdata_list, plot_by="time", ax=axes[1,1], plot_interval=plot_interval, legend_loc=legend_loc, **kwargs)

    plt.tight_layout();

    if fig_loc is not None:
        plt.savefig(f"{fig_loc}.pdf", dpi=300);

def plot_rnorms_varyl(l_list, rel_rnorms_list, ax=None, plot_interval=False, label=None, fig_loc=None, legend_loc="best", **kwargs):
    ### Plots relative residual norms (with respect to x0) after a fixed number of epochs against the block size l
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4));
        plot_rnorms_varyl(l_list, rel_rnorms_list, ax=ax, plot_interval=plot_interval, label=label, fig_loc=fig_loc, legend_loc=legend_loc, **kwargs)
        plt.tight_layout();
        if fig_loc is not None:
            plt.savefig(f"{fig_loc}.pdf", dpi=300);
    else:
        if plot_interval:
            ax.plot(l_list, rel_rnorms_list[0], label=label, **kwargs);
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(l_list, rel_rnorms_list[1], rel_rnorms_list[2], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(l_list, rel_rnorms_list[1], rel_rnorms_list[2], alpha=0.25, linewidth=0);
        else:
            ax.plot(l_list, rel_rnorms_list, label=label, **kwargs);

        ax.set_xlabel(r"Block size $\ell$");
        ax.set_ylabel("Relative residual norm");
        ax.set_yscale("log");
        if legend_loc is not None:
            ax.legend(loc=legend_loc, frameon=False);

def plot_rnorms_varyd(d_list, rel_rnorms_list, ax=None, plot_interval=False, label=None, fig_loc=None, legend_loc="best", **kwargs):
    ### Plots relative residual norms (with respect to x0) after a fixed number of epochs against the approximation rank d
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4));
        plot_rnorms_varyd(d_list, rel_rnorms_list, ax=ax, plot_interval=plot_interval, label=label, fig_loc=fig_loc, legend_loc=legend_loc, **kwargs)
        plt.tight_layout();
        if fig_loc is not None:
            plt.savefig(f"{fig_loc}.pdf", dpi=300);
    else:
        if plot_interval:
            ax.plot(d_list, rel_rnorms_list[0], label=label, **kwargs);
            # Plot upper and lower intervals
            if "color" in kwargs:
                ax.fill_between(d_list, rel_rnorms_list[1], rel_rnorms_list[2], alpha=0.25, linewidth=0, color=kwargs["color"]);
            else:
                ax.fill_between(d_list, rel_rnorms_list[1], rel_rnorms_list[2], alpha=0.25, linewidth=0);
        else:
            ax.plot(d_list, rel_rnorms_list, label=label, **kwargs);

        ax.set_xlabel(r"Approximation rank $d$");
        ax.set_ylabel("Relative residual norm");
        ax.set_yscale("log");
        if legend_loc is not None:
            ax.legend(loc=legend_loc, frameon=False);
