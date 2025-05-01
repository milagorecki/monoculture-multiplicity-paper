from __future__ import annotations
import matplotlib.pyplot as plt

# import matplotlib as mpl
import torch
import scipy as sp
import pandas as pd
import numpy as np
from typing import Tuple, Callable
from matplotlib import axes
from .utils import prettify_model_name, key_to_model, get_size_and_it
from .metrics import get_observed_acceptance_df, get_observed_rejections_df
from pathlib import Path
import matplotlib.colors as pltcolors
import logging

# plotting:
# plot functions take ax and plot given thing on this ax


def plot_multiple_tasks_horizontal(
    tasks: list,
    plot_fun: Callable,
    title: str = "",
    figsize=(18, 5),
    xlabel="",
    ylabel="",
    save_path=None,
    legend_idx: int = None,
    label_idx: Tuple = (None, None),
    **plot_fun_kwargs,
):
    fig, axs = plt.subplots(1, len(tasks), figsize=figsize, sharey=True)
    fig.suptitle(title)

    if len(tasks) == 1:
        axs = [axs]
    for i, task in enumerate(tasks):
        axs[i] = plot_fun(axs[i], **plot_fun_kwargs)

        axs[i].set_title(
            f"{task.replace('ACS', 'ACS ').replace('_', ' ')}"
        )  # (M = {M})")

    # set legend and label
    axs[label_idx[1] if label_idx[1] else 0].set_ylabel(ylabel)
    axs[label_idx[0] if label_idx[0] else len(tasks) // 2].set_xlabel(xlabel)
    axs[legend_idx if legend_idx else 0].legend()

    if save_path:
        plt.savefig(save_path)
    return fig, axs


# -------------------------------------------------------------
# GENERAL PERFORMANCE
# -------------------------------------------------------------


def plot_accuracies(
    ax: axes,
    task: str,
    evals: dict,
    baseline_evals: dict,
    models_above_baseline: dict,
    ylims: Tuple = (0, 0.85),
    label_rotation: int = 50,
    ha: str = "right",
    color: str = "grey",
    highlight_color: str = "black",
    hline_xmin=0,
    hline_xmax=1,  #
    marker="o",
    scatter_label=None,
):
    if baseline_evals:
        # plot hline for baselines
        ax.hlines(
            baseline_evals[task]["Constant"]["accuracy"],
            xmin=hline_xmin,
            xmax=hline_xmax,
            label="Constant",
            colors="dodgerblue",
            linestyle="-",
            zorder=0,
        )
        ax.hlines(
            baseline_evals[task]["XGBoost"]["accuracy"],
            xmin=hline_xmin,
            xmax=hline_xmax,
            label="XGBoost",
            colors="yellowgreen",
            linestyle="-",
            zorder=0,
        )

    # plot model accuracies
    x_labels = []
    scatter_labels = set()
    for idx, (model_key, val) in enumerate(
        sorted(
            evals[task]["accuracy"].items(), key=lambda item: get_size_and_it(item[0])
        )
    ):
        keep_model = model_key in models_above_baseline[task]
        ax.scatter(
            idx,
            val,
            zorder=1,
            s=20,
            color="black" if keep_model else "grey",
            marker=marker,
            label=scatter_label if scatter_label not in scatter_labels else None,
        )
        scatter_labels.add(scatter_label)
        x_labels.append(model_key)

    # set xticks
    ax.set_xticks(
        torch.arange(len(x_labels)),
        [prettify_model_name(key_to_model(key)) for key in x_labels],
        rotation=label_rotation,
        ha=ha,
        color=color,
    )
    for m in x_labels:
        if m in models_above_baseline[task]:
            idx = x_labels.index(m)
            plt.setp(ax.get_xticklabels()[idx], color=highlight_color)
    ax.set_ylim(ylims)
    ax.set_xlim(right=len(x_labels))

    return ax


def plot_label_dist(
    ax: axes, labels: pd.Series, ylims: Tuple = (0, 0.75), bar_width=0.2
):
    counts = labels.value_counts()
    ax.bar(
        counts.index * 2 * bar_width, counts.values / labels.shape[0], width=bar_width
    )
    ax.set_xticks(ticks=counts.index * 2 * bar_width, labels=counts.index)
    ax.set_xlabel("$y$")
    ax.set_ylim(ylims)
    return ax


def plot_neg_predicted(
    ax: axes,
    task: str,
    evals: dict,
    models_above_baseline: dict,
    ylims=(0, 1),
    label_rotation: int = 50,
    ha: str = "right",
    color: str = "grey",
    highlight_color: str = "black",
    omit_non_highlight_labels=False,
    bar_width=0.5,
):
    x_labels = []
    n_samples = evals[task].get("n_samples")
    num_pred_negatives = evals[task].get("num_pred_negatives")
    assert set(n_samples.keys()) == set(num_pred_negatives.keys())
    for idx, model_key in enumerate(num_pred_negatives.keys()):
        keep_model = model_key in models_above_baseline[task]
        ax.bar(
            idx,
            num_pred_negatives[model_key] / n_samples[model_key],
            # width=bar_width,
            zorder=1,
            color="black" if keep_model else "grey",
        )
        x_labels.append(model_key)

    # set xticks
    if not omit_non_highlight_labels:
        ax.set_xticks(
            torch.arange(len(x_labels)),
            [prettify_model_name(key_to_model(key)) for key in x_labels],
            rotation=label_rotation,
            ha=ha,
            color=color,
        )
        for m in x_labels:
            if m in models_above_baseline[task]:
                idx = x_labels.index(m)
                plt.setp(ax.get_xticklabels()[idx], color=highlight_color)
    else:
        ax.set_xticks(
            torch.arange(len(x_labels)),
            [
                (
                    prettify_model_name(key_to_model(key))
                    if key in models_above_baseline[task]
                    else ""
                )
                for key in x_labels
            ],
            rotation=label_rotation,
            ha=ha,
            color=highlight_color,
        )
    ax.set_ylim(ylims)
    ax.set_xlim(right=len(x_labels))

    return ax


# -------------------------------------------------------------
# AGREEMENT/RECOURSE
# -------------------------------------------------------------


def minimal_example_binomial(color_by_recourse=False, save_path=None):
    num = 5
    cut_off = (num + 1) // 2
    if color_by_recourse:
        colors = (
            ["forestgreen"]
            + (cut_off - 1) * ["yellowgreen"]
            + (cut_off - 1) * ["orange"]
            + ["firebrick"]
        )
    else:
        colors = "C0"
    probs = np.array(
        [sp.special.binom(num, i) * 2 ** (-num) for i in range(0, num + 1)]
    )
    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(num + 1), probs, color=colors)
    ax.bar_label(bars, fmt="%.3f", label_type="edge", size=8, padding=2)
    ax.set_xticks(np.arange(num + 1))

    if save_path:
        plt.savefig(save_path)
    plt.show()
    return probs[0], probs[:cut_off].sum(), probs[cut_off:-1].sum(), probs[-1]


def plot_agreement_matrix(
    models,
    agreements: torch.Tensor,
    title: str = "",
    file_name: str | Path = None,
    figsize=(8, 8),
    vmin=0,
    vmax=1,
    cmap="magma_r",
    norm=None,
    exclude_diagonal=False,
    annotate: bool = False,
):
    if exclude_diagonal:
        mask = torch.zeros_like(agreements).fill_diagonal_(torch.nan)
        agreements += mask
    if norm is None:
        norm = pltcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title(title)
    num_models = agreements.shape[0]
    assert len(models) == num_models
    im = ax.imshow(
        agreements.numpy(),
        aspect="equal",
        extent=(0, num_models, num_models, 0),  # left, right, bottom, top
        origin="upper",
        cmap=cmap,
        norm=norm,
    )
    # im = ax.pcolormesh(agreements.numpy(), cmap='magma_r', vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(
        torch.arange(num_models) + 0.5,
        [prettify_model_name(key_to_model(m)) for m in models],
        rotation=90,
        ha="right",  # horizontal alignment (right, center, left)
        # rotation_mode="anchor",
    )
    ax.set_yticks(
        torch.arange(num_models) + 0.5,
        [prettify_model_name(key_to_model(m)) for m in models],
    )

    if annotate:
        for y_index, y in enumerate(torch.arange(num_models)):
            for x_index, x in enumerate(torch.arange(num_models)):
                if y_index != x_index or not exclude_diagonal:
                    if not (
                        agreements[y_index, x_index].item()
                        != agreements[y_index, x_index].item()
                    ):  # not nan
                        label = f"{agreements[y_index, x_index].item():.2f}"
                        text_x = x + 0.5
                        text_y = y + 0.5
                        ax.text(
                            text_x,
                            text_y,
                            label,
                            color="black",
                            ha="center",
                            va="center",
                        )

    plt.colorbar(im, fraction=0.046, pad=0.04)
    if file_name:
        plt.savefig(file_name)
    plt.show()
    # return fig, ax


def plot_model_agreement_barplot(
    ax: plt.Axes,
    predictions: torch.Tensor,
    baseline_rates: torch.Tensor = None,
    y_true: torch.Tensor = None,
    restrict_only_pos_instances: bool = True,
    plot_cumulative: bool = False,
    xlabel: str = "fraction of models rejecting",
    ylabel: str = "fraction of positive instances",
    title: str = "",
    baseline_label: str = "random error",
    relative_x: bool = True,
    count_accepted=True,
    indicate_mean=False,
):
    """
    Plots the agreement between model predictions and observed data, optionally
    comparing with a baseline.

    Parameters:
        ax: The matplotlib axis to plot on.
        predictions: A tensor of predictions from the model.
        y_true: True labels (optional).
        baseline_rates: Baseline rejection rates for comparison (optional).
        plot_cumulative: Whether to plot cumulative probabilities (default = False, else True
          to plot from left to right, else 'left' to add up all the way to the left)
        restrict_only_pos_instances: Restrict calculation to positive instances.
        ylabel: Label for the y-axis.
        baseline_label: Label for the baseline comparison.
        relative_x: Normalize x-axis to [0, 1].
        count_accepted: Count positive predictions (True) or rejections (False), default: True,
        indicate_mean: Whether to plot a small triangle below the x-axis to indicate the mean value, default=False.

    Returns:
        ax: The matplotlib axis with the plot.
    """
    # Validate inputs
    _, M = predictions.shape
    if M == 0:
        print(ValueError("Predictions tensor is empty."))
        return ax

    # Configure the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if plot_cumulative:
        ax.set_ylim(0.0, 1.02)
    # Determine x-axis scaling
    width = 0.45 / M if relative_x else 0.5
    xs = torch.arange(M + 1, dtype=torch.float32)
    if relative_x:
        xs /= M
        ax.set_xlim(-0.01, 1.02 + width)

    # Compute observed agreement
    logging.warning(
        f"Counting {'acceptances, make sure to use accuracy as baseline rate.' if count_accepted else 'rejections, make sure to use error rate as baseline rate.'}"
    )
    fun = get_observed_acceptance_df if count_accepted else get_observed_rejections_df
    num_models_agreeing, frequencies = fun(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        true_labels=y_true,
        padding=True,
    )
    N = frequencies.sum()

    # plot observed
    ax.bar(xs, frequencies / N, width=width, color="C0", label="observed")

    if baseline_rates:
        # compute expected agreement
        prob_expected = torch.tensor(
            [
                sp.stats.poisson_binom.pmf(k=num_agreeing, p=baseline_rates)
                for num_agreeing in range(M + 1)
            ]
        )
        # plot baseline
        ax.bar(xs + width, prob_expected, width=width, color="C1", label=baseline_label)

    if plot_cumulative:

        def cumulative_sum(t):
            if plot_cumulative == "left":
                # from right to left (total at xmin)
                return torch.flip(
                    torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0]
                )
            else:
                # from left to right (total at xmax)
                return torch.cumsum(t, dim=0)

        ax.plot(xs, cumulative_sum(frequencies / N), color="C0")
        if baseline_rates:
            ax.plot(xs, cumulative_sum(prob_expected), color="C1")

    if indicate_mean:
        mean_obs = (num_models_agreeing * frequencies).sum() / (frequencies.sum() * M)
        mean_exp = (xs * prob_expected).sum() / (prob_expected.sum())
        print(mean_obs, mean_exp)
        # ax.text(0.4, 0.1, 'aha')
        ax.text(
            mean_obs,
            ax.get_ylim()[0],
            r"$\blacktriangle$",
            color="C0",
            fontsize=12,
            ha="center",
            va="top",
            alpha=0.5,
        )
        ax.text(
            mean_exp,
            ax.get_ylim()[0],
            r"$\blacktriangle$",
            color="C1",
            fontsize=12,
            ha="center",
            va="top",
            alpha=0.5,
        )  # '▲'

    ax.set_xticks(torch.arange(0, 1.01, 0.2))

    return ax


def plot_model_agreement_multiple_tasks(
    tasks: list,
    predictions: dict,
    baseline_rates: dict = None,
    data: dict = None,
    restrict_only_pos_instances: bool = True,
    plot_cumulative: bool = False,
    baseline_label: str = "random error",
    relative_x: bool = True,
    xlabel: str = "fraction of models accepting",
    ylabel="fraction of positive instances",
    title: str = "",
    legend_pos: int = 0,
    figsize=(18, 5),
    save_path=None,
    count_accepted=True,
    indicate_mean=False,
):

    fig, axs = plt.subplots(1, len(tasks), figsize=figsize, sharey=True)
    fig.suptitle(title)

    if len(tasks) == 1:
        axs = [axs]
    for i, task in enumerate(tasks):
        _, M = predictions[task].shape
        y_true = data[task][1]

        axs[i] = plot_model_agreement_barplot(
            axs[i],
            predictions=predictions[task],
            y_true=y_true,
            baseline_rates=(
                list(baseline_rates[task].values) if baseline_rates else baseline_rates
            ),
            plot_cumulative=plot_cumulative,
            restrict_only_pos_instances=restrict_only_pos_instances,
            indicate_mean=indicate_mean,
            ylabel="",  # omit label if setting it later
            xlabel="",
            baseline_label=baseline_label,
            relative_x=relative_x,
            count_accepted=count_accepted,
        )
        axs[i].set_title(
            f"{task.replace('ACS', 'ACS ').replace('_', ' ')}"
        )  # (M = {M})")

    axs[0].set_ylabel(ylabel)
    axs[len(tasks) // 2].set_xlabel(xlabel)
    axs[legend_pos].legend()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_agreement_lineplot(
    ax,
    agreements_observed: torch.Tensor | np.ndarray,
    agreements_exp: torch.Tensor | np.ndarray,
    sort_by=None,
    title="",
    plot_scatter=False,
    ylim=(0.0, 1.0),
):
    if isinstance(agreements_observed, torch.Tensor):
        agreements_observed = agreements_observed.numpy()
    if isinstance(agreements_exp, torch.Tensor):
        agreements_exp = agreements_exp.numpy()
    if sort_by:
        assert sort_by in [
            "observed",
            "expected",
        ], "If provided, 'sort_by' has to be one of ['observed', 'expected']."
        if sort_by == "obs":
            sort_indices = np.argsort(agreements_observed)
        elif sort_by == "exp":
            sort_indices = np.argsort(agreements_exp)
        agreements_observed = agreements_observed[sort_indices]
        agreements_exp = agreements_exp[sort_indices]
    else:
        agreements_observed = sorted(agreements_observed)
        agreements_exp = sorted(agreements_exp)

    fraction_model_pairs = np.arange(len(agreements_observed)) / len(
        agreements_observed
    )
    ax.plot(
        fraction_model_pairs,
        np.ones_like(agreements_observed),
        label="monoculture",
        color="red",
    )
    scatter_args = {"marker": "o", "markersize": 3} if plot_scatter else {}
    ax.plot(fraction_model_pairs, agreements_observed, label="observed", **scatter_args)
    ax.plot(fraction_model_pairs, agreements_exp, label="random error", **scatter_args)

    ax.plot(
        fraction_model_pairs,
        np.zeros_like(agreements_observed) + 0.5,
        label="coin flip",  # "chance level",
        color="grey",
        linestyle="dashed",
        linewidth=1,
        zorder=-1,
    )
    ax.fill_between(
        fraction_model_pairs,
        agreements_observed,
        np.ones_like(agreements_observed),
        color="lightgrey",
        alpha=0.4,
    )
    ax.set_ylim(bottom=ylim[0], top=ylim[1] + 0.01)
    ax.set_title(title.replace("ACS", "ACS ").replace("_", " "))
    return ax
