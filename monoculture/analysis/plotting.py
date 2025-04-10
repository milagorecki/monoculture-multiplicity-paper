from __future__ import annotations
import matplotlib.pyplot as plt

# import matplotlib as mpl
import torch
import scipy as sp
import pandas as pd
from typing import Tuple
from matplotlib import axes
from .utils import prettify_model_name, key_to_model
from pathlib import Path
import matplotlib.colors as pltcolors


def get_observed_rejections(
    predictions: pd.DataFrame,
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
    padding=True,
):
    N, M = predictions.shape

    assert not (restrict_only_pos_instances and restrict_only_neg_instances)
    if restrict_only_pos_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 1]
    elif restrict_only_neg_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 0]
    sum_rejected = M - predictions.sum(axis=1)
    num_models_rejecting, counts = torch.Tensor(sum_rejected.values).unique(
        return_counts=True
    )

    if padding:
        counts_padded = torch.zeros(M + 1)
        for i in range(len(num_models_rejecting)):
            counts_padded[int(num_models_rejecting[i])] = counts[i]
        return num_models_rejecting, counts_padded

    return num_models_rejecting, counts


def get_observed_acceptance(
    predictions: pd.DataFrame,
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
    padding=True,
):
    N, M = predictions.shape

    assert not (restrict_only_pos_instances and restrict_only_neg_instances)
    if restrict_only_pos_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 1]
    elif restrict_only_neg_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 0]
    sum_accepted = predictions.sum(axis=1)
    num_models_accepting, counts = torch.Tensor(sum_accepted.values).unique(
        return_counts=True
    )

    if padding:
        counts_padded = torch.zeros(M + 1)
        for i in range(len(num_models_accepting)):
            counts_padded[int(num_models_accepting[i])] = counts[i]
        return torch.arange(M + 1), counts_padded

    return num_models_accepting, counts


def get_observed_k_rejections(
    k: int,
    predictions: pd.DataFrame,
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
):
    N, M = predictions.shape

    assert not (restrict_only_pos_instances and restrict_only_neg_instances)
    if restrict_only_pos_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 1]
    elif restrict_only_neg_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 1]
    sum_rejected = M - predictions.sum(axis=1)
    return sum_rejected[sum_rejected == k].shape[0]


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
):
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
    for idx, (model_key, eval_dict) in enumerate(evals[task].items()):
        keep_model = model_key in models_above_baseline[task]
        ax.scatter(
            idx,
            eval_dict["accuracy"],
            zorder=1,
            s=20,
            color="black" if keep_model else "grey",
        )
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
    for idx, (model_key, eval_dict) in enumerate(evals[task].items()):
        keep_model = model_key in models_above_baseline[task]
        ax.bar(
            idx,
            eval_dict["num_pred_negatives"] / eval_dict["n_samples"],
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
):
    """
    Plots the agreement between model predictions and observed data, optionally
    comparing with a baseline.

    Parameters:
        ax: The matplotlib axis to plot on.
        predictions: A tensor of predictions from the model.
        y_true: True labels (optional).
        baseline_rates: Baseline rejection rates for comparison (optional).
        plot_cumulative: Whether to plot cumulative probabilities.
        restrict_only_pos_instances: Restrict calculation to positive instances.
        ylabel: Label for the y-axis.
        baseline_label: Label for the baseline comparison.
        relative_x: Normalize x-axis to [0, 1].

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
    x = torch.arange(M + 1, dtype=torch.float32)
    if relative_x:
        x /= M
        ax.set_xlim(-0.01, 1.02 + width)

    # Compute observed agreement
    num_models_rejecting, frequencies = get_observed_rejections(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        true_labels=y_true,
    )
    N = frequencies.sum()

    # pad counts for consistent slicing
    frequencies_padded = torch.zeros(M + 1)
    for i in range(len(num_models_rejecting)):
        frequencies_padded[num_models_rejecting.int()[i]] = frequencies[i]

    # plot observed
    ax.bar(x, frequencies_padded / N, width=width, color="C0", label="observed")

    if baseline_rates:
        # compute expected agreement
        prob_expected = torch.tensor(
            [
                sp.stats.poisson_binom.pmf(k=num_rejections, p=baseline_rates)
                for num_rejections in range(M + 1)
            ]
        )
        # plot baseline
        ax.bar(x + width, prob_expected, width=width, color="C1", label=baseline_label)

    if plot_cumulative:
        # accumulated = lambda tens: [tens[i:].sum() for i in range(len(tens))]
        def cumulative_sum(t):
            return torch.flip(torch.cumsum(torch.flip(t, dims=[0]), dim=0), dims=[0])

        ax.plot(x, cumulative_sum(frequencies_padded / N), color="C0")
        if baseline_rates:
            ax.plot(x, cumulative_sum(prob_expected), color="C1")

    return ax
