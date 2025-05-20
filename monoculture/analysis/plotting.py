from __future__ import annotations
import matplotlib.pyplot as plt

# import matplotlib as mpl
import torch
import scipy as sp
import pandas as pd
import numpy as np
from typing import Tuple, Callable
from matplotlib import axes
from .utils import prettify_model_name, key_to_model, get_size_and_it, cumulative_sum
from .metrics import (
    poisson_binom_agreement,
    get_obs_agreement_counts,
    get_obs_acceptance_aggregated,
    get_obs_rejections_aggregated,
    get_fraction_obs_acceptance_aggregated,
)
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
    num_pred_negatives = sorted(
        evals[task].get("num_pred_negatives").items(),
        key=lambda item: get_size_and_it(item[0]),
    )
    assert set(n_samples.keys()) == set(dict(num_pred_negatives).keys())

    for idx, (model_key, num_neg) in enumerate(num_pred_negatives):
        keep_model = model_key in models_above_baseline[task]
        ax.bar(
            idx,
            num_neg / n_samples[model_key],
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


def plot_recourse_barplot(
    ax: plt.Axes,
    predictions: torch.Tensor,
    baseline_rates: torch.Tensor = None,
    y_true: torch.Tensor = None,
    restrict_only_pos_instances: bool = True,
    restrict_only_neg_instances: bool = False,
    plot_cumulative: bool = False,
    xlabel: str = "fraction of models rejecting",
    ylabel: str = "fraction of positive instances",
    title: str = "",
    baseline_label: str = "random error",
    relative_x: bool = True,
    count_accepted=True,
    indicate_mean=False,
    show_monoc=True,
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
    fun = (
        get_obs_acceptance_aggregated
        if count_accepted
        else get_obs_rejections_aggregated
    )
    num_models_agreeing, frequencies = fun(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=y_true,
        padding=True,
    )
    N = frequencies.sum()

    # plot observed
    ax.bar(
        xs - (width / 2 if baseline_rates else 0),
        frequencies / N,
        width=width,
        color="C0",
        label="observed",
    )

    if show_monoc:
        # plot monoculture
        mean_rate = (
            1.0 - np.mean(baseline_rates) if count_accepted else np.mean(baseline_rates)
        )
        ax.step(
            x=xs,
            y=[mean_rate] * (len(xs) - 1) + [1.0],
            where="post",
            color="red",
            label="monoculture",
            linestyle="dotted",
            zorder=-1,
        )

    if baseline_rates:
        # compute expected agreement
        prob_expected = torch.tensor(
            [
                poisson_binom_agreement(baseline_rate=baseline_rates, k=num_agreeing)
                for num_agreeing in range(M + 1)
            ]
        )
        # plot baseline
        ax.bar(
            xs + width / 2, prob_expected, width=width, color="C1", label=baseline_label
        )

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

        ax.step(xs, cumulative_sum(frequencies / N), color="C0", where="post")
        if baseline_rates:
            ax.plot(xs, cumulative_sum(prob_expected), color="C1")

        print(
            xs.shape,
            cumulative_sum(prob_expected).shape,
            cumulative_sum(frequencies / N).shape,
        )

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


def recourse_barplot_categories(predictions):
    raise NotImplementedError(
        "Check 'same-prompt_individual.ipynb' for implementation."
    )


def no_leading_zero(x, pos):
    if abs(x) < 1:
        return f"{x:.2f}".lstrip("0").replace("-0", "-")
    else:
        return f"{x:.2f}"


def plot_recourse_lineplot(
    ax: plt.Axes,
    predictions: torch.Tensor,
    baseline_rates: torch.Tensor = None,
    y_true: torch.Tensor = None,
    restrict_only_pos_instances: bool = True,
    restrict_only_neg_instances: bool = False,
    xlabel: str = "fraction of positive instances",
    ylabel: str = "fraction of models accepting",
    title: str = "",
    baseline_label: str = "random error",
    relative_x: bool = True,
    count_accepted=True,
    at_least=True,
    show_monoc=True,
    plot_pdf=True,
    alpha=1.0,
    num_ticks=5,
):
    """
    x individuals are accepted by **at least** y models

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

    num_nan_models = predictions.isna().any().sum()
    if num_nan_models > 0:
        logging.warning(
            f"Ignoring {num_nan_models} models because they contain NaN values."
        )
    # Compute observed agreement
    logging.warning(
        f"Counting {'acceptances, make sure to use accuracy as baseline rate.' if count_accepted else 'rejections, make sure to use error rate as baseline rate.'}"
    )
    num_model_agreeing = get_obs_agreement_counts(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=y_true,
    )
    if not count_accepted:
        num_model_agreeing = M - num_model_agreeing

    N = len(num_model_agreeing)

    fraction_individuals = torch.arange(N, dtype=torch.float32)
    if relative_x:
        fraction_individuals /= N
        ax.set_xlim(-0.01, 1.05)

    # plot observed
    M_not_na = predictions.notna().sum(axis=1).to_numpy()
    fraction_models_observed = sorted(
        num_model_agreeing / M_not_na,
        reverse=at_least,
    )
    ax.plot(
        fraction_individuals,
        fraction_models_observed,
        color="C0",
        label="observed",
        zorder=0,
        alpha=alpha,
    )

    if show_monoc:
        # plot monoculture
        mean_rate = abs(float(not at_least) - np.mean(baseline_rates))
        # if count_accepted
        # else np.mean(baseline_rates)

        ax.step(
            x=[0.0, mean_rate, 1.0],
            y=[1.0, 0.0, 0.0] if at_least else [0.0, 1.0, 1.0],
            where="post",
            color="red",
            label="monoculture",
            linestyle="dotted",
            zorder=-1,
            alpha=alpha,
        )

    if baseline_rates:
        # compute expected agreement
        prob_expected = torch.tensor(
            [
                poisson_binom_agreement(baseline_rate=baseline_rates, k=num_agreeing)
                for num_agreeing in range(M + 1)
            ]
        )

        width = 0.4 / M if relative_x else 0.5
        # plot baseline
        fraction_individuals_at_rand = (
            1.0 - cumulative_sum(prob_expected)
            if at_least
            else cumulative_sum(prob_expected)
        )
        fraction_models = np.arange(M + 1) / M
        ax.step(
            fraction_individuals_at_rand + width,
            fraction_models,
            color="C1",
            label=baseline_label,
            zorder=-1,
            alpha=alpha,
        )

    ax.set_xticks(np.linspace(0, 1.0, num=num_ticks))

    if plot_pdf:
        # plot baseline
        print(M)
        width = 0.4 / M if relative_x else 0.5

        if baseline_rates:
            ax.barh(
                np.arange(M + 1) / M + width / 2,
                prob_expected,
                height=width,
                color="C1",
                label=baseline_label,
                alpha=0.7,
            )

        fun = (
            get_fraction_obs_acceptance_aggregated
            # get_obs_acceptance_aggregated
            if count_accepted
            else get_obs_rejections_aggregated
        )
        frac_models_accepting, frequencies = fun(
            predictions=predictions,
            restrict_only_pos_instances=restrict_only_pos_instances,
            restrict_only_neg_instances=restrict_only_neg_instances,
            true_labels=y_true,
            padding=False,
        )
        # print(type(frequencies))

        bins = np.linspace(0, 1, M)  # edges: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

        # Bin the values using pandas
        binned = pd.cut(frac_models_accepting, bins=bins, include_lowest=True)
        df = pd.DataFrame({"bin": binned, "vals": frequencies})
        bin_sums = df.groupby("bin")["vals"].sum()
        midpoints = bin_sums.index.map(lambda interval: float(interval.right)).astype(
            float
        )

        ax.barh(
            midpoints - (width / 2 if baseline_rates else -width / 2),
            bin_sums.values / N,
            height=width,
            color="C0",
            label="observed",
            alpha=0.7,
            zorder=0,
        )

    obs = np.column_stack((fraction_individuals, fraction_models_observed))
    at_rnd = (
        np.column_stack((fraction_individuals_at_rand, fraction_models))
        if baseline_rates
        else None
    )
    return (
        ax,
        obs[np.argsort(obs[:, 0])],
        at_rnd[np.argsort(at_rnd[:, 0])],
    )


def plot_recourse_lineplot_mean_stderr(
    ax: plt.Axes,
    predictions: list[torch.Tensor],
    baseline_rates: list[torch.Tensor] = None,
    y_true: torch.Tensor = None,
    restrict_only_pos_instances: bool = True,
    restrict_only_neg_instances: bool = False,
    xlabel: str = "fraction of positive instances",
    ylabel: str = "fraction of models accepting",
    title: str = "",
    baseline_label: str = "random error",
    relative_x: bool = True,
    count_accepted=True,
    at_least=True,
    show_monoc=True,
    plot_pdf=True,
    alpha=1.0,
    plot_all_samples=True,
    num_ticks=5,
):
    """
    x individuals are accepted by **at least** y models

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
    assert isinstance(predictions, list) and all(
        p.shape == predictions[0].shape for p in predictions
    )
    # Validate inputs
    _, M = predictions[0].shape
    if M == 0:
        print(ValueError("Predictions tensor is empty."))
        return ax

    # Configure the plot
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Compute observed agreement
    logging.warning(
        f"Counting {'acceptances, make sure to use accuracy as baseline rate.' if count_accepted else 'rejections, make sure to use error rate as baseline rate.'}"
    )

    fraction_models_agreeing_observed = []
    for preds in predictions:
        num_model_agreeing = get_obs_agreement_counts(
            predictions=preds,
            restrict_only_pos_instances=restrict_only_pos_instances,
            restrict_only_neg_instances=restrict_only_neg_instances,
            true_labels=y_true,
        )
        if not count_accepted:
            num_model_agreeing = M - num_model_agreeing

        # sort observed agreement
        fraction_models_agreeing_observed.append(
            sorted(num_model_agreeing / M, reverse=at_least)
        )

    N = len(fraction_models_agreeing_observed[0])
    fraction_individuals = torch.arange(N, dtype=torch.float32)
    if relative_x:
        fraction_individuals /= N
        ax.set_xlim(-0.01, 1.02)

    # plot observed
    mean_fraction_models_observed = np.array(fraction_models_agreeing_observed).mean(
        axis=0
    )
    stderr_fraction_models_observed = np.array(fraction_models_agreeing_observed).std(
        axis=0
    ) / np.sqrt(len(predictions))
    ax.plot(
        fraction_individuals,
        mean_fraction_models_observed,
        color="C0",
        label="observed",
        zorder=0,
        alpha=alpha,
    )
    ax.fill_between(
        fraction_individuals,
        mean_fraction_models_observed - stderr_fraction_models_observed,
        mean_fraction_models_observed + stderr_fraction_models_observed,
        color="C0",
        alpha=0.5 * alpha,
    )
    if plot_all_samples:
        lines = ax.get_lines()[0]
        for sample_frac in fraction_models_agreeing_observed:
            ax.plot(
                fraction_individuals,
                sample_frac,
                color="C0",
                label="observed",
                zorder=0,
                alpha=0.15,
                linewidth=0.5 * lines.get_linewidth(),
            )

    if show_monoc:
        # plot monoculture

        mean_rates = []
        for rates in baseline_rates:
            mean_rates.append(abs(float(not at_least) - np.mean(rates)))
        # if count_accepted
        # else np.mean(baseline_rates)

        ax.step(
            x=[0.0, np.array(mean_rates).mean(axis=0), 1.0],
            y=[1.0, 0.0, 0.0] if at_least else [0.0, 1.0, 1.0],
            where="post",
            color="red",
            label="monoculture",
            linestyle="dotted",
            zorder=-1,
            alpha=alpha,
        )

    if baseline_rates:
        # compute expected agreement
        fraction_individuals_at_rand = []
        for rates in baseline_rates:
            prob_expected = torch.tensor(
                [
                    poisson_binom_agreement(baseline_rate=rates, k=num_agreeing)
                    for num_agreeing in range(M + 1)
                ]
            )

            # plot baseline
            fraction_individuals_at_rand.append(
                1.0 - cumulative_sum(prob_expected)
                if at_least
                else cumulative_sum(prob_expected)
            )

        fraction_models = np.arange(M + 1) / M
        mean_fraction_individuals_at_rand = np.array(fraction_individuals_at_rand).mean(
            axis=0
        )
        stderr_fraction_models_observed = np.array(fraction_individuals_at_rand).std(
            axis=0
        ) / np.sqrt(len(baseline_rates))
        ax.plot(
            mean_fraction_individuals_at_rand,
            fraction_models,
            color="C1",
            label=baseline_label,
            zorder=-1,
            alpha=alpha,
        )

        if plot_all_samples:
            for sample_frac in fraction_individuals_at_rand:
                ax.plot(
                    sample_frac,
                    fraction_models,
                    color="C1",
                    label=baseline_label,
                    zorder=-1,
                    alpha=0.2,
                    linewidth=0.5 * lines.get_linewidth(),
                )

    ax.set_xticks(np.linspace(0, 1.0, num=num_ticks))

    if plot_pdf:
        # plot baseline
        width = 0.4 / M if relative_x else 0.5

        if baseline_rates:
            ax.barh(
                np.arange(M + 1) / M + width / 2,
                prob_expected,
                height=width,
                color="C1",
                label=baseline_label,
                alpha=0.7,
            )

        freqs = []
        for pred in predictions:
            fun = (
                get_obs_acceptance_aggregated
                if count_accepted
                else get_obs_rejections_aggregated
            )
            _, frequencies = fun(
                predictions=pred,
                restrict_only_pos_instances=restrict_only_pos_instances,
                restrict_only_neg_instances=restrict_only_neg_instances,
                true_labels=y_true,
                padding=True,
            )
            freqs.append(frequencies)

        print(frequencies)
        mean_frequencies = np.array(freqs).mean(axis=0)
        print(mean_frequencies)
        print(N)
        ax.barh(
            np.arange(M + 1) / M - (width / 2 if baseline_rates else 0),
            mean_frequencies / N,
            height=width,
            color="C0",
            label="observed",
            alpha=0.7,
            zorder=0,
        )

    obs = np.column_stack((fraction_individuals, mean_fraction_models_observed))
    at_rnd = (
        np.column_stack((mean_fraction_individuals_at_rand, fraction_models))
        if baseline_rates
        else None
    )
    return (
        ax,
        obs[np.argsort(obs[:, 0])],
        at_rnd[np.argsort(at_rnd[:, 0])],
    )


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

        axs[i] = plot_recourse_barplot(
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
    ylim=(0.45, 1.0),
    ylabel="",
    xlabel="",
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

    M_pairs = len(agreements_observed)
    fraction_model_pairs = np.arange(0, M_pairs - 1 + 0.01, 1.0) / (M_pairs - 1)
    ax.plot(
        fraction_model_pairs,
        np.ones_like(agreements_observed),
        label="monoculture",
        color="C3",
    )
    scatter_args = {"marker": "o", "markersize": 3} if plot_scatter else {}
    ax.plot(fraction_model_pairs, agreements_observed, label="observed", **scatter_args)
    ax.plot(fraction_model_pairs, agreements_exp, label="random error", **scatter_args)

    ax.plot(
        fraction_model_pairs,
        np.zeros_like(agreements_observed) + 0.5,
        label="random prediction",  # "chance level",
        color="C7",
        linestyle="dashed",
        linewidth=1,
        zorder=-1,
    )
    ax.fill_between(
        fraction_model_pairs,
        agreements_observed,
        np.ones_like(agreements_observed),
        color="C7",
        alpha=0.1,
        zorder=-2,
    )
    ax.set_ylim(bottom=ylim[0], top=ylim[1] + 0.01)
    ax.set_xlim(0, 1.05)
    ax.set_title(title.replace("ACS", "ACS ").replace("_", " "))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax
