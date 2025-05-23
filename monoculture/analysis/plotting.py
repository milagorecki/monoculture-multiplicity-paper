from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

# import matplotlib as mpl
import torch
import scipy as sp
import pandas as pd
import numpy as np
from typing import Tuple, Union
from matplotlib import axes
from .utils import (
    prettify_model_name,
    key_to_model,
    get_size_and_it,
    cumulative_sum,
    df_to_dict,
)
from .metrics import (
    poisson_binom_agreement,
    get_obs_agreement_counts,
    get_obs_acceptance_aggregated,
    get_obs_rejections_aggregated,
    get_fraction_obs_acceptance_aggregated,
    get_fraction_obs_rejections_aggregated,
)
from pathlib import Path
import matplotlib.colors as pltcolors
import logging

# -------------------------------------------------------------
# GENERAL PERFORMANCE
# -------------------------------------------------------------


def plot_accuracies(
    ax: axes,
    evals: dict,
    baseline_accs: dict,
    models_above_baseline: list,
    ylims: Tuple = (0, 0.85),
    label_rotation: int = 90,
    ha: str = "center",
    color: str = "grey",
    highlight_color: str = "black",
    hline_xmin=0,
    hline_xmax=1,  #
    marker="o",
    scatter_label=None,
):
    if baseline_accs:
        # plot hline for baselines
        for baseline, c in [("Constant", "dodgerblue"), ("XGBoost", "yellowgreen")]:
            ax.hlines(
                baseline_accs[baseline],
                xmin=hline_xmin,
                xmax=hline_xmax,
                label=baseline,
                colors=c,
                linestyle="-",
                zorder=0,
            )

    # plot model accuracies
    x_labels = []
    scatter_labels = set()
    for idx, (model_key, val) in enumerate(
        sorted(evals.items(), key=lambda item: get_size_and_it(item[0]))
    ):
        keep_model = model_key in models_above_baseline
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
        if m in models_above_baseline:
            idx = x_labels.index(m)
            plt.setp(ax.get_xticklabels()[idx], color=highlight_color)
    ax.set_ylim(ylims)
    ax.set_xlim(-0.5, right=len(x_labels))

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
    fractions_neg_pred: dict,
    models_above_baseline: list,
    ylims=(0, 1),
    label_rotation: int = 90,
    ha: str = "center",
    color: str = "grey",
    highlight_color: str = "black",
    omit_non_highlight_labels=False,
    bar_width=0.5,
):
    x_labels = []
    models_sorted = sorted(fractions_neg_pred.keys(), key=get_size_and_it)
    for idx, model_key in enumerate(models_sorted):
        keep_model = model_key in models_above_baseline
        ax.bar(
            idx,
            fractions_neg_pred[model_key],
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
            if m in models_above_baseline:
                idx = x_labels.index(m)
                plt.setp(ax.get_xticklabels()[idx], color=highlight_color)
    else:
        ax.set_xticks(
            torch.arange(len(x_labels)),
            [
                (
                    prettify_model_name(key_to_model(key))
                    if key in models_above_baseline
                    else ""
                )
                for key in x_labels
            ],
            rotation=label_rotation,
            ha=ha,
            color=highlight_color,
        )
    ax.set_ylim(ylims)
    ax.set_xlim(-0.5, right=len(x_labels))

    return ax


def plot_general_performance_task(
    axs_row,
    df,
    baseline_evals,
    ytrue,
    models_to_highlight,
    marker: str = "o",
    scatter_label=None,
    acc="accuracy",
    label_rotation: int = 90,
    ha: str = "center",
):
    """Plot label distribution, accuracy and fraction of negative predictions next to each other for a given task"""
    if len(axs_row) == 3:
        ax_label, ax_accuracy, ax_frac_neg = axs_row
    else:
        ax_label, ax_accuracy = axs_row
    # label distribution
    plot_label_dist(ax_label, ytrue, ylims=(0, 0.95))

    # model accuracy
    acc_dict = df_to_dict(df[["model", acc]])
    plot_accuracies(
        ax=ax_accuracy,
        evals=acc_dict,
        baseline_accs={
            baseline: evals[acc] for baseline, evals in baseline_evals.items()
        },
        models_above_baseline=models_to_highlight,
        ylims=(0, 0.95),
        hline_xmin=-0.2,
        hline_xmax=len(acc_dict.keys()) + 0.1,
        marker=marker,
        scatter_label=scatter_label,
        label_rotation=label_rotation,
        ha=ha,
    )

    if len(axs_row) == 3:
        # fraction of negative predicted samples
        frac_neg_pred_dict = df_to_dict(
            pd.concat(
                [df["model"], df["num_pred_negatives"] / df["n_samples"]], axis=1
            ).rename(columns={0: "frac_negative_preds"})
        )
        plot_neg_predicted(
            ax_frac_neg,
            fractions_neg_pred=frac_neg_pred_dict,
            models_above_baseline=models_to_highlight,
            label_rotation=label_rotation,
            ha=ha,
        )


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


def plot_recourse_lineplot(
    ax: plt.Axes,
    predictions: Union[torch.Tensor, pd.DataFrame],
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
    if predictions.shape[1] == 0:
        print(ValueError("Predictions tensor is empty."))
        return ax
    M = predictions.shape[1]
    bar_width = 0.4 / M if relative_x else 0.5

    if show_monoc:
        plot_monoculture_line(
            ax=ax,
            baseline_rates=baseline_rates,
            at_least=at_least,
        )

    if baseline_rates:
        fraction_individuals_at_random, fraction_models = (
            compute_baseline_recourse_curve(baseline_rates, at_least=at_least)
        )

        plot_recourse_curve(
            ax,
            fraction_individuals=fraction_individuals_at_random,
            fraction_models=fraction_models,
            color="C1",
            label=baseline_label,
        )

        if plot_pdf:
            prob_expected = compute_expected_agreement_distribution(baseline_rates)

            plot_recourse_pdf(
                ax,
                fraction_models,
                prob_expected,
                label=baseline_label,
                width=bar_width,
                shift=-bar_width / 2,
                color="C1",
            )

    fraction_individuals, fraction_models_observed = compute_observed_recourse_curve(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        y_true=y_true,
        at_least=at_least,
        count_acceptances=count_accepted,
    )

    plot_recourse_curve(
        ax=ax,
        fraction_individuals=fraction_individuals,
        fraction_models=fraction_models_observed,
        color="C0",
        label="observed",
    )

    if plot_pdf:
        fun = (
            get_fraction_obs_acceptance_aggregated
            if count_accepted
            else get_fraction_obs_rejections_aggregated
        )
        frac_models, frequencies = fun(
            predictions=predictions,
            restrict_only_pos_instances=restrict_only_pos_instances,
            restrict_only_neg_instances=restrict_only_neg_instances,
            true_labels=y_true,
            padding=False,
        )

        edges, values = bin_fractions(
            M, fractions_models=frac_models, frequencies=frequencies
        )

        plot_recourse_pdf(
            ax,
            edges,
            values / sum(frequencies),
            label="observed",
            width=bar_width,
            shift=bar_width / 2 if baseline_rates else 0,
            color="C0",
        )

    # Configure the plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(0, 1.0, num=num_ticks))
    ax.set_xlim(-0.01, 1.05)

    obs = np.column_stack((fraction_individuals, fraction_models_observed))
    at_rnd = (
        np.column_stack((fraction_individuals_at_random, fraction_models))
        if baseline_rates
        else None
    )
    return (
        ax,
        obs[np.argsort(obs[:, 0])],
        at_rnd[np.argsort(at_rnd[:, 0])],
    )


def bin_fractions(M, fractions_models, frequencies):
    # Bin the fractions (only relevant when some models contains NaN predictions)
    bins = np.concatenate([[-np.inf], np.linspace(0.0, 1.0, M + 1)])
    binned = pd.cut(fractions_models, bins=bins, include_lowest=True, right=True)
    df = pd.DataFrame({"bin": binned, "vals": frequencies})
    bin_sums = df.groupby("bin", observed=False)["vals"].sum()
    edges = bin_sums.index.map(lambda interval: float(interval.right)).astype(float)
    return edges, bin_sums.values


def compute_expected_agreement_distribution(baseline_rates):
    M = len(baseline_rates)
    prob_expected = torch.tensor(
        [
            poisson_binom_agreement(baseline_rate=baseline_rates, k=num_agreeing)
            for num_agreeing in range(M + 1)
        ]
    )
    return prob_expected


def compute_observed_recourse_curve(
    predictions,
    y_true,
    restrict_only_pos_instances=True,
    restrict_only_neg_instances=False,
    at_least=True,
    count_acceptances=True,
):
    num_nan_models = predictions.isna().any().sum()
    if num_nan_models > 0:
        logging.warning(
            f"Ignoring {num_nan_models} models because they contain NaN values."
        )
    # Compute observed agreement
    logging.warning(
        f"Counting {'acceptances'if count_acceptances else 'rejections'}, make sure to use correct baseline rate."
    )

    # get the number of models agreeing for each inidvidual
    num_models_agreeing = get_obs_agreement_counts(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=y_true,
        count_acceptances=count_acceptances,
    )
    # for every individual, count the number of models that agreed, ignore NaN predictions
    M_not_na = predictions.notna().sum(axis=1).to_numpy()
    fraction_models_observed = sorted(
        num_models_agreeing / M_not_na,
        reverse=at_least,
    )

    # number of individuals for which we look at agreements
    N = len(num_models_agreeing)
    fraction_individuals = np.linspace(0, 1, num=N)
    return fraction_individuals, fraction_models_observed


def compute_baseline_recourse_curve(baseline_rates, at_least=True):
    M = len(baseline_rates)
    prob_expected = compute_expected_agreement_distribution(
        baseline_rates=baseline_rates
    )
    fraction_individuals_at_random = (
        1.0 - cumulative_sum(prob_expected)
        if at_least
        else cumulative_sum(prob_expected)
    )
    fraction_models = np.linspace(0, 1, num=M + 1)
    return fraction_individuals_at_random, fraction_models


def plot_recourse_pdf(
    ax,
    fraction_models,
    fraction_individuals,
    label,
    width,
    shift=0,
    color="C0",
):
    ax.barh(
        fraction_models - shift,
        fraction_individuals,
        height=width,
        color=color,
        label=label,
        alpha=0.7,
        zorder=0,
    )


def plot_recourse_curve(
    ax, fraction_individuals, fraction_models, color, label="", **kwargs
):
    ax.step(
        fraction_individuals,
        fraction_models,
        color=color,
        label=label,
        zorder=0,
        alpha=1,
        **kwargs,
    )


def plot_monoculture_line(ax, baseline_rates, at_least=True, alpha=1):
    mean_rate = np.mean(baseline_rates)
    ref = abs(float(not at_least) - mean_rate)
    ax.step(
        x=[0.0, ref, 1.0],
        y=[1.0, 0.0, 0.0] if at_least else [0.0, 1.0, 1.0],
        where="post",
        linestyle="dotted",
        color="red",
        alpha=alpha,
        label="Monoculture",
        zorder=-1,
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
    bar_width = 0.4 / M if relative_x else 0.5

    if show_monoc and baseline_rates is not None:
        mean_rates = []
        for rates in baseline_rates:
            mean_rates.append(abs(float(not at_least) - np.mean(rates)))
        plot_monoculture_line(ax, baseline_rates=mean_rates, at_least=at_least)

    if baseline_rates:
        # compute expected agreement
        fracs_individual_at_random = []
        for rates in baseline_rates:
            fraction_individuals_at_random, fraction_models = (
                compute_baseline_recourse_curve(baseline_rates=rates, at_least=at_least)
            )

            fracs_individual_at_random.append(fraction_individuals_at_random)
        mean_frac_individuals_at_rand = np.array(fracs_individual_at_random).mean(
            axis=0
        )

        plot_recourse_curve(
            ax,
            fraction_individuals=mean_frac_individuals_at_rand,
            fraction_models=fraction_models,
            color="C1",
            label=baseline_label,
        )

        if plot_all_samples:
            lines = ax.get_lines()[0]
            for i, sample_frac_individuals in enumerate(fracs_individual_at_random):
                plot_recourse_curve(
                    ax,
                    fraction_individuals=sample_frac_individuals,
                    fraction_models=fraction_models,
                    color="C1",
                    label=baseline_label if i == 0 else None,
                    alpha=0.2,
                    linewidth=0.5 * lines.get_linewidth(),
                )

        if plot_pdf:
            prob_expected = compute_expected_agreement_distribution(baseline_rates)

            plot_recourse_pdf(
                ax,
                fraction_models,
                prob_expected,
                label=baseline_label,
                width=bar_width,
                shift=-bar_width / 2,
                color="C1",
            )

    # observed
    fracs_models_observed = []
    for preds in predictions:

        fraction_individuals, fraction_models_observed = (
            compute_observed_recourse_curve(
                predictions=preds,
                restrict_only_pos_instances=restrict_only_pos_instances,
                restrict_only_neg_instances=restrict_only_neg_instances,
                y_true=y_true,
                at_least=at_least,
                count_acceptances=count_accepted,
            )
        )
        fracs_models_observed.append(fraction_models_observed)

    mean_frac_models_observed = np.array(fracs_models_observed).mean(axis=0)
    stderr_frac_models_observed = np.array(fracs_models_observed).std(axis=0) / np.sqrt(
        len(predictions)
    )

    plot_recourse_curve(
        ax=ax,
        fraction_individuals=fraction_individuals,
        fraction_models=mean_frac_models_observed,
        color="C0",
        label="observed",
    )

    ax.fill_between(
        fraction_individuals,
        mean_frac_models_observed - stderr_frac_models_observed,
        mean_frac_models_observed + stderr_frac_models_observed,
        color="C0",
        alpha=0.5 * alpha,
    )

    if plot_all_samples:
        lines = ax.get_lines()[0]
        for i, sample_frac_models in enumerate(fracs_models_observed):
            plot_recourse_curve(
                ax,
                fraction_individuals=fraction_individuals,
                fraction_models=sample_frac_models,
                color="C0",
                label="observed" if i == 0 else None,
                alpha=0.2,
                linewidth=0.5 * lines.get_linewidth(),
            )

    if plot_pdf:
        freqs = []
        for pred in predictions:
            fun = (
                get_fraction_obs_acceptance_aggregated
                if count_accepted
                else get_fraction_obs_rejections_aggregated
            )

            frac_models, frequencies = fun(
                predictions=pred,
                restrict_only_pos_instances=restrict_only_pos_instances,
                restrict_only_neg_instances=restrict_only_neg_instances,
                true_labels=y_true,
                padding=False,
            )

            edges, values = bin_fractions(
                M, fractions_models=frac_models, frequencies=frequencies
            )
            freqs.append(values)
        mean_frequencies = np.array(freqs).mean(axis=0)

        plot_recourse_pdf(
            ax,
            edges,
            mean_frequencies,
            label="observed",
            width=bar_width,
            shift=bar_width / 2 if baseline_rates else 0,
            color="C0",
        )

    # Configure the plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(0, 1.0, num=num_ticks))
    ax.set_xlim(-0.01, 1.05)

    obs = np.column_stack((fraction_individuals, mean_frac_models_observed))
    at_rnd = (
        np.column_stack((mean_frac_individuals_at_rand, fraction_models))
        if baseline_rates
        else None
    )
    return (
        ax,
        obs[np.argsort(obs[:, 0])],
        at_rnd[np.argsort(at_rnd[:, 0])],
    )


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


# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------


def configure_legend(fig, axs, offset=0.32):
    # monoculture
    solid_line = mlines.Line2D([], [], color="red", linestyle="-")
    dotted_line = mlines.Line2D([], [], color="red", linestyle=":")

    handles, labels = axs.flat[0].get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h

    unique["monoculture"] = (solid_line, dotted_line)

    desired_order = ["observed", "random error", "random prediction", "monoculture"]
    ordered_labels = [label for label in desired_order if label in unique]
    ordered_handles = [unique[label] for label in ordered_labels]

    lowest_y = min(ax.get_position().y0 for ax in axs.flat)

    fig.legend(
        ordered_handles,
        ordered_labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="lower center",
        ncol=len(ordered_labels),
        bbox_to_anchor=(0.5, lowest_y - offset),
        frameon=False,
        framealpha=1.0,
        edgecolor="black",
        fancybox=True,
    )
