# Measure reourse
# Collect all analysis functionalities

from typing import List, Union
import pandas as pd
import torch
import itertools
import numpy as np

# import more_itertools
# import logging
import scipy as sp
from .setup import LLM_MODELS
from .utils import key_to_model

# import numpy as np
# from functools import reduce
# from operator import mul, add

# -------------------------------------------------------------
# AGREEMENT RATES
# -------------------------------------------------------------


# AGREEMENT AT RANDOM
def poisson_binom_agreement(
    baseline_rate: Union[List[float], torch.Tensor],
    k: int,
):
    """
    Poisson Binomal Model: Assume each model to correspond to an independent
    Bernoulli trial with the sucess probability given by the TPR (1-FNR).
    What is the probability of k models to accept (TPR as baseline_rate)?
    """
    return sp.stats.poisson_binom.pmf(k=k, p=baseline_rate)


def pairwise_agreement_at_random(acc1: float, acc2: float) -> float:
    return acc1 * acc2 + (1 - acc1) * (1 - acc2)


# AGREEMENT OBSERVED
def get_observed_pairwise_agreement_rate(
    predictions_m1: Union[pd.DataFrame, torch.Tensor],
    predictions_m2: Union[pd.DataFrame, torch.Tensor],
) -> float:
    assert (
        predictions_m1.shape == predictions_m2.shape
    ), "Dataframes have different shapes"
    # assert predictions_m1.shape[1] <= 1, "Only one column expected"
    num_samples = predictions_m1.shape[0]
    observed_agreement = (predictions_m1.values == predictions_m2.values).sum(
        axis=0
    ) / num_samples
    return observed_agreement  # float(observed_agreement[0])


def get_pairwise_neg_agreement_rate(
    predictions_m1: Union[pd.DataFrame, torch.Tensor],
    predictions_m2: Union[pd.DataFrame, torch.Tensor],
) -> float:
    assert (
        predictions_m1.shape == predictions_m2.shape
    ), "Dataframes have different shapes"
    num_samples = predictions_m1.shape[0]
    observed_agreement = (
        (predictions_m1 == 0).values & (predictions_m2 == 0).values
    ).sum(axis=0) / num_samples
    return observed_agreement  # float(observed_agreement[0])


def get_pairwise_pos_agreement_rate(
    predictions_m1: Union[pd.DataFrame, torch.Tensor],
    predictions_m2: Union[pd.DataFrame, torch.Tensor],
) -> float:
    assert (
        predictions_m1.shape == predictions_m2.shape
    ), "Dataframes have different shapes"
    num_samples = predictions_m1.shape[0]
    observed_agreement = (
        (predictions_m1 == 1).values & (predictions_m2 == 1).values
    ).sum(axis=0) / num_samples
    return observed_agreement  # float(observed_agreement[0])


def observed_agreement_wrapper(
    m1: str,
    m2: str,
    predictions: pd.DataFrame,
    fun=get_observed_pairwise_agreement_rate,
):
    assert (m1 in predictions.columns) & (
        m2 in predictions.columns
    ), "Both models must be present as column in predictions"

    return fun(
        predictions[m1],
        predictions[m2],
    )


def ratio_agreement_wrapper(m1, m2, predictions, evals, metric="accuracy"):
    assert (m1 in predictions.columns) & (
        m2 in predictions.columns
    ), "Both models must be present as column in predictions"

    observed = observed_agreement_wrapper(
        m1, m2, predictions, fun=get_observed_pairwise_agreement_rate
    )
    expected = pairwise_agreement_at_random(
        evals[metric][m1],
        evals[metric][m2],
    )
    return observed / expected


def diff_agreement_wrapper(m1, m2, predictions, evals, metric="accuracy"):
    assert (m1 in predictions.columns) & (
        m2 in predictions.columns
    ), "Both models must be present as column in predictions"

    observed = observed_agreement_wrapper(
        m1, m2, predictions, fun=get_observed_pairwise_agreement_rate
    )
    expected = pairwise_agreement_at_random(
        evals[metric][m1],
        evals[metric][m2],
    )
    return observed - expected


def get_fraction_no_recourse(predictions_array: Union[np.array, torch.Tensor]):
    if isinstance(predictions_array, torch.Tensor):
        predictions_array = predictions_array.to_numpy()
    return (
        np.sum(np.bitwise_or.reduce(predictions_array, axis=1) == 0)
        / predictions_array.shape[0]
    )


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


def get_obs_agreement_counts(
    predictions: pd.DataFrame,
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
):
    assert not (restrict_only_pos_instances and restrict_only_neg_instances)
    if restrict_only_pos_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 1]
    elif restrict_only_neg_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 0]
    num_accepting = predictions.sum(axis=1).values
    return num_accepting


def get_obs_acceptance_aggregated(
    predictions: Union[pd.DataFrame, torch.Tensor, np.array],
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
    padding=True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.to_numpy()
    N, M = predictions.shape

    assert not (restrict_only_pos_instances and restrict_only_neg_instances)
    if restrict_only_pos_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 1]
    elif restrict_only_neg_instances:
        assert true_labels is not None, "Provide labels to restrict data."
        predictions = predictions[true_labels == 0]
    counts_accepting = predictions.sum(axis=1)
    # aggregate
    num_models_accepting, counts = torch.Tensor(counts_accepting).unique(
        return_counts=True
    )

    if padding:
        counts_padded = torch.zeros(M + 1)
        for i in range(len(num_models_accepting)):
            counts_padded[int(num_models_accepting[i])] = counts[i]
        return torch.arange(M + 1), counts_padded

    return num_models_accepting, counts


def get_obs_rejections_aggregated(
    predictions: pd.DataFrame,
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
    padding=True,
):
    N, M = predictions.shape

    num_models_accepting, counts = get_obs_acceptance_aggregated(
        predictions,
        restrict_only_pos_instances,
        restrict_only_neg_instances,
        true_labels,
        padding,
    )
    num_models_rejecting = num_models_accepting
    counts = counts.flip(dims=[0])

    return num_models_rejecting, counts


def get_agreement_matrix(models, dictionary: dict, fun: callable):
    matrix = torch.zeros(len(models), len(models))
    for i, mi in enumerate(models):
        for j in range(i, len(models)):
            mj = models[j]
            matrix[i][j] = fun(dictionary[mi], dictionary[mj])
            matrix[j][i] = matrix[i][j]
    return matrix


def get_ambiguity(predictions: pd.DataFrame):
    # assume every column contains the predictions of one model
    N, M = predictions.shape
    num_accept = predictions.sum(axis=1)
    # count how often at least 1 model disagrees (1 to (M-1) accepts)
    return num_accept[(num_accept > 0) & (num_accept < M)].shape[0] / N


def get_discrepancy(predictions: pd.DataFrame):
    # assume every column contains the predictions of one model
    # find max number of different predictions between any two columns (Hamming dist)
    N, M = predictions.shape
    assert all(
        [key_to_model(col) in LLM_MODELS for col in predictions.columns]
    ), "Columns should be only model predictions (column name = model key)."

    max_diff = 0
    for col1, col2 in itertools.combinations(predictions, 2):
        diff = (predictions[col1] != predictions[col2]).sum()
        max_diff = max(max_diff, diff)

    return max_diff / N


# ---------------
# UTILS
# ----------------


def matrix_pairwise_evals(models, fun, only_lower_diag=True):
    """evaluate function on each pair of models"""
    num_models = len(models)
    model_combinations = list(
        itertools.combinations_with_replacement(range(num_models), 2)
    )  # _with_replacement only to add diagonal

    matrix = torch.zeros(num_models, num_models).fill_(torch.nan)
    for idx1, idx2 in model_combinations:
        m1 = models[idx1]
        m2 = models[idx2]
        matrix[idx1, idx2] = fun(m1, m2)
        if not only_lower_diag:
            matrix[idx2, idx1] = matrix[idx1, idx2]

    # return lower diagonal ma
    return matrix.T
