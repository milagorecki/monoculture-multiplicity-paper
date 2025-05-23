# Measure reourse
# Collect all analysis functionalities

from typing import List, Union, Optional
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

# ------------------------------
# Baseline
# ------------------------------


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


# ------------------------------
# Agreement and Recourse
# ------------------------------


def pairwise_agreement_at_random(acc1: float, acc2: float) -> float:
    return acc1 * acc2 + (1 - acc1) * (1 - acc2)


def get_observed_pairwise_agreement_rate(
    predictions_m1: Union[pd.DataFrame, torch.Tensor],
    predictions_m2: Union[pd.DataFrame, torch.Tensor],
) -> float:
    assert (
        predictions_m1.shape == predictions_m2.shape
    ), "Dataframes have different shapes"
    # assert predictions_m1.shape[1] <= 1, "Only one column expected"
    valid_mask = predictions_m1.notna() & predictions_m2.notna()
    num_samples = predictions_m1[valid_mask].shape[0]
    observed_agreement = (
        predictions_m1[valid_mask].values == predictions_m2[valid_mask].values
    ).sum(axis=0) / num_samples
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


def get_fraction_no_recourse(
    predictions: Union[np.array, torch.Tensor],
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    count_acceptances=True,
):
    num_models, counts = get_model_agreement_histogram(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        count_acceptances=count_acceptances,
        padding=True,
    )
    # no recourse: num_models = 0
    return counts[0] / sum(counts)


def get_obs_acceptance_aggregated(
    predictions: Union[pd.DataFrame, torch.Tensor, np.array],
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
):
    return get_model_agreement_histogram(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=true_labels,
        padding=True,
        count_acceptances=True,
    )


def get_obs_rejections_aggregated(
    predictions: Union[pd.DataFrame, torch.Tensor, np.array],
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
):
    return get_model_agreement_histogram(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=true_labels,
        padding=True,
        count_acceptances=False,
    )


def get_fraction_obs_acceptance_aggregated(
    predictions: Union[pd.DataFrame, torch.Tensor, np.array],
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
    padding=True,  # not used, fractions cannot be meaningfully padded
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_model_agreement_histogram(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=true_labels,
        return_fractions=True,
        padding=False,
        count_acceptances=True,
    )


def get_fraction_obs_rejections_aggregated(
    predictions: Union[pd.DataFrame, torch.Tensor, np.array],
    restrict_only_pos_instances=False,
    restrict_only_neg_instances=False,
    true_labels: pd.Series = None,
    padding=False,  # not used, fractions cannot be meaningfully padded
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_model_agreement_histogram(
        predictions=predictions,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        true_labels=true_labels,
        return_fractions=True,
        padding=False,
        count_acceptances=False,
    )


def get_obs_agreement_counts(
    predictions: Union[pd.DataFrame, torch.Tensor, np.ndarray],
    true_labels: Optional[Union[pd.Series, torch.Tensor, np.ndarray]] = None,
    restrict_only_pos_instances: bool = False,
    restrict_only_neg_instances: bool = False,
    return_fractions: bool = False,
    count_acceptances: bool = True,
):

    assert not (
        restrict_only_pos_instances and restrict_only_neg_instances
    ), "Cannot restrict to both positive and negative instances."

    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.values
    if isinstance(true_labels, pd.Series):
        true_labels = true_labels.values

    N, M = predictions.shape

    # filter based on labels
    if restrict_only_pos_instances or restrict_only_neg_instances:
        assert true_labels is not None, "True labels are required to filter by label."
        label_val = int(restrict_only_pos_instances)
        mask = true_labels == label_val
        predictions = predictions[mask]

    counts_per_instance = predictions.sum(axis=1)
    if not count_acceptances:
        counts_per_instance = M - counts_per_instance

    if return_fractions:
        # filter out NaN values to compute fractions
        total_not_na_counts = np.sum(~np.isnan(predictions), axis=1)
        counts_per_instance = counts_per_instance / total_not_na_counts

    counts_per_instance = torch.tensor(counts_per_instance)

    return counts_per_instance


def get_model_agreement_histogram(
    predictions: Union[pd.DataFrame, torch.Tensor, np.ndarray],
    true_labels: Optional[Union[pd.Series, torch.Tensor, np.ndarray]] = None,
    restrict_only_pos_instances: bool = False,
    restrict_only_neg_instances: bool = False,
    count_acceptances: bool = True,
    return_fractions: bool = False,
    padding: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes histogram of model agreements (either acceptances or rejections).

    Parameters:
        predictions: Predictions (instances, models).
        true_labels: optional binary labels
        restrict_only_pos_instances: Whether to include only positive instances.
        restrict_only_neg_instances: Whether to include only negative instances.
        count_acceptances: If False, counts rejections instead.
        padding: whether to return a padded histogram (length M+1).

    Returns:
        (bin_values, counts): Tensors of unique counts or full histogram.
    """
    M = predictions.shape[1]
    counts_per_instance = get_obs_agreement_counts(
        predictions=predictions,
        true_labels=true_labels,
        restrict_only_pos_instances=restrict_only_pos_instances,
        restrict_only_neg_instances=restrict_only_neg_instances,
        return_fractions=return_fractions,
        count_acceptances=count_acceptances,
    )
    # aggregate counts into histogram
    values, counts = torch.unique(counts_per_instance, return_counts=True)

    # apply padding
    if padding:
        padded_counts = torch.zeros(M + 1, dtype=counts.dtype)
        padded_counts[values.long()] = counts
        return torch.arange(M + 1).numpy(), padded_counts.numpy()

    return values.numpy(), counts.numpy()


def get_agreement_matrix(models, dictionary: dict, fun: callable):
    matrix = torch.zeros(len(models), len(models))
    for i, mi in enumerate(models):
        for j in range(i, len(models)):
            mj = models[j]
            matrix[i][j] = fun(dictionary[mi], dictionary[mj])
            matrix[j][i] = matrix[i][j]
    return matrix


# ------------------------------
# Ambiguity and Discrepancy
# ------------------------------


def get_ambiguity(h0_predictions: pd.Series, predictions: pd.DataFrame):
    N, M = predictions.shape
    # check if same predictions as baseline
    assert all(
        h0_predictions.index == predictions.index
    ), "Index must match to compare correctly"
    # check where different from ref
    diff = predictions.ne(h0_predictions, axis=0)
    # check where not NaN
    valid_mask = predictions.notna()
    # only compare entries that are different and valid
    any_change = (diff & valid_mask).any(axis=1)
    return any_change.sum() / N


def get_discrepancy(h0_predictions: pd.Series, predictions: pd.DataFrame):
    N, M = predictions.shape
    # check if same predictions as baseline
    assert all(
        h0_predictions.index == predictions.index
    ), "Index must match to compare correctly"
    if predictions.isna().any().any():
        print("NaNs contained")
        nan_mask_col = predictions.isna().any()
        num_disagreements_with_h0 = (
            predictions.loc[:, ~nan_mask_col].ne(h0_predictions, axis=0).sum(axis=0)
        )
        max_disagree = num_disagreements_with_h0.max() / N
        # for columns with NaNs check only among those that are not NaN
        no_nan_rows = predictions.notna().all(axis=1)
        num_disagree_filtered_nan = (
            predictions.loc[no_nan_rows, nan_mask_col]
            .ne(h0_predictions[no_nan_rows], axis=0)
            .sum(axis=0)
        ) / h0_predictions[no_nan_rows].shape[0]
        print("reduces to", h0_predictions.shape)
        return max(num_disagree_filtered_nan.max(), max_disagree)
    else:
        num_disagreements_with_h0 = predictions.ne(h0_predictions, axis=0).sum(axis=0)
        return num_disagreements_with_h0.max() / N


def get_ambiguity_all_models(predictions: pd.DataFrame):
    # assume every column contains the predictions of one model
    N, M = predictions.shape
    num_accept = predictions.sum(axis=1)
    # count how often at least 1 model disagrees (1 to (M-1) accepts)
    return num_accept[(num_accept > 0) & (num_accept < M)].shape[0] / N


def get_discrepancy_all_models(predictions: pd.DataFrame):
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


# ------------------------------
# UTILS
# ------------------------------


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
