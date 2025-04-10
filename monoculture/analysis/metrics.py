# Measure reourse
# Collect all analysis functionalities

from typing import List, Union
import pandas as pd
import torch
import itertools
import more_itertools
import logging

# import numpy as np
# from functools import reduce
# from operator import mul, add


def poisson_binom_agreement(
    fn_rates: Union[List[float], torch.Tensor],
    k: int,
    epsilon: float = 0.0001,
):
    """
    Poisson Binomal Model: Assume each model to correspond to an independent Bernoulli trial with the sucess probability given by the TPR (1-FNR). What is the probability of k models to reject?
    """
    M = len(fn_rates)
    # make sure none of the rates is exactly 0 or 1, otherwise one term will set whole product to 0 --- test line length
    fn_rates = torch.Tensor(
        [r + (epsilon * (r == 0)) - (epsilon * (r == 1)) for r in fn_rates]
    )

    if M <= 30:
        logging.info("Computing exact expectation using closed form.")
        # created initial list to indicate which model rejects
        rej_indicators = torch.concat(
            [torch.ones(k, dtype=bool), torch.zeros(M - k, dtype=bool)]
        ).tolist()
        # get all unique permutations of k rejections
        logging.info(f"Getting distinct permutations for {M} models and {k} rejects")
        unique_permutations = list(more_itertools.distinct_permutations(rej_indicators))

        logging.debug(
            f"Compute probabilities under Poisson binomal model for {len(unique_permutations)} distinct permutations"
        )
        permutation_mat = torch.Tensor(unique_permutations).bool()
        fn_rates_permut = torch.matmul(permutation_mat.float(), torch.diag(fn_rates))
        tp_rates_permut = torch.matmul(
            (~permutation_mat).float(), torch.diag(1 - fn_rates)
        )

        # products = []
        # for indices in unique_permutations:
        #     # for each permutations, compute the product of fnrs
        #     fnr_or_tpr_based_on_permutation = [
        #         p ** indices[i] * (1 - p) ** (1 - indices[i])
        #         for i, p in enumerate(fn_rates)
        #     ]
        #     print(indices, fnr_or_tpr_based_on_permutation)
        #     products.append(
        #         reduce(
        #             mul,
        #             fnr_or_tpr_based_on_permutation,
        #         )
        #     )
        # sum over all permutations
        # return reduce(add, products)
        return torch.prod(tp_rates_permut + fn_rates_permut, axis=1).sum(axis=0).item()
    else:
        logging.info(
            f"M too large, Estimate expectation by sampling fixed number of permutations."
        )
        raise NotImplementedError("Simulated expectation is not yet implemented")


def expected_pairwise_agreement(acc1: float, acc2: float) -> float:
    return acc1 * acc2 + (1 - acc1) * (1 - acc2)


def expected_agreement_modelset(model_accuracies: dict | list, k: int = None):
    # if not k, compute for every possible k
    pass


# -------------------------------------------------------------
# AGREEMENT RATES
# -------------------------------------------------------------


def get_observed_pairwise_agreement(
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


def get_pairwise_neg_agreement(
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


def get_pairwise_pos_agreement(
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


def get_agreement_matrix(models, dictionary: dict, fun: callable):
    matrix = torch.zeros(len(models), len(models))
    for i, mi in enumerate(models):
        for j in range(i, len(models)):
            mj = models[j]
            matrix[i][j] = fun(dictionary[mi], dictionary[mj])
            matrix[j][i] = matrix[i][j]
    return matrix


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
