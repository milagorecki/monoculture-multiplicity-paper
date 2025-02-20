## Measure reourse
# Collect all analysis functionalities
from typing import List, Union
import pandas as pd
import torch
import numpy as np

import more_itertools
from functools import reduce
from operator import mul, add
import logging


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


def expected_agreement_two_models(acc1: float, acc2: float) -> float:
    return acc1 * acc2 + (1 - acc1) * (1 - acc2)


## AGREEMENT RATES


def get_observed_agreement(
    predictions_m1: Union[pd.DataFrame, torch.Tensor],
    predictions_m2: Union[pd.DataFrame, torch.Tensor],
) -> float:
    assert (
        predictions_m1.shape == predictions_m2.shape
    ), "Dataframes have different shapes"
    assert predictions_m1.shape[1] == 1, "Only one column expected"
    num_samples = predictions_m1.shape[0]
    observed_agreement = (predictions_m1.values == predictions_m2.values).sum(
        axis=0
    ) / num_samples
    return float(observed_agreement[0])


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
    return float(observed_agreement[0])


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
    return float(observed_agreement[0])
