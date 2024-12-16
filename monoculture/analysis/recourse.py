## Measure reourse
# Collect all analysis functionalities
from typing import List, Union
import pandas as pd
import torch

import more_itertools
from functools import reduce
from operator import mul, add


def baseline_agreement(
    fn_rates: Union[List[float], torch.Tensor],
    k: int,
    epsilon: float = 0.0001,
):
    """
    Poisson Binomal Model: Assume each model to correspond to an independent
    Bernoulli trial with the sucess probability given by the TPR (1-FNR).
    What is the probability of k models to reject?
    """
    M = len(fn_rates)
    # make sure none of the rates is exactly 0 or 1, otherwise one term will set whole product to 0 --- test line length2
    fn_rates = [r + (epsilon * (r == 0)) - (epsilon * (r == 1)) for r in fn_rates]

    # creatd initial list to indicate which model rejects
    rej_indicators = torch.concat(
        [torch.ones(k, dtype=int), torch.zeros(M - k, dtype=int)]
    ).tolist()
    # get all unique permutations of k rejections
    unique_permutations = more_itertools.distinct_permutations(rej_indicators)

    products = []
    for indices in unique_permutations:
        # for each permutations, compute the product of fnrs
        products.append(
            reduce(
                mul,
                [
                    p ** indices[i] * (1 - p) ** (1 - indices[i])
                    for i, p in enumerate(fn_rates)
                ],
            )
        )
    # sum over all permutations
    return reduce(add, products)


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
