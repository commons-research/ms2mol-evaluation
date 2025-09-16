import typing as T

import numpy as np
import pandas as pd
from tqdm import tqdm


def get_fraction_of_true(
    scores: np.ndarray,
    columns: T.List[str],
    index: T.List[str],
    id_to_inchikey: T.Dict[str, str],
    score_threshold=0.0,
) -> T.Tuple[float, float]:
    """
    Given a threshold, this function calculates the fraction of rows
    where the score for the true compound is above the threshold
    """
    scores_smaller = scores.copy()
    scores_smaller[scores_smaller < score_threshold] = np.nan
    df_smaller = pd.DataFrame(scores_smaller)
    df_smaller.columns = columns
    df_smaller.index = index

    # we iterate over the rows of the dataframe
    fraction_of_true_among_df = 0
    for i, row in df_smaller.iterrows():
        if np.isnan(row[id_to_inchikey[row.name]]):
            continue
        fraction_of_true_among_df += 1

    fraction_of_true = fraction_of_true_among_df / df_smaller.shape[0]
    fraction_of_df = 1 - (df_smaller.isna().sum().sum() / df_smaller.size)
    return fraction_of_true, fraction_of_df


def get_fraction_of_true_top_n(
    scores: np.ndarray,
    columns: T.List[str],
    index: T.List[str],
    id_to_inchikey: T.Dict[str, str],
    top_n: int = 1,
) -> T.Tuple[float, float]:
    scores_top_n = np.full_like(scores, np.nan)
    for i in range(scores.shape[0]):
        row = scores[i]
        if np.all(np.isnan(row)):
            continue
        # Get indices of top N scores (ignoring NaNs)
        valid_idx = np.where(~np.isnan(row))[0]
        if len(valid_idx) == 0:
            continue
        top_idx = valid_idx[np.argsort(row[valid_idx])[-top_n:]]
        scores_top_n[i, top_idx] = row[top_idx]

    df_top_n = pd.DataFrame(scores_top_n, columns=columns, index=index)

    fraction_of_true_among_df = 0
    for i, row in df_top_n.iterrows():
        if np.isnan(row[id_to_inchikey[row.name]]):
            continue
        fraction_of_true_among_df += 1

    fraction_of_true = fraction_of_true_among_df / df_top_n.shape[0]
    fraction_of_df = 1 - (df_top_n.isna().sum().sum() / df_top_n.size)
    return fraction_of_true, fraction_of_df


def evaluate_fraction_of_true(
    scores: np.ndarray,
    all_inchikeys: T.List[str],
    identifiers: T.List[str],
    identifier_to_inchikey: T.Dict[str, str],
    interval: T.Iterable[float] = np.arange(0.0, 1.0, 0.05),
) -> T.Tuple[T.List[float], T.List[float]]:

    fraction_true_lst = []
    fraction_df_lst = []
    for threshold in tqdm(interval, desc="Thresholds"):
        fraction_true, fraction_df = get_fraction_of_true(
            scores,
            all_inchikeys,
            identifiers,
            identifier_to_inchikey,
            score_threshold=threshold,
        )
        fraction_true_lst.append(fraction_true)
        fraction_df_lst.append(fraction_df)

    return fraction_true_lst, fraction_df_lst


def evaluate_top_n_of_true(
    scores: np.ndarray,
    all_inchikeys: T.List[str],
    identifiers: T.List[str],
    identifier_to_inchikey: T.Dict[str, str],
    interval: T.Iterable[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500],
) -> T.Tuple[T.List[float], T.List[float]]:

    fraction_true_lst = []
    fraction_df_lst = []
    for threshold in tqdm(interval, desc="Top N"):
        fraction_true, fraction_df = get_fraction_of_true_top_n(
            scores,
            all_inchikeys,
            identifiers,
            identifier_to_inchikey,
            top_n=threshold,
        )
        fraction_true_lst.append(fraction_true)
        fraction_df_lst.append(fraction_df)

    return fraction_true_lst, fraction_df_lst
