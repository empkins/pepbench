from collections.abc import Sequence

import numpy as np
import pandas as pd

__all__ = ["match_heartbeat_lists"]

from scipy.spatial import KDTree, minkowski_distance


def match_heartbeat_lists(
    *,
    heartbeats_reference: pd.DataFrame,
    heartbeats_extracted: pd.DataFrame,
    tolerance_ms: int | float = 10,
    sampling_rate_hz: float,
    one_to_one: bool = True,
    heartbeats_reference_suffix: str | None = "_reference",
    heartbeats_extracted_suffix: str | None = "",
) -> pd.DataFrame:
    """Find True Positives, False Positives and True Negatives by comparing extracted heartbeats with ground truth.

    This compares a extracted heartbeat list with a ground truth heartbeat list and returns True Positives,
    False Positives and True Negatives matches.
    The comparison is purely based on the start and end values of each heartbeat in the lists.
    Two heartbeat are considered a positive match, if both their start and their end values differ by less than the
    `threshold`.

    By default (controlled by the one-to-one parameter), if multiple heartbeats of the extracted heartbeat list would
    match to a single ground truth heartbeat (or vise-versa), only the heartbeat with the lowest distance is considered
    an actual match.
    If `one_to_one` is set to False, all matches would be considered True positives.
    This might lead to unexpected results in certain cases and should not be used to calculate traditional metrics like
    precision and recall.

    It is highly recommended to order the heartbeat lists and remove heartbeat with large overlaps before applying this
    method to get reliable results.

    Parameters
    ----------
    heartbeats_reference : :class:`~pandas.DataFrame`
        The ground truth heartbeat list.
    heartbeats_extracted : :class:`~pandas.DataFrame`
        The list of extracted heartbeats.
    sampling_rate_hz : int or float
        The sampling rate of the ECG signal in Hz.
    tolerance_ms : int or float, optional
        The allowed tolerance between labels in milliseconds.
        The comparison is done as `distance <= tolerance_ms`.
    one_to_one : bool, optional
        If True, only a single unique match per heartbeat is considered.
        In case of multiple matches, the one with the lowest distance is considered.
        If case of multiple matches with the same distance, the first match will be considered.
        If False, multiple matches are possible.
        If this is set to False, some calculated metrics from these matches might not be well defined!
    heartbeats_extracted_suffix : str, optional
        A suffix that will be appended to the index name of the extracted stride list in the output.
    heartbeats_reference_suffix : str, optional
        A suffix that will be appended to the index name of the ground truth in the output.

    Returns
    -------
    matches
        A 3 column dataframe with the column names `heartbeat_id{heartbeats_extracted_suffix}`,
        `heartbeat_id{heartbeats_reference_suffix}`, and `match_type`.
        Each row is a match containing the index value of the left and the right list, that belong together.
        The `match_type` column indicates the type of match.
        For all extracted heartbeats that have a match in the ground truth list, this will be "tp" (true positive).
        Extracted heartbeats that do not have a match will be mapped to a NaN and the match-type will be "fp" (false
        positives)
        All ground truth strides that do not have a extracted counterpart are marked as "fn" (false negative).
        In case MultiSensorStrideLists were used as inputs, a dictionary of such dataframes is returned.


    Examples
    --------
    >>> heartbeat_reference = pd.DataFrame(
    ...     [[10, 21], [20, 34], [31, 40]], columns=["start_sample", "end_sample"]
    ... ).rename_axis("heartbeat_id")
    >>> stride_list_seg = pd.DataFrame(
    ...     [[10, 20], [21, 30], [31, 40], [50, 60]], columns=["start_sample", "end_sample"]
    ... ).rename_axis("heartbeat_id")
    >>> matches = match_heartbeat_lists(
    ...     heartbeats_reference=heartbeats_reference,
    ...     heartbeats_extracted=heartbeats_extracted,
    ...     sampling_rate_hz=500,
    ...     tolerance_ms=10,
    ... )
    >>> matches
      heartbeat_id heartbeat_id_reference match_type
    0    0                 0         tp
    1    1               NaN         fp
    2    2                 2         tp
    3    3               NaN         fp
    4  NaN                 1         fn

    """
    tolerance_samples = int(tolerance_ms / 1000 * sampling_rate_hz)
    matches = _match_heartbeat_lists(
        heartbeats_extracted,
        heartbeats_reference,
        match_cols=["start_sample", "end_sample"],
        tolerance_samples=tolerance_samples,
        one_to_one=one_to_one,
        suffix_a=heartbeats_extracted_suffix,
        suffix_b=heartbeats_reference_suffix,
    )

    segmented_index_name = heartbeats_extracted.index.name + heartbeats_extracted_suffix
    reference_index_name = heartbeats_reference.index.name + heartbeats_reference_suffix

    tp_idx = ~matches.isna().any(axis=1)
    matches = matches.assign(match_type=pd.NA)
    matches.loc[tp_idx, "match_type"] = "tp"
    matches.loc[matches[reference_index_name].isna(), "match_type"] = "fp"
    matches.loc[matches[segmented_index_name].isna(), "match_type"] = "fn"

    return matches


def _match_heartbeat_lists(
    heartbeat_list_a: pd.DataFrame,
    heartbeat_list_b: pd.DataFrame,
    match_cols: str | Sequence[str],
    tolerance_samples: int | float = 0,
    one_to_one: bool = True,
    suffix_a: str = "_a",
    suffix_b: str = "_b",
) -> pd.DataFrame:
    if suffix_a == suffix_b:
        raise ValueError("The suffix for the first and the second heartbeat list must be different.")

    if tolerance_samples < 0:
        raise ValueError("The tolerance must be larger 0.")

    # sanitize input
    match_cols = [match_cols] if isinstance(match_cols, str) else list(match_cols)

    left_indices, right_indices = _match_label_lists(
        heartbeat_list_a[match_cols].to_numpy(),
        heartbeat_list_b[match_cols].to_numpy(),
        tolerance_samples=tolerance_samples,
        one_to_one=one_to_one,
    )

    index_name_a = heartbeat_list_a.index.name + suffix_a
    index_name_b = heartbeat_list_b.index.name + suffix_b

    matches_a = pd.DataFrame(index=heartbeat_list_a.index.copy(), columns=[index_name_b])
    matches_a.index.name = index_name_a

    matches_b = pd.DataFrame(index=heartbeat_list_b.index.copy(), columns=[index_name_a])
    matches_b.index.name = index_name_b

    heartbeat_list_a_idx = heartbeat_list_a.iloc[left_indices].index
    heartbeat_list_b_idx = heartbeat_list_b.iloc[right_indices].index

    matches_a.loc[heartbeat_list_a_idx, index_name_b] = heartbeat_list_b_idx
    matches_b.loc[heartbeat_list_b_idx, index_name_a] = heartbeat_list_a_idx

    matches_a = matches_a.reset_index()
    matches_b = matches_b.reset_index()

    matches = (
        pd.concat([matches_a, matches_b])
        .drop_duplicates()
        .sort_values([index_name_a, index_name_b])
        .reset_index(drop=True)
    )

    return matches


def _match_label_lists(
    list_a: np.ndarray, list_b: np.ndarray, tolerance_samples: int, one_to_one: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Find matches in two lists based on the distance between their vectors.

    Parameters
    ----------
    list_a : array with shape (n, d)
        An n long array of d-dimensional vectors
    list_b : array with shape (m, d)
        An m long array of d-dimensional vectors
    tolerance_samples
        Max allowed Chebyshev distance between matches.
        The comparison is done as "distance <= tolerance".
    one_to_one
        If True only valid one-to-one matches are returned (see more below)

    Returns
    -------
    indices_a
        Indices from list a that have a match in the list b.
        If `one_to_one` is False, indices might repeat.
    indices_b
        Indices from list b that have a match in list b.
        If `one_to_one` is False, indices might repeat.
        A valid match pare is then `(indices_b[i], indices_b[i]) for all i.

    Notes
    -----
    This function supports 2 modes:

    `one_to_one` = False:
        In this mode every match is returned as long the distance in all dimensions between the matches is at most
        tolerance.
        This is equivalent to the Chebyshev distance between the matches
        (aka `np.max(np.abs(match_a - match_b)) < tolerance`).
        This means multiple matches for each vector will be returned.
        This means the respective indices will occur multiple times in the output vectors.
    `one_to_one` = True:
        In this mode only a single match per index is allowed in both directions.
        This means that every index will only occur once in the output arrays.
        If multiple matches are possible based on the tolerance of the Chebyshev distance, the closest match will be
        selected based on the Manhattan distance (aka `np.sum(np.abs(match_a - match_b`).
        Only this match will be returned.
        This is done, because in case the input arrays are multi-dimensional, the Chebyshev distance is not really
        well suited for comparison.

    """
    if len(list_a) == 0 or len(list_b) == 0:
        return np.array([]), np.array([])

    tree_a = KDTree(list_b)
    tree_b = KDTree(list_a)

    if one_to_one is False:
        # p = np.inf is used to select the Chebyshev distance
        keys = list(
            zip(
                # We force sort the keys here to make sure the order is deterministic, even if the storage order of
                # sparse array might not be.
                *sorted(
                    tree_a.sparse_distance_matrix(tree_b, tolerance_samples, p=np.inf, output_type="dict").keys(),
                    key=lambda x: x[1],
                ),
                strict=False,
            )
        )
        # All values are returned that have a valid match
        return (np.array([]), np.array([])) if len(keys) == 0 else (np.array(keys[1]), np.array(keys[0]))

    # one_to_one is True
    # We calculate the closest neighbor based on the Manhattan distance in both directions and then find only the cases
    # were the right side closest neighbor resulted in the same pairing as side a closest neighbor ensuring
    # that we have true one-to-one-matches and we have already the closest match based on our final criteria.

    # p = 1 is used to select the Manhattan distance
    nearest_distance_a, nearest_neighbor_a = tree_a.query(list_a, p=1, workers=-1)
    _, nearest_neighbor_b = tree_b.query(list_b, p=1, workers=-1)

    # Filter the once that are true one-to-one matches
    indices_a = np.arange(len(list_a))
    combined_indices = np.vstack([indices_a, nearest_neighbor_a]).T
    boolean_map = nearest_neighbor_b[nearest_neighbor_a] == indices_a
    valid_matches = combined_indices[boolean_map]

    # Check if the remaining matches are inside our Chebyshev tolerance distance.
    # If not, delete them.
    valid_matches_distance = nearest_distance_a[boolean_map]
    # First we check if any of the Manhattan distances is larger than the threshold.
    # If not, all the Chebyshev distances are smaller than the threshold, too.
    index_large_matches = np.where(~(valid_matches_distance <= tolerance_samples))[0]
    if index_large_matches.size > 0:
        # Minkowski with p = np.inf uses the Chebyshev distance
        boolean_map = (
            minkowski_distance(list_a[index_large_matches], list_b[valid_matches[index_large_matches, 1]], p=np.inf)
            <= tolerance_samples
        )

        valid_matches = np.delete(valid_matches, index_large_matches[~boolean_map], axis=0)

    valid_matches = valid_matches.T

    return valid_matches[0], valid_matches[1]
