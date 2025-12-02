
import pandas as pd
import numpy as np
import pytest

from pepbench.data_handling._data_handling import (
    get_reference_data,
    get_reference_pep,
    get_data_for_algo,
    get_pep_for_algo,
    describe_pep_values,
    rr_interval_to_heart_rate,
    add_unique_id_to_results_dataframe,
    merge_result_metrics_from_multiple_annotators,
    merge_results_per_sample_from_different_annotators,
    get_error_by_group,
    correlation_reference_pep_heart_rate,
    get_performance_metric,
)


@pytest.fixture
def sample_results_per_sample():
    # index: algorithm levels + participant + heartbeat
    index = pd.MultiIndex.from_tuples(
        [
            ("q1", "b1", "out1", "subj1", 0),
            ("q1", "b1", "out1", "subj1", 1),
            ("q1", "b1", "out2", "subj2", 0),
            ("q2", "b2", "out1", "subj2", 0),
        ],
        names=["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm", "participant", "heartbeat"],
    )

    cols = pd.MultiIndex.from_tuples(
        [
            ("pep_ms", "reference"),
            ("pep_ms", "estimated"),
            ("rr_interval_ms", "reference"),
            ("absolute_error_per_sample_ms", "algo"),
        ],
        names=[None, None],
    )

    df = pd.DataFrame(
        [
            [120.0, 125.0, 800.0, 5.0],
            [121.0, 119.0, 810.0, 2.0],
            [115.0, 118.0, 790.0, 3.0],
            [130.0, 128.0, 770.0, 2.5],
        ],
        index=index,
        columns=cols,
    )
    return df


def test_get_reference_data_and_pep(sample_results_per_sample):
    ref = get_reference_data(sample_results_per_sample)
    # reference extraction drops algorithm levels and returns frame with reference columns
    assert "pep_ms" in ref.columns
    pep = get_reference_pep(sample_results_per_sample)
    # pep is a DataFrame with pep_ms column
    assert list(pep.columns) == ["pep_ms"]
    # current implementation returns the first algorithm group only -> check those values
    assert pep.iloc[0, 0] == 120.0
    assert pep.iloc[1, 0] == 121.0


def test_get_data_for_algo_and_get_pep_for_algo(sample_results_per_sample):
    # select by a full algorithm tuple (current implementation expects the full combo)
    data_q1 = get_data_for_algo(sample_results_per_sample, ("q1", "b1", "out1"))
    # xs drops the selected algorithm index levels, so remaining index names should be participant and heartbeat
    assert data_q1.index.names[-2:] == ["participant", "heartbeat"]
    # should return the two rows that match ("q1","b1","out1")
    assert data_q1.shape[0] == 2
    # verify specific values for the selected combination
    assert data_q1.loc[("subj1", 0)][("pep_ms", "estimated")] == 125.0
    assert data_q1.loc[("subj1", 1)][("pep_ms", "estimated")] == 119.0

    # get_pep_for_algo expects full algo combination; pass tuple to select full combo
    pep_q1b1 = get_pep_for_algo(sample_results_per_sample, ("q1", "b1", "out1"))
    # pep_q1b1 should be a DataFrame/Series with pep values (estimated)
    assert "pep_ms" in getattr(pep_q1b1, "columns", []) or getattr(pep_q1b1, "name", "") == "pep_ms"


def test_describe_pep_values():
    df = pd.DataFrame({"phase": ["A", "A", "B", "B"], "pep_ms": [100.0, 110.0, 120.0, 140.0]})
    desc = describe_pep_values(df, group_cols="phase", metrics=["mean", "std"])
    # should have grouping rows for metrics and columns for pep_ms
    assert ("pep_ms", "mean") in desc.columns
    assert ("pep_ms", "std") in desc.columns
    # check a numeric value
    assert np.isclose(desc[("pep_ms", "mean")].loc["A"], 105.0)


def test_rr_interval_to_heart_rate():
    df = pd.DataFrame({"rr_interval_ms": [1000.0, 500.0, 666.6666667]})
    out = rr_interval_to_heart_rate(df)
    # hr = 60 * 1000 / rr_interval_ms
    assert "heart_rate_bpm" in out.columns
    assert np.isclose(out["heart_rate_bpm"].iloc[0], 60.0)
    assert np.isclose(out["heart_rate_bpm"].iloc[1], 120.0)


def test_add_unique_id_to_results_dataframe(sample_results_per_sample):
    res = add_unique_id_to_results_dataframe(sample_results_per_sample)
    # resulting index should include id_concat
    assert "id_concat" in res.index.names
    # id_concat values should be strings
    concat_vals = res.index.get_level_values("id_concat")
    assert all(isinstance(x, str) for x in concat_vals)


def test_merge_result_metrics_and_per_sample_merge():
    # create two simple metrics tables for annotators
    a1 = pd.DataFrame({"Mean Absolute Error [ms]": [5.0], "Mean Error [ms]": [0.5]}, index=["algoA"])
    a2 = pd.DataFrame({"Mean Absolute Error [ms]": [6.0], "Mean Error [ms]": [0.0]}, index=["algoA"])
    # do not request annotation difference here because the simple fixture does not match the grouping assumptions
    merged = merge_result_metrics_from_multiple_annotators([a1, a2], add_annotation_difference=False)
    # merged should contain both annotators as top-level columns
    assert any("Annotator" in str(lv) for lv in merged.columns.get_level_values(0))
    # per-sample: create two small per-sample frames and merge
    idx = pd.MultiIndex.from_tuples([("q1", "b1", "out1", "subj1", 0)], names=["q_peak_algorithm", "b_point_algorithm", "outlier_correction_algorithm", "participant", "heartbeat"])
    df1 = pd.DataFrame([[120.0]], index=idx, columns=pd.MultiIndex.from_tuples([("pep_ms", "estimated")]))
    df2 = pd.DataFrame([[118.0]], index=idx, columns=pd.MultiIndex.from_tuples([("pep_ms", "estimated")]))
    combined = merge_results_per_sample_from_different_annotators([df1, df2])
    # combined should have top-level annotator labels on columns
    assert "Annotator 1" in combined.columns.get_level_values(0)
    assert "Annotator 2" in combined.columns.get_level_values(0)


def test_get_error_by_group(sample_results_per_sample):
    # use absolute_error_per_sample_ms column
    err = get_error_by_group(sample_results_per_sample, error_metric="absolute_error_per_sample_ms", grouper="participant")
    # should have 'metric' level in columns and mean/std for groups
    assert "metric" in err.columns.names
    # rows should correspond to participants present
    assert "subj1" in err.index.get_level_values("participant") or "subj2" in err.index.get_level_values("participant")


def test_get_performance_metric_and_compute_performance_like(sample_results_per_sample):
    # create simple metric columns with last level and test droplevel behavior
    metric = get_performance_metric(sample_results_per_sample, "absolute_error_per_sample_ms")
    # droplevel removed last column level -> resulting columns should be single-level
    assert metric.columns.nlevels == 1 or list(metric.columns) == ["absolute_error_per_sample_ms"]


def test_correlation_reference_pep_heart_rate_monkeypatched(monkeypatch, sample_results_per_sample):
    # ensure reference rr -> heart rate column exists
    # create heart rate column inside reference slice: we already have rr_interval_ms reference
    df = sample_results_per_sample.copy()
    # compute heart rate column for reference
    hr = 60 * 1000 / df[("rr_interval_ms", "reference")]
    df[("heart_rate_bpm", "reference")] = hr

    # monkeypatch pingouin functions used by the module
    class FakePG:
        @staticmethod
        def linear_regression(X, y, remove_na=True):
            # return a tiny DataFrame like pingouin would
            return pd.DataFrame({"beta": [0.1], "se": [0.01]})

        @staticmethod
        def corr(x, y, method="pearson"):
            # return a fake correlation DataFrame
            return pd.DataFrame({"r": [0.5], "p-val": [0.05]})

    # patch the module's pg object
    import pepbench.data_handling._data_handling as dh

    monkeypatch.setattr(dh, "pg", FakePG)

    res = correlation_reference_pep_heart_rate(df)
    assert "linear_regression" in res and "correlation" in res
    # check types
    assert isinstance(res["linear_regression"], pd.DataFrame)
    assert isinstance(res["correlation"], pd.DataFrame)

if __name__ == "__main__":
    pytest.main([__file__])
