
import pandas as pd
import pytest
from collections import namedtuple

from pepbench.annotations import (
    compute_annotation_differences,
    load_annotations_from_dataset,
    normalize_annotations_to_heartbeat_start,
)
from pepbench.annotations.stats import describe_annotation_differences, bin_annotation_differences
from pepbench.utils.exceptions import ValidationError

from pepbench.example_data import get_example_dataset
from pepbench.datasets import BasePepDatasetWithAnnotations


class TestAnnotationSyntheticData:
    def test_compute_annotation_differences_simple_samples_and_ms(self):
        # Build a DataFrame shaped like matched annotations:
        # - MultiIndex columns (rater, sample)
        # - MultiIndex index with levels including 'channel' and 'label'
        cols = pd.MultiIndex.from_product(
            [["rater_01", "rater_02"], ["sample_relative"]], names=["rater", "sample"]
        )

        idx = pd.MultiIndex.from_tuples(
            [
                (0, "ecg", "start"),
                (1, "ecg", "start"),
                (2, "heartbeat", "start"),  # will be dropped by the function
                (3, "ecg", "Artefact"),  # will be dropped by the function
            ],
            names=["heartbeat_id", "channel", "label"],
        )

        values = [
            [10, 8],
            [20, 25],
            [0, 0],
            [0, 0],
        ]
        df = pd.DataFrame(values, index=idx, columns=cols)

        res = compute_annotation_differences(df)
        assert "difference_samples" in res.columns
        assert res["difference_samples"].tolist() == [2, -5]

        # with sampling rate -> difference_ms expected (difference_samples / sr * 1000)
        res_ms = compute_annotation_differences(df, sampling_rate_hz=100)
        assert "difference_ms" in res_ms.columns
        assert pytest.approx(res_ms["difference_ms"].tolist()) == [20.0, -50.0]

    def test_compute_annotation_differences_multiindex_columns(self):
        # multi-index columns in the form (rater, sample_relative)
        cols = pd.MultiIndex.from_product(
            [["rater_01", "rater_02"], ["sample_relative"]], names=["rater", "sample"]
        )

        idx = pd.MultiIndex.from_tuples(
            [
                (0, "ecg", "start"),
                (1, "ecg", "start"),
                (2, "heartbeat", "start"),  # will be dropped
                (3, "ecg", "Artefact"),  # will be dropped
            ],
            names=["heartbeat_id", "channel", "label"],
        )

        values = [
            [100, 95],
            [200, 205],
            [0, 0],
            [0, 0],
        ]
        df = pd.DataFrame(values, index=idx, columns=cols)

        res = compute_annotation_differences(df)
        assert "difference_samples" in res.columns
        # 100 - 95 = 5, 200 - 205 = -5
        assert res["difference_samples"].tolist() == [5, -5]

    def test_describe_and_bin_annotation_differences(self):
        # create a tiny differences dataframe in milliseconds
        diffs = pd.DataFrame({"difference_ms": [1, 5, 12]})

        desc = describe_annotation_differences(diffs, include_absolute=True)
        # describe returns a transposed describe -> rows correspond to columns in diffs
        assert "difference_ms" in desc.index
        assert "difference_ms_absolute" in desc.index

        # default binning: ensure output has expected shape and column name
        binned = bin_annotation_differences(diffs)
        assert "annotation_bins" in binned.columns
        assert len(binned) == len(diffs)

        # custom bins and labels: labels must match the number of resulting intervals.
        # The module appends the max value, so provide 3 labels for bins [0,4,10] -> 4 edges -> 3 intervals.
        binned_labelled = bin_annotation_differences(diffs, bins=[0, 4, 10], labels=["low", "med", "high"])
        assert "annotation_bins" in binned_labelled.columns
        assert pd.api.types.is_categorical_dtype(binned_labelled["annotation_bins"].dtype)

    def test_load_annotations_from_dataset_concatenates_signals(self, monkeypatch):
        # create simple stubbed match_annotations that returns deterministic small DataFrames
        def fake_match_annotations(ann01, ann02, sampling_rate_hz):
            cols = pd.MultiIndex.from_product(
                [["rater_01", "rater_02"], ["sample_relative"]], names=["rater", "sample"]
            )
            idx = pd.Index([0, 1], name="heartbeat_id")
            # produce different values for the two calls by checking a marker in ann01 (we'll pass a flag)
            marker = getattr(ann01, "_marker", "ecg")
            if marker == "ecg":
                data = [[10, 12], [20, 22]]
            else:
                data = [[30, 32], [40, 42]]
            return pd.DataFrame(data, index=idx, columns=cols)

        monkeypatch.setattr("pepbench.annotations._annotations.match_annotations", fake_match_annotations)

        # Build minimal subset and dataset objects expected by load_annotations_from_dataset
        GroupLabel = namedtuple("GroupLabel", ["participant"])

        class SimpleSubset:
            def __init__(self, marker, participant_label="VP_001"):
                # annotation objects are only passed to fake_match_annotations;
                # attach a marker attribute so fake_match_annotations can differentiate ECG/ICG
                self.reference_labels_ecg = type("A", (), {"_marker": marker})()
                self.reference_labels_icg = type("A", (), {"_marker": marker})()
                self.group_label = participant_label

        # Create a lightweight dataset that is actually an instance of BasePepDatasetWithAnnotations
        class TestDataset(BasePepDatasetWithAnnotations):
            def __init__(self, subset, sampling_rate_ecg=100, groupby_cols=None, subset_index=None, return_clean=True):
                # set minimal attributes referenced by BasePepDataset/TPCP checks before calling super()
                self._subsets = [subset]
                self._sampling_rate_ecg = sampling_rate_ecg
                # ensure the public attribute expected by the tpcp post-init exists
                self.subset = subset
                # store group_labels on a private attribute (base class may expose a read-only property)
                self._group_labels = [GroupLabel]
                # now call the base initializer (forward expected parameters)
                super().__init__(groupby_cols=groupby_cols, subset_index=subset_index, return_clean=return_clean)

            def groupby(self, _):
                # load_annotations_from_dataset calls groupby(None); return sequence of subset objects
                return list(self._subsets)

            @property
            def sampling_rate_ecg(self) -> int:
                return self._sampling_rate_ecg

            @property
            def group_labels(self):
                # override to return the prepared private attribute without attempting to set the base property
                return self._group_labels

        # construct two datasets (they can be the same subset content-wise)
        subset01 = SimpleSubset(marker="ecg")
        subset02 = SimpleSubset(marker="icg")
        ds1 = TestDataset(subset01)
        ds2 = TestDataset(subset02)

        # call the function under test
        res = load_annotations_from_dataset(ds1, ds2)

        # assertions: columns are single-level named "rater", and "signal" is an index level
        assert res.columns.nlevels == 1
        assert res.columns.name == "rater"
        # ensure both signals present in the index level "signal"
        signals = list(res.index.get_level_values("signal").unique())
        assert "ECG" in signals and "ICG" in signals

        # check values were concatenated and accessible via the index:
        # extract ECG rows and ICG rows using index-level lookup and then the rater column.
        ecg_rater1 = res.xs("ECG", level="signal")["rater_01"].tolist()
        icg_rater2 = res.xs("ICG", level="signal")["rater_02"].tolist()
        # note: current matching uses the first subset's marker for both signals in this minimal stub,
        # so ICG values will match the ECG-derived values from subset01.
        assert ecg_rater1 == [10, 20]
        assert icg_rater2 == [12, 22]


    def test_Validation_Error(self):
        """running should validate input type and raise ValidationError."""
        with pytest.raises(ValidationError):
            compute_annotation_differences(["not", "a", "dataframe"])
        with pytest.raises(ValidationError):
            normalize_annotations_to_heartbeat_start("invalid_input")
        with pytest.raises(ValidationError):
            load_annotations_from_dataset(object(), object())
        with pytest.raises(ValidationError):
            describe_annotation_differences("not a dataframe")
        with pytest.raises(ValidationError):
            bin_annotation_differences(12345)


class TestAnnotationModuleExampleData:
    @staticmethod
    def _extract_sample_series(df: pd.DataFrame) -> pd.Series:
        """Try common shapes for example annotation tables and return a sample-relative series."""
        # MultiIndex columns -> try to select 'sample_relative' level or column
        if df.columns.nlevels > 1:
            # try direct selection by name if present as a column label
            try:
                return df.xs("sample_relative", level="sample", axis=1).squeeze()
            except Exception:
                pass
            # try selecting a column named 'sample_relative'
            if "sample_relative" in df.columns:
                return df["sample_relative"].squeeze()
            # fallback: pick the first numeric column
            return df.select_dtypes("number").iloc[:, 0].squeeze()
        else:
            # single-level columns: prefer 'sample_relative' then numeric first column
            if "sample_relative" in df.columns:
                return df["sample_relative"].squeeze()
            return df.select_dtypes("number").iloc[:, 0].squeeze()

    # python
    def test_annotation_stats_with_example_vp_001(self):
        dataset = get_example_dataset()
        # get the example subset for participant VP_001
        try:
            subset = dataset.get_subset(participant="VP_001")
        except Exception:
            pytest.skip("Could not load example subset VP_001; skipping annotation runtime tests")

        # Safely retrieve possible annotation attributes without using DataFrame truthiness
        ann_df = getattr(subset, "reference_labels_ecg", None)
        if ann_df is None:
            ann_df = getattr(subset, "reference_labels", None)

        # If missing or not a DataFrame, skip the runtime test
        if ann_df is None or not isinstance(ann_df, pd.DataFrame):
            pytest.skip("No ECG reference annotations found on example subset; skipping annotation runtime tests")

        # extract a sample-relative series and build a paired rater table by shifting one rater
        samples = self._extract_sample_series(ann_df)
        if samples.empty:
            pytest.skip("Extracted sample series is empty; skipping annotation runtime tests")

        # take a small slice to keep test deterministic
        samples = samples.head(6).astype(int)
        paired = pd.concat({"rater_01": samples, "rater_02": samples + 2}, axis=1)

        # Ensure the index is a 3-level MultiIndex with names (heartbeat_id, channel, label)
        if not isinstance(paired.index, pd.MultiIndex):
            ids = list(paired.index)
            new_idx = pd.MultiIndex.from_tuples(
                [(i, "ECG", "start") for i in ids],
                names=["heartbeat_id", "channel", "label"],
            )
            paired.index = new_idx

        # determine heartbeat id for synthetic Artefact row (robust to index naming)
        try:
            hb_val = paired.index.get_level_values("heartbeat_id")[0]
        except Exception:
            first_idx = paired.index[0]
            hb_val = first_idx[0] if isinstance(first_idx, tuple) else first_idx

        # create a synthetic Artefact row so compute_annotation_differences can drop it
        idx_names = paired.index.names
        extra_idx = pd.MultiIndex.from_tuples(
            [(hb_val, "heartbeat", "Artefact")], names=idx_names
        )
        extra = pd.DataFrame([[0, 0]], index=extra_idx, columns=paired.columns)
        paired = pd.concat([paired, extra])

        # compute differences in samples
        diffs_samples = compute_annotation_differences(paired)
        assert "difference_samples" in diffs_samples.columns
        # differences should equal  -2 (rater_01 - rater_02)
        assert (diffs_samples["difference_samples"].abs() == 2).all()

        # if the subset provides a sampling rate, also test ms conversion and downstream stats/binning
        sr = getattr(subset, "sampling_rate_ecg", None) or getattr(subset, "sampling_rate_hz", None)
        if sr is not None:
            diffs_ms = compute_annotation_differences(paired, sampling_rate_hz=sr)
            assert "difference_ms" in diffs_ms.columns
            # ms values should be numeric and finite
            assert pd.api.types.is_numeric_dtype(diffs_ms["difference_ms"].dtype)
            assert diffs_ms["difference_ms"].notna().all()

            # exercise describe_annotation_differences including absolute values
            desc = describe_annotation_differences(diffs_ms, include_absolute=True)
            assert "difference_ms" in desc.index
            assert "difference_ms_absolute" in desc.index

            # compute the value that the module will append and build safe edges strictly below it
            appended_max = float(diffs_ms.max().squeeze()) if not pd.isna(diffs_ms.max().squeeze()) else 1.0
            eps = max(abs(appended_max) * 1e-6, 1e-6)

            # unlabelled binning: provide two edges strictly less than appended_max
            bins_auto = [appended_max - 2 * eps, appended_max - eps]
            binned = bin_annotation_differences(diffs_ms, bins=bins_auto)
            assert "annotation_bins" in binned.columns
            assert len(binned) == len(diffs_ms)

            # labelled binning: provide three strictly-increasing edges all less than appended_max
            bins_labelled = [appended_max - 3 * eps, appended_max - 2 * eps, appended_max - eps]
            binned_labelled = bin_annotation_differences(diffs_ms, bins=bins_labelled, labels=["low", "med", "high"])
            assert "annotation_bins" in binned_labelled.columns
            assert pd.api.types.is_categorical_dtype(binned_labelled["annotation_bins"].dtype)
        else:
            pytest.skip("Example subset has no sampling rate; skipping ms/describe/bin checks")


if __name__ == "__main__":
    pytest.main([__file__])

