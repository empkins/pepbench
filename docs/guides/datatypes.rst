Common Datatypes
=================

pepbench tries to stick to common data-containers - namely :class:`~numpy.ndarray`, :class:`~pandas.DataFrame`, :py:class:`dict` and :class:`~pandas.Series` - to store all in- and outputs of the used algorithm. However, based on the above mentioned containers, a set of certain data-types are defined and used throughout the library.
This makes it easy for users to handle complex problems and makes it possible to perform sanity checks that prevent common issues.
The following explains these data-structures in details to ease to process of preparing your data for the use of pepbench and help to understand the outputs.

Units
-----
.. _units:

Before talking about data-types the physical units for all values stored in these data-types should be clear.
The following table provides an overview over the units commonly used in the pepbench package and what they refer to.

.. table:: Common Units in pepbench

   ==============================  ======================
   Value                           Unit
   ==============================  ======================
   Time (seconds)                  s
   Time (milliseconds)             ms
   Sampling rate / frequency       Hz
   Heart rate                      bpm
   Relative / percentage values    %
   Sample indices / counts         samples
   ==============================  ======================

Signal Amplitudes
-----------------
Signal amplitude units (for ECG/ICG traces) are dataset-dependent and are not enforced by pepbench; common dataset units are volts (V) or millivolts (mV). pepbench algorithms expect that the amplitude unit is consistent within a dataset.

Naming Conventions in pepbench
------------------------------
The codebase uses a few naming conventions / column suffixes to indicate units and facilitate automatic dtype coercion:

- Columns/suffixes ending with ``_ms``: values expressed in milliseconds (ms). Examples from the codebase: ``pep_ms``, ``rr_interval_ms``, ``error_per_sample_ms``.
- Columns/suffixes ending with ``_percent`` or represented with ``%`` in labels: percentage values (%%). Examples: ``absolute_relative_error_per_sample_percent``.
- ``heart_rate_bpm``: heart rate values given in beats per minute (bpm).
- ``_data`` / sample indices: many functions operate on sample indices (integer counts). When plotting or when requested, pepbench can convert time-like indexes to seconds (s).

Start-End Indices
-----------------
Many pepbench tables and functions (for example heartbeat tables and annotation loaders) represent time ranges using sample indices named ``start_sample`` and ``end_sample``. pepbench follows the common Python slicing convention: the start index is inclusive and the end index is exclusive, i.e. the interval represented is [start, end).

Practical consequences and examples from the codebase:

- To obtain the last sample inside a region the code frequently uses ``end_sample - 1``. See ``pepbench.plotting._utils._get_heartbeat_borders`` which maps ``end_sample`` to the final index with ``end_sample - 1``.
- Durations in samples are obtained via ``end - start`` (for example heartbeat durations or when shifting indices to a zero origin).
- Edge cases:
  - A region that starts at the first sample of a recording has ``start = 0``.
  - A region that includes the last sample of a recording uses ``end = len(dataset)``.
  - Adjacent regions share boundaries: the ``end`` of the first region equals the ``start`` of the next region.

Heartbeat Lists
---------------
Heartbeats are represented as :class:`~pandas.DataFrame` with one row per beat and columns for the start and end
sample indices, as well as the R-peak sample index. A well-defined heartbeat list makes it easy to align ECG/ICG
segments, run extraction algorithms and evaluate results.

A *SingleSensorHeartbeatList* is a plain :class:`pandas.DataFrame` that should at least contain the columns
``start_sample`` and ``end_sample``. In many cases a ``r_peak_sample`` column is present as well (the detected R-peak
within the heartbeat). The index is expected to have one level with the name ``heartbeat_id``. If you prefer to keep
``heartbeat_id`` as a column instead, convert it to the index with ``df = df.set_index("heartbeat_id")`` before
passing it into functions that expect the index.

All sample-based columns are expressed in samples relative to the start of the recording (not relative to the start of
each heartbeat). Durations can be obtained using the sampling rate (``fs``) and converted to seconds or milliseconds.

Required/Recommended columns and units
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``start_sample`` (int): inclusive start index (samples) of the heartbeat in the recording.
- ``end_sample`` (int): exclusive end index (samples) of the heartbeat in the recording (pepbench uses half-open
  intervals [start, end)).
- ``r_peak_sample`` (int, optional but recommended): sample index of the R-peak. If present it should satisfy
  ``start_sample <= r_peak_sample < end_sample``.

Recommended additional/derived columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- ``duration_samples`` = ``end_sample - start_sample`` (int)
- ``duration_ms`` = ``duration_samples / fs * 1000`` (float)
- ``r_peak_offset`` = ``r_peak_sample - start_sample`` (int)
- ``quality_score`` (float), ``label`` (str) or ``source_channel`` (str) for metadata or curation

Index and format conventions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Index name: ``heartbeat_id`` is the canonical index name used in examples and several internal functions. Many
  internal examples and helpers assume the index has this name (see :func:`~pepbench.heartbeat_matching.match_heartbeat_lists`).
- Columns always refer to absolute sample indices in the recording (not time or per-beat offsets). Use the sampling
  rate to convert to seconds if needed.

Invariants and validation rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Rows should be sorted by ``start_sample`` (increasing).
- ``start_sample >= 0`` and ``end_sample > start_sample`` (no negative or zero-length regions unless explicitly
  documented).
- If ``r_peak_sample`` is present, it must satisfy ``start_sample <= r_peak_sample < end_sample``.
- Adjacent beats are allowed: the ``end_sample`` of one beat can equal the ``start_sample`` of the next beat.
- Overlapping beats should either be resolved (merge/remove) or annotated (for example with ``quality_score``).


Common operations and examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A typical heartbeat list example (assume ``fs = 1000`` Hz):

>>> import pandas as pd
>>> df = pd.DataFrame(
...     [[0, 300, 150], [300, 620, 455], [620, 930, 750]],
...     columns=["start_sample", "end_sample", "r_peak_sample"],
... )
>>> df.index.name = "heartbeat_id"
>>> df
           start_sample  end_sample  r_peak_sample
heartbeat_id
0                   0         300            150
1                 300         620            455
2                 620         930            750

Compute derived columns:

>>> df["duration_samples"] = df["end_sample"] - df["start_sample"]
>>> df["duration_ms"] = df["duration_samples"] / 1000 * 1000  # = duration_samples for fs=1000

Filtering short beats:

>>> min_samples = int(0.2 * 1000)  # 200 ms
>>> df_filtered = df[df["duration_samples"] >= min_samples]

Integration with algorithms and helpers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Heartbeat segmentation algorithms used in pepbench (for example :class:`~biopsykit.signals.ecg.segmentation.HeartbeatSegmentationNeurokit`) provide a ``heartbeat_list_`` attribute that already follows the sample-index convention used here.
- Annotation loaders / dataset helpers expose reference heartbeats in the format expected by algorithms. The helper
  :func:`~pepbench.datasets._helper.compute_reference_heartbeats` reformats annotation tables (dropping channel-level,
  renaming columns to ``*_sample``) to a heartbeat table suitable for matching and evaluation.
- To evaluate and match two heartbeat lists use :func:`~pepbench.heartbeat_matching.match_heartbeat_lists`. This function
  compares start/end borders (in samples) and returns true/false positive/negative matches; it assumes the column
  names ``start_sample`` and ``end_sample`` and an index named ``heartbeat_id`` (see its docstring for examples).

Edge cases and recommendations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Missing ``r_peak_sample``: allow NaN when segmentation algorithms fail to detect a clear R-peak — downstream steps that
  require the R-peak should handle or skip those beats explicitly.
- Beats at recording boundaries: ``start_sample == 0`` and ``end_sample == len(recording)`` are valid and indicate
  coverage to the recording edges.
- Overlaps and duplicates: prefer to resolve these during preprocessing. When storing heartbeat lists on disk prefer
  parquet to preserve dtypes.


Datasets
--------

Compared to the low level datatypes, datasets are higher level abstractions, containing all data and metadata
associated with a set of recordings.
They are based on the :class:`~tpcp.Dataset` class and allow to easily load and access otherwise complex data
structures.

A dataset that has only one "row" (i.e. one recording) is referred to as a "datapoint" and is the expected input for
all the pipelines in pepbench.

A dataset that contains exactly one recording (a single row in the :class:`tpcp.Dataset` sense) is referred to as a
"datapoint". This is the expected input for the package's extraction pipelines — typically you should pass a
:class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDataset` instance for extraction pipelines, or a
:class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDatasetWithAnnotations` instance when reference labels
(heartbeats/PEP) are required for evaluation.

Using this dataset abstraction allows us to easily apply the same algorithms to different datasets and to use
higher-level tpcp-features like the :func:`~tpcp.validate.cross_validate` to run and evaluate our pipelines on
subsets of our datasets in a consistent manner.

The simplest dataset that we provide out of the box is the :class:`~pepbench.datasets.ExampleDataset`, which can be
used to load the example data that we provide with pepbench.

If you have already loaded your own data and want to use it with a pepbench pipeline, you can use the
:class:`~pepbench.datasets.WrapperDataset` class to quickly create a compatible dataset from your data.
For long-term use and clearer integration we highly encourage creating a custom dataset class that subclasses
:class:`~pepbench.datasets.BasePepDataset` (or
:class:`~pepbench.datasets.BasePepDatasetWithAnnotations` when reference labels are required).
This simplifies many tasks and provides a clean abstraction for your data.
