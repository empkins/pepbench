Common Datatypes
=================

pepbench tries to stick to common data-containers - namely :class:`~numpy.ndarray`, :class:`~pandas.DataFrame`, :py:class:`dict` and :class:`~pandas.Series` - to store all in- and outputs of the used algorithm. However, based on the above mentioned containers, a set of certain data-types are defined and used throughout the library.
This makes it easy for users to handle complex problems and makes it possible to perform sanity checks that prevent common issues.
The following explains these data-structures in details to ease to process of preparing your data for the use of pepbench and help to understand the outputs.

Units
------
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
~~~~~~~~~~~~~~~~~~
Signal amplitude units (for ECG/ICG traces) are dataset-dependent and are not enforced by pepbench; common dataset units are volts (V) or millivolts (mV). pepbench algorithms expect that the amplitude unit is consistent within a dataset.

Naming Conventions in pepbench
--------------------------------
The codebase uses a few naming conventions / column suffixes to indicate units and facilitate automatic dtype coercion:

- Columns/suffixes ending with ``_ms``: values expressed in milliseconds (ms). Examples from the codebase: ``pep_ms``, ``rr_interval_ms``, ``error_per_sample_ms``.
- Columns/suffixes ending with ``_percent`` or represented with ``%`` in labels: percentage values (%%). Examples: ``absolute_relative_error_per_sample_percent``.
- ``heart_rate_bpm``: heart rate values given in beats per minute (bpm).
- ``_data`` / sample indices: many functions operate on sample indices (integer counts). When plotting or when requested, pepbench can convert time-like indexes to seconds (s).

Start-End Indices
-------------------
Many pepbench tables and functions (for example heartbeat tables and annotation loaders) represent time ranges using sample indices named ``start_sample`` and ``end_sample``. pepbench follows the common Python slicing convention: the start index is inclusive and the end index is exclusive, i.e. the interval represented is [start, end).

Practical consequences and examples from the codebase:

- To obtain the last sample inside a region the code frequently uses ``end_sample - 1``. See ``pepbench.plotting._utils._get_heartbeat_borders`` which maps ``end_sample`` to the final index with ``end_sample - 1``.
- Durations in samples are obtained via ``end - start`` (for example heartbeat durations or when shifting indices to a zero origin).
- Edge cases:
  - A region that starts at the first sample of a recording has ``start = 0``.
  - A region that includes the last sample of a recording uses ``end = len(dataset)``.
  - Adjacent regions share boundaries: the ``end`` of the first region equals the ``start`` of the next region.

Datasets
----------

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
:class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDataset` (or
:class:`~pepbench.datasets._base_pep_extraction_dataset.BasePepDatasetWithAnnotations` when reference labels are required).
This simplifies many tasks and provides a clean abstraction for your data.