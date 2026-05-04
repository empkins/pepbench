.. _user_guide_datasets:

Datasets
========

pepbench uses `tpcp <https://pypi.org/project/tpcp/>`_ dataset classes.
Each datapoint provides ECG/ICG signals plus metadata required by pipelines
and (optionally) evaluation.

Available dataset classes
-------------------------

pepbench currently provides:

* :class:`~pepbench.datasets.EmpkinsDataset`
* :class:`~pepbench.datasets.GuardianDataset`
* :class:`~pepbench.datasets.ExampleDataset` (small demo/test dataset)

All dataset implementations follow the same core interface based on
:class:`~pepbench.datasets.BasePepDataset` and
:class:`~pepbench.datasets.BasePepDatasetWithAnnotations`.

Core interface requirements
---------------------------

All PEP extraction pipelines expect:

* :attr:`~pepbench.datasets.BasePepDataset.ecg` – ECG signal as a pandas DataFrame
* :attr:`~pepbench.datasets.BasePepDataset.icg` – ICG signal as a pandas DataFrame
* :attr:`~pepbench.datasets.BasePepDataset.sampling_rate_ecg` – ECG sampling rate (Hz)
* :attr:`~pepbench.datasets.BasePepDataset.sampling_rate_icg` – ICG sampling rate (Hz)
* :attr:`~pepbench.datasets.BasePepDataset.heartbeats` – segmented heartbeats (start, end, R-peak per beat)

For evaluation workflows, datasets should additionally expose:

* :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_pep` – reference PEP values (per sample or per beat)
* :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_heartbeats` – reference heartbeat segmentation
* :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_labels_ecg` / :attr:`~pepbench.datasets.BasePepDatasetWithAnnotations.reference_labels_icg` – label annotations
* Labeled sections of the continuous signal (dataset-specific; attribute name may vary, e.g. ``labeled_segments`` — DataFrame with ``start``, ``end``, ``label``; ``annotation_intervals`` — list of ``(start, end, label)`` tuples; ``label_mask`` — boolean Series aligned to the signal)

EmpkinsDataset
----------------------------

:class:`~pepbench.datasets.EmpkinsDataset` is the primary dataset class for
EmpkinS recordings and annotations.

Typical usage:

.. code-block:: python

   from pepbench.datasets import EmpkinsDataset

   ds = EmpkinsDataset(
       base_path="/path/to/empkins/root",
       only_labeled=True,
       exclude_missing_data=True,
       label_type="rater_01",  # or "rater_02" / "average"
   )

   ds.create_index()  # optional; index is created lazily on first access
   print(len(ds))

Use datapoints exactly like any other tpcp dataset:

.. code-block:: python

   dp = next(iter(ds))
   ecg, icg = dp.ecg, dp.icg
   fs_ecg, fs_icg = dp.sampling_rate_ecg, dp.sampling_rate_icg
   heartbeats = dp.heartbeats

For full attribute and parameter details, see the API reference:
:class:`~pepbench.datasets.EmpkinsDataset`.

GuardianDataset
---------------

:class:`~pepbench.datasets.GuardianDataset` follows the same interface as
:class:`~pepbench.datasets.EmpkinsDataset` and can be used with the same
pipelines/evaluation code.

The practical difference is in study design and signal characteristics
(e.g., protocol and sampling specifics), not in the programming interface.

Example Dataset
----------------
Pepbench also provides a small :class:`~pepbench.datasets.ExampleDataset` for testing and demonstration purposes.
It contains two patients' ECG/ICG signals with known PEP values and annotations, allowing you to quickly test pipelines without needing access to the full Empkins or Guardian datasets.

* Hands-on notebook: :download:`Example Dataset <../examples/_notebooks/Example_Dataset.ipynb>`

Integration of Own Data
--------------------------

If you already have ECG and ICG signals loaded in memory and want to use them with pepbench pipelines
without creating a full custom dataset class, the :class:`~pepbench.datasets.WrapperDataset` is the ideal
solution. It wraps your raw signals into a compatible dataset format that works seamlessly with all
pepbench pipelines and evaluation tools.

Quick Start with WrapperDataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~pepbench.datasets.WrapperDataset` requires only your signal data and their sampling rates:

.. code-block:: python

   import pandas as pd
   from biopsykit.utils.dtypes import EcgRawDataFrame, IcgRawDataFrame
   from pepbench.datasets import WrapperDataset

   # Assume you have loaded your ECG and ICG data
   # They should be pandas DataFrames with appropriate structure for BiopsyKit
   ecg_data = pd.read_csv("path/to/ecg_data.csv", index_col=0)
   icg_data = pd.read_csv("path/to/icg_data.csv", index_col=0)

   # Ensure proper BiopsyKit dtypes (EcgRawDataFrame, IcgRawDataFrame)
   # These are specialized pandas DataFrames with specific metadata
   ecg = EcgRawDataFrame(ecg_data)
   icg = IcgRawDataFrame(icg_data)

   # Create the wrapper dataset with your data
   ds = WrapperDataset(
       ecg=ecg,
       icg=icg,
       sampling_rate_ecg=500,  # ECG sampling rate in Hz
       sampling_rate_icg=500,  # ICG sampling rate in Hz
   )

   # Compatible with regular pipelines
   from pepbench.pipelines import PepExtractionPipeline

   PepExtractionPipeline().run(ds)

Use :class:`~pepbench.datasets.WrapperDataset` when you want quick integration
without implementing indexing/grouping over many files.

.. _user_guide_creating_custom_datasets:
Create your Own Dataset Class
------------------------------
If you have a larger collection of ECG/ICG recordings or want to integrate with pepbench's indexing and grouping features, it's best to create a custom dataset class by subclassing :class:`~pepbench.datasets.BasePepDataset` or :class:`~pepbench.datasets.BasePepDatasetWithAnnotations`.
This allows you to implement the required properties and methods while leveraging the full power of the tpcp framework for indexing, grouping, and iteration.

Why Create a Custom Dataset?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a custom dataset class is recommended when:

* You have a structured collection of ECG/ICG recordings across multiple subjects/sessions
* You need to support filtering, grouping, or subsetting operations on your data
* You want to integrate with pepbench's evaluation framework
* You have complex data loading logic that depends on file structure or metadata
* You want to provide a reusable interface for your data that works with all pepbench pipelines

Step-by-Step Implementation Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. Choose Your Base Class**

Decide which base class to inherit from based on your needs:

* Use :class:`~pepbench.datasets.BasePepDataset` if you only need to run pipelines on your data
* Use :class:`~pepbench.datasets.BasePepDatasetWithAnnotations` if you also have reference annotations for evaluation

**2. Implement Required Methods and Properties**

All custom datasets must implement:

* ``create_index()`` - Returns a pandas DataFrame defining all datapoints in your dataset
* ``ecg`` property - Returns ECG signal for the current subset
* ``icg`` property - Returns ICG signal for the current subset
* ``sampling_rate_ecg`` property - Returns ECG sampling rate in Hz
* ``sampling_rate_icg`` property - Returns ICG sampling rate in Hz
* ``heartbeats`` property - Returns heartbeat segmentation

If using :class:`~pepbench.datasets.BasePepDatasetWithAnnotations`, also implement:

* ``reference_labels_ecg`` property - Returns reference Q-peak labels
* ``reference_labels_icg`` property - Returns reference B-point labels
* ``reference_heartbeats`` property - Returns reference heartbeat segmentation

**3. Basic Example: Custom Dataset without Annotations**

Here's a minimal example for a dataset containing ECG/ICG files organized by participant:

.. code-block:: python

   from pathlib import Path
   import pandas as pd
   from biopsykit.utils.dtypes import EcgRawDataFrame, IcgRawDataFrame
   from biopsykit.signals.ecg.segmentation import HeartbeatSegmentationNeurokit
   from pepbench.datasets import BasePepDataset

   class MyCustomDataset(BasePepDataset):
       """Custom dataset for my ECG/ICG recordings.

       Parameters
       ----------
       base_path : Path or str
           Root directory containing participant folders
       """

       def __init__(self, base_path, **kwargs):
           self.base_path = Path(base_path)
           super().__init__(**kwargs)

       def create_index(self) -> pd.DataFrame:
           """Create index with one row per participant."""
           # Find all participant directories (e.g., P001, P002, ...)
           participant_ids = sorted([
               p.name for p in self.base_path.glob("P*")
               if p.is_dir()
           ])

           # Create DataFrame with participant column
           return pd.DataFrame({"participant": participant_ids})

       @property
       def sampling_rate_ecg(self) -> int:
           """ECG sampling rate in Hz."""
           return 1000  # Adjust to your data

       @property
       def sampling_rate_icg(self) -> int:
           """ICG sampling rate in Hz."""
           return 1000  # Adjust to your data

       @property
       def ecg(self) -> EcgRawDataFrame:
           """Load ECG signal for current subset."""
           # Ensure we're accessing a single datapoint
           if not self.is_single(None):
               raise ValueError("ECG can only be accessed for a single datapoint!")

           # Get participant ID from current index
           participant = self.index["participant"].iloc[0]

           # Load ECG file
           ecg_file = self.base_path / participant / "ecg.csv"
           data = pd.read_csv(ecg_file, index_col=0)

           return EcgRawDataFrame(data)

       @property
       def icg(self) -> IcgRawDataFrame:
           """Load ICG signal for current subset."""
           if not self.is_single(None):
               raise ValueError("ICG can only be accessed for a single datapoint!")

           participant = self.index["participant"].iloc[0]
           icg_file = self.base_path / participant / "icg.csv"
           data = pd.read_csv(icg_file, index_col=0)

           return IcgRawDataFrame(data)

       @property
       def heartbeats(self):
           """Compute heartbeats from ECG."""
           # Use BiopsyKit's heartbeat segmentation
           segmenter = HeartbeatSegmentationNeurokit()
           segmenter.segment(
               ecg=self.ecg,
               sampling_rate_hz=self.sampling_rate_ecg
           )
           return segmenter.heartbeats_

**4. Advanced Example: Dataset with Annotations and Multiple Index Levels**

For more complex scenarios with multiple conditions or sessions:

.. code-block:: python

   from itertools import product
   from pepbench.datasets import BasePepDatasetWithAnnotations

   class MyAnnotatedDataset(BasePepDatasetWithAnnotations):
       """Custom dataset with reference annotations.

       Parameters
       ----------
       base_path : Path or str
           Root directory
       only_labeled : bool
           Whether to restrict to labeled segments
       """

       CONDITIONS = ["rest", "exercise"]  # Define study conditions

       def __init__(self, base_path, only_labeled=False, **kwargs):
           self.base_path = Path(base_path)
           super().__init__(only_labeled=only_labeled, **kwargs)

       def create_index(self) -> pd.DataFrame:
           """Create index with participant and condition columns."""
           # Find participants
           participants = sorted([
               p.name for p in self.base_path.glob("P*")
               if p.is_dir()
           ])

           # Create all combinations of participant × condition
           index_tuples = list(product(participants, self.CONDITIONS))
           return pd.DataFrame(
               index_tuples,
               columns=["participant", "condition"]
           )

       @property
       def ecg(self) -> EcgRawDataFrame:
           """Load ECG for current participant and condition."""
           if not self.is_single(None):
               raise ValueError("Access single datapoint only!")

           p_id = self.index["participant"].iloc[0]
           condition = self.index["condition"].iloc[0]

           # Load from condition-specific file
           ecg_file = self.base_path / p_id / f"ecg_{condition}.csv"
           return EcgRawDataFrame(pd.read_csv(ecg_file, index_col=0))

       @property
       def icg(self) -> IcgRawDataFrame:
           """Load ICG for current participant and condition."""
           if not self.is_single(None):
               raise ValueError("Access single datapoint only!")

           p_id = self.index["participant"].iloc[0]
           condition = self.index["condition"].iloc[0]

           icg_file = self.base_path / p_id / f"icg_{condition}.csv"
           return IcgRawDataFrame(pd.read_csv(icg_file, index_col=0))

       @property
       def sampling_rate_ecg(self) -> int:
           return 500

       @property
       def sampling_rate_icg(self) -> int:
           return 500

       @property
       def heartbeats(self):
           """Compute heartbeats."""
           segmenter = HeartbeatSegmentationNeurokit()
           segmenter.segment(
               ecg=self.ecg,
               sampling_rate_hz=self.sampling_rate_ecg
           )
           return segmenter.heartbeats_

       @property
       def reference_labels_ecg(self) -> pd.DataFrame:
           """Load reference ECG labels (Q-peaks)."""
           if not self.is_single(None):
               raise ValueError("Access single datapoint only!")

           p_id = self.index["participant"].iloc[0]
           condition = self.index["condition"].iloc[0]

           # Load labels file
           labels_file = self.base_path / p_id / f"labels_ecg_{condition}.csv"
           labels = pd.read_csv(labels_file)

           # Must return MultiIndex format: (heartbeat_id, channel, label)
           labels = labels.set_index(["heartbeat_id", "channel", "label"])
           return labels

       @property
       def reference_labels_icg(self) -> pd.DataFrame:
           """Load reference ICG labels (B-points)."""
           if not self.is_single(None):
               raise ValueError("Access single datapoint only!")

           p_id = self.index["participant"].iloc[0]
           condition = self.index["condition"].iloc[0]

           labels_file = self.base_path / p_id / f"labels_icg_{condition}.csv"
           labels = pd.read_csv(labels_file)
           labels = labels.set_index(["heartbeat_id", "channel", "label"])
           return labels

       @property
       def reference_heartbeats(self) -> pd.DataFrame:
           """Load or compute reference heartbeat segmentation."""
           # Option 1: Load from file if available
           # Option 2: Compute from reference labels
           from pepbench.datasets._helper import compute_reference_heartbeats

           return compute_reference_heartbeats(
               self.reference_labels_ecg,
               sampling_rate_hz=self.sampling_rate_ecg
           )

**5. Key Implementation Tips**

* **Index Structure**: The ``create_index()`` method defines all available datapoints. Each row represents one accessible subset of your data. Column names become index levels you can group by.

* **Single Datapoint Access**: Properties like ``ecg``, ``icg``, etc. should typically only be accessed when ``self.is_single(None)`` is ``True``. Use this check to prevent ambiguous multi-subset access.

* **tpcp Integration**: By inheriting from :class:`~pepbench.datasets.BasePepDataset`, you automatically get:

  * Subsetting via ``get_subset()``
  * Grouping via ``groupby()``
  * Iteration over datapoints
  * Reproducible indexing for benchmarking

* **Data Loading**: Implement efficient data loading in your properties. Consider using caching (``@cached_property``) if loading is expensive:

  .. code-block:: python

     from functools import cached_property

     @cached_property
     def ecg(self) -> EcgRawDataFrame:
         # Expensive loading only happens once
         return self._load_ecg_file()

* **Reference Label Format**: Reference labels must be MultiIndex DataFrames with levels ``(heartbeat_id, channel, label)`` and columns including ``sample_relative`` and ``sample_absolute``.

**6. Using Your Custom Dataset**

Once implemented, use your dataset just like built-in ones:

.. code-block:: python

   # Create dataset instance
   ds = MyCustomDataset(base_path="/path/to/data")

   # Build index
   ds.create_index()

   # Iterate over all datapoints
   for datapoint in ds:
       ecg = datapoint.ecg
       icg = datapoint.icg
       # Process...

   # Group by participant
   for participant_subset in ds.groupby("participant"):
       # Process all conditions for this participant
       pass

   # Use with pipelines
   from pepbench.pipelines import PepExtractionPipeline

   pipeline = PepExtractionPipeline()
   pipeline.run(ds.get_subset(index=[0]))
   results = pipeline.result_

**7. Testing Your Dataset**

Always test your custom dataset implementation:

.. code-block:: python

   # Verify index creation
   ds = MyCustomDataset("/path/to/data")
   index = ds.create_index()
   assert len(index) > 0
   assert "participant" in index.columns

   # Test data access
   single_dp = ds.get_subset(index=[0])
   ecg = single_dp.ecg
   assert ecg.shape[0] > 0
   assert single_dp.sampling_rate_ecg > 0

   # Test iteration
   for dp in ds:
       assert dp.is_single(None)
       # Verify each datapoint is accessible

For complete examples, see :class:`~pepbench.datasets.EmpkinsDataset` and :class:`~pepbench.datasets.ExampleDataset` implementations in the pepbench source code.



Summary checklist
-----------------

To be pipeline-compatible, a dataset should:

* expose ECG/ICG signals and sampling rates
* provide heartbeat segmentation
* inherit from :class:`~pepbench.datasets.BasePepDataset` (or compatible interface)
* use deterministic indexing for reproducibility

For evaluation, it should additionally provide reference annotations.
