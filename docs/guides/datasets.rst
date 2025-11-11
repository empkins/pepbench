.. _user_guide_datasets:

Datasets & Data Requirements
============================

pepbench is built around **object-oriented datasets** using the
`tpcp <https://pypi.org/project/tpcp/>`_ framework.
Each datapoint bundles the necessary ECG/ICG signals, annotations, and
metadata required to run PEP extraction pipelines.

Available Dataset Classes
-------------------------

pepbench currently provides two main datasets:

* :class:`pepbench.datasets.EmpkinsDataset`
* :class:`pepbench.datasets.GuardianDataset`

Both are subclasses of :class:`pepbench.datasets.BasePepDataset` /
:class:`pepbench.datasets.BasePepDatasetWithAnnotations` and therefore
implement a **unified interface** for PEP extraction.

Core dataset attributes
~~~~~~~~~~~~~~~~~~~~~~~

All PEP extraction pipelines expect datasets to provide at least:

* ``ecg`` – ECG signal as a pandas DataFrame
* ``icg`` – ICG signal as a pandas DataFrame
* ``sampling_rate_ecg`` – ECG sampling rate (Hz)
* ``sampling_rate_icg`` – ICG sampling rate (Hz)
* ``heartbeats`` – segmented heartbeats (start, end, R-peak per beat)

Datasets that support evaluation additionally expose:

* ``reference_pep`` – reference PEP values (per sample or per beat)
* ``reference_heartbeats`` – reference heartbeat segmentation
* ``reference_labels_ecg`` / ``reference_labels_icg`` – label annotations
* ``labeling_borders`` – labeled sections of the continuous signal

The concrete attribute list for :class:`EmpkinsDataset` is documented in
the API reference.

EmpkinsDataset in a nutshell
----------------------------

:class:`pepbench.datasets.EmpkinsDataset` is the tpcp dataset class for
the EmpkinS study. It provides:

* Biopac-derived ECG and ICG channels (raw or preprocessed)
* Timelogs for experimental phases
* Human expert annotations (Q, B, labels)
* Basic metadata such as age, gender, BMI

Typical construction:

.. code-block:: python

   from pepbench.datasets import EmpkinsDataset

   ds = EmpkinsDataset(
       base_path="/path/to/empkins/root",
       only_labeled=True,      # restrict to segments with reference labels
       exclude_missing_data=True,
       label_type="rater_01",  # or "rater_02" / "average"
   )

   # Always build the index once (done lazily otherwise)
   ds.create_index()
   print(len(ds), "datapoints in the dataset")

Accessing signals and metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each element of the dataset is a *datapoint* that exposes properties:

.. code-block:: python

   # Access a single datapoint (by position)
   datapoint = ds.get_subset(index=[ds.index_as_tuples()[0]])

   ecg = datapoint.ecg                    # ECG channel
   icg = datapoint.icg                    # ICG channel
   fs_ecg = datapoint.sampling_rate_ecg   # ECG sampling rate
   fs_icg = datapoint.sampling_rate_icg   # ICG sampling rate

   # Metadata
   meta = datapoint.metadata
   print(meta[["age", "gender", "bmi"]])

Iterating over datapoints
~~~~~~~~~~~~~~~~~~~~~~~~~

Because EmpkinsDataset is a tpcp dataset, it is iterable:

.. code-block:: python

   for dp in ds:
       # Each dp is a single participant/condition/phase combination
       ecg = dp.ecg
       icg = dp.icg
       heartbeats = dp.heartbeats
       # ... run a pipeline or custom analysis

You can also iterate on *groups* (e.g., by participant) using
:meth:`EmpkinsDataset.groupby`:

.. code-block:: python

   # Create per-participant subsets
   ds_participant = ds.groupby(["participant"])

   for participant_subset in ds_participant.iter_level(level=0):
       print("Participant:", participant_subset.metadata["participant"])
       for dp in participant_subset:
           do_something(dp)

Working with heartbeats
-----------------------

Heartbeats are stored as a DataFrame with one row per beat and
columns giving start and end indices plus the R-peak sample.

.. code-block:: python

   dp = next(iter(ds))
   hb = dp.heartbeats

   print(hb.head())

   for beat_id, row in hb.iterrows():
       start = row["start"]
       end = row["end"]
       r_peak = row["r_peak"]
       # Use this to align ECG/ICG segments or to debug detected peaks

GuardianDataset overview
------------------------

:class:`pepbench.datasets.GuardianDataset` follows the same interface as
:class:`EmpkinsDataset`, but is based on the Guardian study data
(including ECG/ICG and metadata such as age, gender, BMI).

The main conceptual difference for pepbench users is simply:

* Different recording protocol and population
* Potentially different sampling rates and signal characteristics

From the perspective of pipelines and evaluation, both datasets can be
used interchangeably as long as they inherit from
:class:`BasePepDatasetWithAnnotations`.

Summary checklist
-----------------

To use a dataset with pepbench pipelines and challenges, make sure it:

* exposes ``ecg``, ``icg``, ``sampling_rate_ecg``, ``sampling_rate_icg``
* for evaluation: additionally exposes reference annotations
  (heartbeats, PEP, labels)
* is a subclass of :class:`BasePepDataset` (or compatible interface)
* is *indexed deterministically* (important for reproducibility)
