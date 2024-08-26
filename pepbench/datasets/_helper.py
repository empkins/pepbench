import ast

import pandas as pd

from pepbench._utils._types import path_t
from pepbench.datasets import BaseUnifiedPepExtractionDataset


def load_labeling_borders(file_path: path_t) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    data = data.assign(description=data["description"].apply(lambda s: ast.literal_eval(s)))

    data = data.set_index("timestamp").sort_index()
    return data


def compute_reference_pep(subset: BaseUnifiedPepExtractionDataset) -> pd.DataFrame:
    reference_icg = subset.reference_labels_icg
    reference_ecg = subset.reference_labels_ecg

    # artefact_ids = pd.concat(
    #     [reference_icg.reindex(["Artefact"], level="label"), reference_ecg.reindex(["Artefact"], level="label")]
    # )
    # artefact_ids = artefact_ids.index.get_level_values("heartbeat_id")

    b_points = reference_icg.xs("ICG", level="channel")
    qwave_onsets = reference_ecg.xs("ECG", level="channel")

    pep_reference = pd.concat([qwave_onsets, b_points]).sort_index()
    pep_reference = pep_reference["sample_relative"].unstack("label")

    pep_reference.columns = ["q_wave_onset_sample", "b_point_sample"]
    pep_reference = pep_reference.assign(pep_sample=-1 * pep_reference.diff(axis=1).dropna(axis=1, how="all"))
    pep_reference = pep_reference.assign(pep_ms=pep_reference["pep_sample"] / subset.sampling_rate_ecg * 1000)
    return pep_reference
