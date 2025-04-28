import pandas as pd

__all__ = ["merge_pep_with_reference"]


def merge_pep_with_reference(pep_results: pd.DataFrame, reference_pep: pd.DataFrame) -> pd.DataFrame:
    pep_merged = reference_pep.join(pep_results, lsuffix="_reference")
    return pep_merged
