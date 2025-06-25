import typing as T

import numpy as np
import pandas as pd
from cache_decorator import Cache
from huggingface_hub import hf_hub_download
from matchms.filtering import default_filters
from matchms.logging_functions import set_matchms_logger_level
from pandarallel import pandarallel

from metfrag_evaluation.spectrum import Spectrum

set_matchms_logger_level("ERROR")
# Initialize pandarallel (add progress bar if you want)
pandarallel.initialize(progress_bar=True)


def parse_spec_array(arr: str) -> np.ndarray:
    return np.array(list(map(float, arr.split(","))))


def hugging_face_download(file_name: str) -> str:
    """
    Download a file from the Hugging Face Hub and return its location on disk.

    Args:
        file_name (str): Name of the file to download.
    """
    return hf_hub_download(
        repo_id="roman-bushuiev/MassSpecGym",
        filename="data/" + file_name,
        repo_type="dataset",
    )


@Cache(use_approximated_hash=True)
def load_massspecgym(fold: T.Optional[str] = None) -> pd.DataFrame:
    """
    Load the MassSpecGym dataset.

    Args:
        fold (str, optional): Fold name to load. If None, the entire dataset is loaded.
    """
    df = pd.read_csv(hugging_face_download("MassSpecGym.tsv"), sep="\t")
    df = df.set_index("identifier")
    df["mzs"] = df["mzs"].apply(parse_spec_array)
    df["intensities"] = df["intensities"].apply(parse_spec_array)
    if fold is not None:
        df = df[df["fold"] == fold]

    df["spectrum"] = df.apply(
        lambda row: np.array([row["mzs"], row["intensities"]]), axis=1
    )
    return df


def to_spectrum(row: pd.Series) -> Spectrum:
    """
    Convert a DataFrame row to a Spectrum object.
    """
    return Spectrum(
        mz=np.array(row["mzs"]),
        intensities=np.array(row["intensities"]),
        metadata={
            "identifier": row.name,
            "smiles": row["smiles"],
            "inchikey": row["inchikey"],
            "formula": row["formula"],
            "precursor_formula": row["precursor_formula"],
            "parent_mass": row["parent_mass"],
            "precursor_mz": row["precursor_mz"],
            "adduct": row["adduct"],
            "instrument_type": row["instrument_type"],
            "collision_energy": row["collision_energy"],
            "fold": row["fold"],
            "simulation_challenge": row["simulation_challenge"],
        },
    )


@Cache(
    cache_path="cache/{function_name}/{_hash}/spectra.pkl",
    use_approximated_hash=True,
)
def to_spectra(df: pd.DataFrame) -> T.List[Spectrum]:
    # Apply to_spectrum + default_filters in parallel
    spectra = df.parallel_apply(
        lambda row: default_filters(to_spectrum(row)), axis=1
    ).tolist()

    spectra = [
        Spectrum(mz=s.mz, intensities=s.intensities, metadata=s.metadata)
        for s in spectra
    ]
    return spectra
