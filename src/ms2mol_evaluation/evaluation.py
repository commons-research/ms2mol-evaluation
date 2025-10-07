import typing as T
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from matchms.filtering import default_filters
from numba import njit
from numpy.typing import NDArray
from pandarallel import pandarallel
from tqdm.auto import tqdm

from ms2mol_evaluation.spectrum import Spectrum

pandarallel.initialize(progress_bar=True)


class Evaluation(ABC):
    def __init__(self, scores: NDArray[np.float32], output_dir: Path) -> None:
        self.scores = scores
        self.output_dir = output_dir
        self.inchikeys_as_columns: T.List[str] = []
        self.identifiers_as_rows: T.List[str] = []
        self.msg_df: pd.DataFrame = Evaluation._load_massspecgym()
        self.msg_spectra: T.List[Spectrum] = Evaluation._to_spectra(self.msg_df)

    def get_scores(self) -> NDArray[np.float32]:
        return self.scores

    @njit
    def get_fraction_of_true(
        self,
        true_column_index: np.ndarray,
        score_threshold=0.0,
    ) -> T.Tuple[float, float]:
        """
        Given a threshold, this function calculates the fraction of rows
        where the score for the true compound is above the threshold
        """
        scores_smaller = self.scores.copy()
        orig_shape = scores_smaller.shape
        scores_smaller = scores_smaller.flatten()
        scores_smaller[scores_smaller < score_threshold] = np.nan
        scores_smaller = scores_smaller.reshape(orig_shape)

        # we iterate over the rows of the array
        fraction_of_true_among_df = 0
        for i, row in zip(true_column_index, scores_smaller):
            if np.isnan(row[i]):
                continue
            fraction_of_true_among_df += 1

        fraction_of_true = fraction_of_true_among_df / scores_smaller.shape[0]
        fraction_of_df = 1 - (np.isnan(scores_smaller).sum() / scores_smaller.size)
        return fraction_of_true, fraction_of_df

    def evaluate_fraction_of_true(
        self,
        all_inchikeys: T.List[str],
        identifiers: T.List[str],
        identifier_to_inchikey: T.Dict[str, str],
        interval: T.Iterable[float] = np.arange(0.0, 1.0, 0.05),
    ) -> T.Tuple[T.List[float], T.List[float]]:
        """TODO: docstring"""
        fraction_true_lst = []
        fraction_df_lst = []
        column_indices = {col: idx for idx, col in enumerate(all_inchikeys)}
        true_indices = np.array(
            [column_indices[identifier_to_inchikey[i]] for i in identifiers]
        )

        for threshold in tqdm(interval, desc="Thresholds"):
            fraction_true, fraction_df = self.get_fraction_of_true(
                true_column_index=true_indices,
                score_threshold=threshold,
            )
            fraction_true_lst.append(fraction_true)
            fraction_df_lst.append(fraction_df)

        return fraction_true_lst, fraction_df_lst

    @staticmethod
    def _load_massspecgym() -> pd.DataFrame:
        """
        Load the MassSpecGym dataset.
        """
        df = df = pl.read_csv(
            "hf://datasets/roman-bushuiev/MassSpecGym/data/MassSpecGym.tsv",
            separator="\t",
        ).to_pandas()
        df = df.set_index("identifier")
        df["mzs"] = df["mzs"].apply(Evaluation._parse_spec_array)
        df["intensities"] = df["intensities"].apply(Evaluation._parse_spec_array)

        df["spectrum"] = df.apply(
            lambda row: np.array([row["mzs"], row["intensities"]]), axis=1
        )
        return df

    @staticmethod
    def _parse_spec_array(arr: str) -> np.ndarray:
        return np.array(list(map(float, arr.split(","))))

    @staticmethod
    def _to_spectrum(row: pd.Series) -> Spectrum:
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

    @staticmethod
    def _to_spectra(df: pd.DataFrame) -> T.List[Spectrum]:
        # Apply to_spectrum + default_filters in parallel
        spectra = df.parallel_apply(
            lambda row: default_filters(Evaluation._to_spectrum(row)), axis=1
        ).tolist()

        spectra = [
            Spectrum(mz=s.mz, intensities=s.intensities, metadata=s.metadata)
            for s in spectra
        ]
        return spectra
