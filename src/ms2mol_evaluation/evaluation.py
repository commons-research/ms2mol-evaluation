import typing as T
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cache_decorator import Cache
from downloaders import BaseDownloader
from matchms.filtering import default_filters
from matchms.importing import load_from_mgf
from numpy.typing import NDArray
from pandarallel import pandarallel
from tqdm.auto import tqdm

from ms2mol_evaluation.spectrum import Spectrum


class Evaluation:
    def __init__(self, output_dir: Union[Path, str]) -> None:
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scores: NDArray[np.float16] = np.array([], dtype=np.float16)
        self.msg_df: pd.DataFrame = Evaluation._load_massspecgym()
        self.msg_spectra: T.List[Spectrum] = Evaluation._to_spectra(self.msg_df)
        self.isdb_spectra: T.List[Spectrum] = Evaluation._load_isdb()
        self.msg_is_filtered: bool = False

    def get_scores(self) -> NDArray[np.float16]:
        return self.scores

    def _get_top_n(
        self,
        columns: T.List[str],
        index: T.List[str],
        id_to_inchikey: T.Dict[str, str],
        top_n: int = 1,
    ) -> T.Tuple[float, float]:
        scores_top_n = np.full_like(self.scores, np.nan)
        for i in range(self.scores.shape[0]):
            row = self.scores[i]
            if np.all(np.isnan(row)):
                continue
            # Get indices of top N scores (ignoring NaNs)
            valid_idx = np.where(~np.isnan(row))[0]
            if len(valid_idx) == 0:
                continue
            top_idx = valid_idx[np.argsort(row[valid_idx])[-top_n:]]
            scores_top_n[i, top_idx] = row[top_idx]

        fraction_of_true_among_df = 0
        column_indices = {col: idx for idx, col in enumerate(columns)}
        for i, row in zip(index, scores_top_n):
            column_index = column_indices[id_to_inchikey[i]]
            if np.isnan(row[column_index]):
                continue
            fraction_of_true_among_df += 1

        fraction_of_true = fraction_of_true_among_df / scores_top_n.shape[0]
        fraction_of_df = 1 - (np.isnan(scores_top_n).sum() / scores_top_n.size)
        return fraction_of_true, fraction_of_df

    def _get_fraction(
        self,
        true_column_index: NDArray[np.int64],
        score_threshold=0.0,
    ) -> T.Tuple[float, float]:
        """
        Given a threshold, this function calculates the fraction of rows
        where the score for the true compound is above the threshold
        """
        scores_smaller = self.scores.copy()
        scores_smaller[scores_smaller < score_threshold] = np.nan

        # we iterate over the rows of the array
        fraction_of_true_among_df = 0
        for i, row in zip(true_column_index, scores_smaller):
            if np.isnan(row[i]):
                continue
            fraction_of_true_among_df += 1

        fraction_of_true = fraction_of_true_among_df / scores_smaller.shape[0]
        fraction_of_df = 1 - (np.isnan(scores_smaller).sum() / scores_smaller.size)
        return fraction_of_true, fraction_of_df

    def evaluate_fraction(
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
            [column_indices[identifier_to_inchikey[i]] for i in identifiers],
            dtype=np.int64,
        )

        for threshold in tqdm(interval, desc="Thresholds"):
            fraction_true, fraction_df = self._get_fraction(
                true_column_index=true_indices,
                score_threshold=threshold,
            )
            fraction_true_lst.append(fraction_true)
            fraction_df_lst.append(fraction_df)

        return fraction_true_lst, fraction_df_lst

    def evaluate_top_n(
        self,
        all_inchikeys: T.List[str],
        identifiers: T.List[str],
        identifier_to_inchikey: T.Dict[str, str],
        interval: T.Iterable[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500],
    ) -> T.Tuple[T.List[float], T.List[float]]:
        fraction_true_lst = []
        fraction_df_lst = []
        for threshold in tqdm(interval, desc="Top N"):
            fraction_true, fraction_df = self._get_top_n(
                all_inchikeys,
                identifiers,
                identifier_to_inchikey,
                top_n=threshold,
            )
            fraction_true_lst.append(fraction_true)
            fraction_df_lst.append(fraction_df)

        return fraction_true_lst, fraction_df_lst

    @staticmethod
    @Cache()
    def _load_massspecgym() -> pd.DataFrame:
        """
        Load the MassSpecGym dataset.
        """
        df = pd.read_csv(
            "hf://datasets/roman-bushuiev/MassSpecGym/data/MassSpecGym.tsv",
            sep="\t",
        )
        df = df.set_index("identifier")
        df["mzs"] = df["mzs"].map(Evaluation._parse_spec_array)
        df["intensities"] = df["intensities"].map(Evaluation._parse_spec_array)

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
    @Cache(
        use_approximated_hash=True,
    )
    def _to_spectra(df: pd.DataFrame) -> T.List[Spectrum]:
        pandarallel.initialize(progress_bar=False)
        # Apply to_spectrum + default_filters in parallel
        spectra = df.parallel_apply(
            lambda row: default_filters(Evaluation._to_spectrum(row)), axis=1
        ).tolist()

        converted_spectra = []
        for s in tqdm(
            spectra,
            desc="Converting to custom Spectrum class",
            leave=False,
        ):
            converted_spectra.append(
                Spectrum(
                    mz=s.mz,
                    intensities=s.intensities,
                    metadata=s.metadata,
                )
            )
        return converted_spectra

    @Cache()
    def _load_isdb() -> T.List[Spectrum]:
        """Load ISDB spectra from MGF file."""
        downloader = BaseDownloader(auto_extract=False)
        report = downloader.download(
            "https://zenodo.org/records/14887271/files/isdb_lotus_pos_energySum.mgf",
            "downloads/isdb/isdb_lotus_pos_energySum.mgf",
        )
        file_name = report.destination.values[0]
        spectra = []
        for spectrum in tqdm(
            load_from_mgf(file_name),
            desc="Loading ISDB spectra",
            leave=False,
        ):
            spectrum = default_filters(spectrum)
            spectrum = Spectrum(
                mz=spectrum.mz,
                intensities=spectrum.intensities,
                metadata=spectrum.metadata,
            )
            spectra.append(spectrum)

        spectra = [
            s for s in spectra if "C" in s.get("smiles") or "c" in s.get("smiles")
        ]
        # keep only spectra that have the following atoms : ['Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S']
        allowed_atoms = {"Br", "C", "Cl", "F", "I", "N", "O", "P", "S"}

        def has_only_allowed_atoms(smiles: str) -> bool:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            atoms = {atom.GetSymbol() for atom in mol.GetAtoms()}
            return atoms.issubset(allowed_atoms)

        spectra = [s for s in spectra if has_only_allowed_atoms(s.get("smiles"))]
        # filter spectra with precursor_mz < 1000
        spectra = [
            s
            for s in spectra
            if s.get("precursor_mz") < 1000 and s.get("precursor_mz") > 20
        ]
        return spectra

    def _filter_massspecgym_spectra(
        self,
        hydrogen_adduct_only: bool = False,
    ) -> T.List[Spectrum]:
        """Filter MassSpecGym spectra to have only the hdyrogen adducts or not."""
        if hydrogen_adduct_only:
            return [s for s in self.msg_spectra if s.get("adduct") == "[M+H]+"]

        self.msg_is_filtered = True
        return self.msg_spectra

    def plot_results(self, x_axis, y_axis, interval, image_name: str) -> None:
        # Plotting
        ax = sns.scatterplot(x=x_axis, y=y_axis, hue=interval)
        ax.set_title("Evaluation Results")
        ax.set_xlabel("Fraction of values that are not NaN")
        ax.set_ylabel("Fraction of rows where true compound is not NaN")
        ax.set_ylim(0, 1.1)
        ax.legend(title="Threshold")
        ax.axhline(y=1.0, color="red", linestyle="--")
        fig = ax.get_figure()
        fig.savefig(self.output_dir / image_name)
        plt.close(fig)
        del ax, fig
