import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import numpy as np
import pandas as pd
from downloaders import BaseDownloader
from dreams.utils.data import MSData
from matchms import calculate_scores
from matchms.filtering import default_filters
from matchms.importing import load_from_mgf
from matchms.similarity import PrecursorMzMatch
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm.auto import tqdm
from tqdm.contrib import tzip

from ms2mol_evaluation.evaluation import Evaluation
from ms2mol_evaluation.spectrum import Spectrum


class GNPSEvaluation(Evaluation):
    def __init__(
        self,
        output_dir: Path | str,
        precursor_mz_tolerance: float,
        precursor_mz_tolerance_type: Literal["Dalton", "ppm"] = "ppm",
    ) -> None:
        super().__init__(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df_file_path = self.output_dir / "gnps_scores.csv"
        if os.path.exists(self.df_file_path):
            os.remove(self.df_file_path)

        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=True)
        self.similarity_score = PrecursorMzMatch(
            tolerance=precursor_mz_tolerance,
            tolerance_type=precursor_mz_tolerance_type,
        )
        self.gnps = self._load_gnps()

    # @staticmethod
    # def load_gnps(in_memory: bool = True, **kwargs) -> MSData:
    #     path: Path = Path(GNPSEvaluation.download_gnps().destination.values[0])
    #     h5_path = path.with_suffix(".hdf5")
    #     if h5_path.exists():
    #         # If the HDF5 already exists, load it directly (this is fastest)
    #         return MSData.from_hdf5(h5_path, in_mem=in_memory)
    #     else:
    #         return MSData.load(path, in_mem=in_memory, **kwargs)

    # @staticmethod
    # def download_gnps(auto_extract: bool = False) -> pd.DataFrame:
    #     return BaseDownloader(auto_extract=auto_extract).download(
    #         "https://external.gnps2.org/processed_gnps_data/matchms.mgf",
    #         str((Path("downloads/gnps") / f"gnps_matchms_filtered.mgf").resolve()),
    #     )

    def _load_gnps(self) -> List[Spectrum]:
        spectra: List[Spectrum] = []
        for spectrum in tqdm(
            load_from_mgf("downloads/gnps/cleaned_spectra.mgf"),
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

        return spectra

    def set_ms2_similarity(self, similarity: BaseSimilarity) -> None:
        self.ms2_similarity = similarity

    def run_eval(self) -> pd.DataFrame:
        if not hasattr(self, "ms2_similarity"):
            raise ValueError(
                "MS2 similarity measure not set. Use `set_ms2_similarity` before using this function."
            )
        scores = calculate_scores(
            references=self.msg_spectra,
            queries=self.gnps,
            array_type="numpy",
            is_symmetric=False,
            similarity_function=self.similarity_score,
        )
        indices = np.where(np.asarray(scores.scores.to_array()))
        idx_row, idx_col = indices

        data = []
        for x, y in tzip(idx_row, idx_col):
            msms_score, n_matches = self.ms2_similarity.pair(
                self.msg_spectra[x], self.gnps[y]
            )[()]

            data.append(
                {
                    self.ms2_similarity.__class__.__name__: msms_score,
                    "matched_peaks": n_matches if n_matches is not None else np.nan,
                    "matched_ratio": n_matches
                    / max(
                        len(self.msg_spectra[x].peaks.intensities),
                        len(self.isdb_spectra[y].peaks.intensities),
                    ),
                    "feature_id": self.msg_spectra[x].get("feature_id") or x + 1,
                    "reference_id": y,  # code copied from https://github.com/mandelbrot-project/met_annot_enhancer/blob/f8346fd3f7a9775d1d6638cf091d019167ba7ce1/src/dev/spectral_lib_matcher.py#L175
                    "inchikey_gnps": self.gnps[y].get("inchikey")[:14],
                    "smiles_gnps": self.gnps[y].get("smiles"),
                    "inchikey_msg": self.msg_spectra[x].get("inchikey"),
                    "smiles_msg": self.msg_spectra[x].get("smiles"),
                    "adduct": self.msg_spectra[x].get("adduct"),
                    "instrument": self.msg_spectra[x].get("instrument_type"),
                    "identifier": self.msg_spectra[x].get("identifier"),
                }
            )
        df = pd.DataFrame(data)
        df.to_csv(
            self.df_file_path,
            header=True,
            sep=",",
            index=False,
        )

        return df

    def _create_scores_array(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        index: List[str] = [s.get("identifier") for s in self.msg_spectra]
        identifier_to_inchikey = {
            s.get("identifier"): s.get("inchikey") for s in self.msg_spectra
        }
        id_to_int = {s.get("identifier"): i for i, s in enumerate(self.msg_spectra)}
        all_inchikeys = sorted(set(s.get("compound_name") for s in self.isdb_spectra))
        inchi_to_int = {inchk: i for i, inchk in enumerate(all_inchikeys)}
        self.scores = np.empty(
            (len(self.msg_spectra), len(all_inchikeys)), dtype=np.float16
        )
        self.scores.fill(np.nan)
        for i, row in tqdm(df.iterrows(), total=len(df)):
            identifier = row["identifier"]
            inchikey = row["inchikey_gnps"]
            if identifier not in id_to_int or inchikey not in inchi_to_int:
                continue
            self.scores[id_to_int[identifier], inchi_to_int[inchikey]] = row[
                self.ms2_similarity.__class__.__name__
            ]

        return (
            all_inchikeys,
            index,
            identifier_to_inchikey,
        )

    def get_fraction_results(
        self,
        df: pd.DataFrame,
        interval: Iterable[float] = np.arange(0.0, 1.0, 0.05),
    ) -> None:
        (
            all_inchikeys,
            identifiers,
            identifier_to_inchikey,
        ) = self._create_scores_array(df)

        y_fraction, x_fraction = self.evaluate_fraction(
            all_inchikeys=all_inchikeys,
            identifiers=identifiers,
            identifier_to_inchikey=identifier_to_inchikey,
            interval=interval,
        )

        self.plot_results(x_fraction, y_fraction, interval, image_name="fraction.png")

    def get_top_n_results(
        self,
        df: pd.DataFrame,
        interval: Iterable[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500],
    ) -> None:
        (
            all_inchikeys,
            identifiers,
            identifier_to_inchikey,
        ) = self._create_scores_array(df)
        y_top_n, x_top_n = self.evaluate_top_n(
            all_inchikeys=all_inchikeys,
            identifiers=identifiers,
            identifier_to_inchikey=identifier_to_inchikey,
            interval=interval,
        )
        self.plot_results(x_top_n, y_top_n, interval, image_name="top_n.png")
