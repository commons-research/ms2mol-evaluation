import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Tuple

import ms_entropy as me
import numpy as np
import pandas as pd
from matchms import calculate_scores
from matchms.similarity import PrecursorMzMatch
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm.auto import tqdm
from tqdm.contrib import tzip

from ms2mol_evaluation.evaluation import Evaluation


class CFMEvaluation(Evaluation):
    def __init__(
        self,
        output_dir: Path,
        precursor_mz_tolerance: float,
        precursor_mz_tolerance_type: Literal["Dalton", "ppm"] = "ppm",
    ) -> None:
        # if the output dir path doesn't exist create all necessary directories
        super().__init__(output_dir)
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.df_file_path = self.output_dir / "cfmid_scores.parquet"
        if os.path.exists(self.df_file_path):
            os.remove(self.df_file_path)

        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=False)
        self.similarity_score = PrecursorMzMatch(
            tolerance=precursor_mz_tolerance,
            tolerance_type=precursor_mz_tolerance_type,
        )

    def set_ms2_similarity(self, similarity: BaseSimilarity) -> None:
        self.ms2_similarity = similarity

    def run_eval(self) -> pd.DataFrame:
        if not hasattr(self, "ms2_similarity"):
            raise ValueError(
                "MS2 similarity measure not set. Use `set_ms2_similarity` before using this function."
            )

        scores = calculate_scores(
            references=self.msg_spectra,
            queries=self.isdb_spectra,
            array_type="numpy",
            is_symmetric=False,
            similarity_function=self.similarity_score,
        )
        indices = np.where(np.asarray(scores.scores.to_array()))
        idx_row, idx_col = indices

        data = []
        for x, y in tzip(idx_row, idx_col):
            query_spectrum = self.msg_spectra[x]
            reference_spectrum = self.isdb_spectra[y]
            msms_score, n_matches = self.ms2_similarity.pair(
                query_spectrum, reference_spectrum
            )[()]

            entropy_sim = me.calculate_entropy_similarity(
                query_spectrum.peaks,
                reference_spectrum.peaks,
            )

            data.append(
                {
                    self.ms2_similarity.__class__.__name__: msms_score,
                    "entropy_similarity": entropy_sim,
                    "feature_id": query_spectrum.get("feature_id") or x + 1,
                    "reference_id": y,  # code copied from https://github.com/mandelbrot-project/met_annot_enhancer/blob/f8346fd3f7a9775d1d6638cf091d019167ba7ce1/src/dev/spectral_lib_matcher.py#L175
                    "inchikey_isdb": reference_spectrum.get("compound_name"),
                    "smiles_isdb": reference_spectrum.get("smiles"),
                    "inchikey_msg": query_spectrum.get("inchikey"),
                    "smiles_msg": query_spectrum.get("smiles"),
                    "adduct": query_spectrum.get("adduct"),
                    "instrument": query_spectrum.get("instrument_type"),
                    "identifier": query_spectrum.get("identifier"),
                    "fold": query_spectrum.get("fold"),
                    "msg_entropy": me.calculate_spectral_entropy(query_spectrum.peaks),
                    "isdb_entropy": me.calculate_spectral_entropy(
                        reference_spectrum.peaks
                    ),
                    "abs_precursor_mz_diff": abs(
                        query_spectrum.get("precursor_mz")
                        - reference_spectrum.get("precursor_mz")
                    ),
                    "ppm_precursor_mz_diff": abs(
                        query_spectrum.get("precursor_mz")
                        - reference_spectrum.get("precursor_mz")
                    )
                    / reference_spectrum.get("precursor_mz")
                    * 1e6,
                }
            )
        df = pd.DataFrame(data)
        df.to_parquet(
            self.df_file_path,
            index=False,
        )

        return df

    def write_top_k_proba_to_json(self, df: pd.DataFrame) -> None:
        df["is_correct"] = df["inchikey_isdb"] == df["inchikey_msg"]
        df["rank"] = (
            df.groupby("identifier")[f"{self.ms2_similarity.__class__.__name__}"]
            .rank(method="dense", ascending=False)
            .astype("int32")
        )
        hist = (
            df[df["is_correct"]]
            .groupby("identifier")["rank"]
            .min()
            .value_counts()
            .sort_index()
        )
        hist_proba = (hist / len(self.msg_spectra)).to_dict()
        with open(self.output_dir / "probabilites.json", "w") as f:
            json.dump(hist_proba, f, indent=4, sort_keys=True)

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
            inchikey = row["inchikey_isdb"]
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
