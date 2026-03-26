import os
from pathlib import Path
from typing import Literal
import ms_entropy as me
from ..gnps.download import load_gnps
import numpy as np
import pandas as pd
from matchms import calculate_scores
from matchms.similarity import PrecursorMzMatch
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm.auto import tqdm
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

        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=False)
        self.similarity_score = PrecursorMzMatch(
            tolerance=precursor_mz_tolerance,
            tolerance_type=precursor_mz_tolerance_type,
        )
        self.gnps: list[Spectrum] = load_gnps("downloads/gnps/GNPS.mgf")
        self.filter_gnps()
        
    def set_ms2_similarity(self, similarity: BaseSimilarity) -> None:
        self.ms2_similarity = similarity

    def filter_gnps(self) -> None:
        msg_hashes = {s.spectrum_hash() for s in self.msg_spectra}
        self.gnps = [
            s
            for s in tqdm(self.gnps, desc="Filtering GNPS lib", leave=False)
            if s.spectrum_hash() not in msg_hashes
        ]

    def run_eval(self) -> pd.DataFrame:
        if not hasattr(self, "ms2_similarity"):
            raise ValueError(
                "MS2 similarity measure not set. Use `set_ms2_similarity` before using this function."
            )
        interval = 1000
        chunks_query = [
            self.msg_spectra[x : x + interval]
            for x in range(0, len(self.msg_spectra), interval)
        ]
        scans_id_map = {}
        i = 0
        data = []
        for chunk_number, chunk in enumerate(tqdm(chunks_query)):
            scores = calculate_scores(chunk, self.gnps, self.similarity_score)
            idx_row = scores.scores[:, :][0]
            idx_col = scores.scores[:, :][1]
            for _ in chunk:
                scans_id_map[i] = i
                i += 1

            for x, y in zip(idx_row, idx_col):
                if x >= y:
                    continue
                res = self.ms2_similarity.pair(chunk[x], self.gnps[y])
                try:
                    msms_score, n_matches = res["score"], res["matches"]
                except:
                    msms_score = res
                    n_matches = None

                entropy_sim = me.calculate_entropy_similarity(
                    chunk[x].peaks,
                    self.gnps[y].peaks,
                    ms2_tolerance_in_da=0.01,
                )

                feature_id = scans_id_map[int(x) + int(interval * chunk_number)]
                data.append(
                    {
                        self.ms2_similarity.__class__.__name__: msms_score,
                        "entropy_similarity": entropy_sim,
                        "matched_peaks": n_matches if n_matches is not None else np.nan,
                        "matched_ratio": n_matches
                        / max(
                            len(self.msg_spectra[x].peaks.intensities),
                            len(self.gnps[y].peaks.intensities),
                        ),
                        "feature_id": feature_id,
                        "reference_id": y,  # code copied from https://github.com/mandelbrot-project/met_annot_enhancer/blob/f8346fd3f7a9775d1d6638cf091d019167ba7ce1/src/dev/spectral_lib_matcher.py#L175
                        "inchikey_gnps": self.gnps[y].get("inchikey")[:14],
                        "smiles_gnps": self.gnps[y].get("smiles"),
                        "inchikey_msg": chunk[x].get("inchikey"),
                        "smiles_msg": chunk[x].get("smiles"),
                        "adduct": chunk[x].get("adduct"),
                        "instrument": chunk[x].get("instrument_type"),
                        "identifier": chunk[x].get("identifier"),
                        "fold": chunk[x].get("fold"),
                    }
                )
        df = pd.DataFrame(data)
        df.to_csv(
            self.df_file_path,
            sep=",",
            index=False,
        )

        return df
