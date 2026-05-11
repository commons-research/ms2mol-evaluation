import os
from pathlib import Path
from typing import Literal

import ms_entropy as me
import pandas as pd
from matchms import calculate_scores
from matchms.similarity import PrecursorMzMatch
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tqdm.auto import tqdm

from ms2mol_evaluation.evaluation import Evaluation
from ms2mol_evaluation.spectrum import Spectrum

from ..gnps.download import load_gnps


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
        self.df_file_path = self.output_dir / "gnps_scores.parquet"
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
        print("Number of spectra in GNPS before filtering: ", len(self.gnps))
        msg_hashes = {s.spectrum_hash() for s in self.msg_spectra}
        self.gnps = [
            s
            for s in tqdm(self.gnps, desc="Filtering GNPS lib", leave=False)
            if s.spectrum_hash() not in msg_hashes
        ]
        print("Number of spectra in GNPS after filtering: ", len(self.gnps))

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
                query_spectrum: Spectrum = chunk[x]
                reference_spectrum: Spectrum = self.gnps[y]
                if x >= y:
                    continue
                res = self.ms2_similarity.pair(query_spectrum, reference_spectrum)
                try:
                    msms_score, _ = res["score"], res["matches"]
                except:
                    msms_score = res
                    _ = None

                entropy_sim = me.calculate_entropy_similarity(
                    query_spectrum.peaks,
                    reference_spectrum.peaks,
                )

                feature_id = scans_id_map[int(x) + int(interval * chunk_number)]
                data.append(
                    {
                        self.ms2_similarity.__class__.__name__: msms_score,
                        "entropy_similarity": entropy_sim,
                        "feature_id": feature_id,
                        "reference_id": y,  # code copied from https://github.com/mandelbrot-project/met_annot_enhancer/blob/f8346fd3f7a9775d1d6638cf091d019167ba7ce1/src/dev/spectral_lib_matcher.py#L175
                        "inchikey_gnps": reference_spectrum.get("inchikey")[:14],
                        "smiles_gnps": reference_spectrum.get("smiles"),
                        "inchikey_msg": query_spectrum.get("inchikey"),
                        "smiles_msg": query_spectrum.get("smiles"),
                        "adduct": query_spectrum.get("adduct"),
                        "instrument": query_spectrum.get("instrument_type"),
                        "identifier": query_spectrum.get("identifier"),
                        "fold": query_spectrum.get("fold"),
                        "msg_entropy": me.calculate_spectral_entropy(
                            query_spectrum.peaks
                        ),
                        "gnps_entropy": me.calculate_spectral_entropy(
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
        df.to_parquet(self.df_file_path, index=False)

        return df
