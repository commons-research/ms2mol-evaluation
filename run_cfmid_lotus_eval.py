import os
import typing as T

import pandas as pd
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, PrecursorMzMatch
from tqdm import tqdm

from ms2mol_evaluation.isdb import download_isdb, filter_massspecgym_spectra, load_isdb
from ms2mol_evaluation.massspecgym import load_massspecgym, to_spectra
from ms2mol_evaluation.spectrum import Spectrum


def main():
    download_isdb()
    massspecgym = load_massspecgym()
    spectra: T.List[Spectrum] = to_spectra(massspecgym)
    isdb: T.List[Spectrum] = load_isdb()

    # we filter the MassSpecGym spectra to only include those present in ISDB
    spectra = filter_massspecgym_spectra(spectra, isdb, hydrogen_adduct_only=True)

    similarity_score = PrecursorMzMatch(tolerance=10.0, tolerance_type="ppm")
    interval = 1000
    chunks_query = [spectra[x : x + interval] for x in range(0, len(spectra), interval)]

    cosinegreedy = CosineGreedy(tolerance=0.01)

    scans_id_map = {}
    i = 0
    for chunk_number, chunk in enumerate(tqdm(chunks_query)):
        scores = calculate_scores(chunk, isdb, similarity_score)
        idx_row = scores.scores[:, :][0]
        idx_col = scores.scores[:, :][1]

        for _ in chunk:
            scans_id_map[i] = i
            i += 1

        data = []
        for x, y in zip(idx_row, idx_col):
            if x >= y:
                continue
            msms_score, n_matches = cosinegreedy.pair(chunk[x], isdb[y])[()]
            # if (msms_score > 0.2) and (n_matches > 6):

            feature_id = scans_id_map[int(x) + int(interval * chunk_number)]
            data.append(
                {
                    "cosine_similarity": msms_score,
                    "matched_peaks": n_matches,
                    "feature_id": feature_id,
                    "reference_id": y,  # code copied from https://github.com/mandelbrot-project/met_annot_enhancer/blob/f8346fd3f7a9775d1d6638cf091d019167ba7ce1/src/dev/spectral_lib_matcher.py#L175
                    "inchikey_isdb": isdb[y].get("compound_name"),
                    "smiles_isdb": isdb[y].get("smiles"),
                    "inchikey_msg": chunk[x].get("inchikey"),
                    "smiles_msg": chunk[x].get("smiles"),
                    "adduct": chunk[x].get("adduct"),
                    "instrument": chunk[x].get("instrument_type"),
                    "identifier": chunk[x].get("identifier"),
                }
            )
        df = pd.DataFrame(data)
        df.to_csv(
            "lotus_cfmid_scores.csv",
            mode="a",
            header=not os.path.exists("lotus_cfmid_scores.csv"),
            sep=",",
            index=False,
        )


if __name__ == "__main__":
    main()
