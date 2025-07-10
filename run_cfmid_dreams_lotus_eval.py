import os
import typing as T

import numpy as np
import pandas as pd
from cache_decorator import Cache
from downloaders import BaseDownloader
from dreams.api import dreams_embeddings
from matchms.exporting import save_as_mgf
from matchms.filtering import default_filters
from matchms.importing import load_from_mgf
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange

from metfrag_evaluation.massspecgym import load_massspecgym, to_spectra
from metfrag_evaluation.spectrum import Spectrum


def download_isdb() -> None:
    downloader = BaseDownloader(auto_extract=False)
    _ = downloader.download(
        "https://zenodo.org/records/14887271/files/isdb_lotus_pos_energySum.mgf",
        "data/isdb/isdb_lotus_pos_energySum.mgf",
    )


def load_isdb_dreams_embedding() -> np.ndarray:
    """Load ISDB spectra from MGF file."""

    dreams_embs = load_dreams_embedding("data/isdb/isdb_lotus_pos_energySum.mgf")

    return dreams_embs


@Cache()
def load_isdb() -> T.List[Spectrum]:
    """Load ISDB spectra from MGF file."""
    spectra = []
    for spectrum in tqdm(
        load_from_mgf("data/isdb/isdb_lotus_pos_energySum.mgf"),
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


def filter_massspecgym_spectra(
    massspecgym_spectra: T.List[Spectrum], isdb_spectra: T.List[Spectrum]
) -> T.List[Spectrum]:
    """Filter MassSpecGym spectra to only include those present in ISDB."""
    isdb_inchikeys = set(s.get("compound_name") for s in isdb_spectra)
    filtered_spectra = [
        s
        for s in tqdm(
            massspecgym_spectra, leave=False, desc="Filtering MassSpecGym spectra"
        )
        if s.get("inchikey") in isdb_inchikeys
    ]
    filtered_spectra = [s for s in filtered_spectra if s.get("adduct") == "[M+H]+"]
    return filtered_spectra


@Cache()
def load_dreams_embedding(path: str) -> np.ndarray:
    """Load DREAMS embeddings from a file."""
    dreams_embs: np.ndarray = dreams_embeddings(path, prec_mz_col="PRECURSOR_MZ")
    return dreams_embs


def main():
    download_isdb()
    massspecgym = load_massspecgym()
    spectra: T.List[Spectrum] = to_spectra(massspecgym)
    isdb: T.List[Spectrum] = load_isdb()

    # load ISDB DreaMS embedding
    isdb_embedding: np.ndarray = load_isdb_dreams_embedding()

    # we filter the MassSpecGym spectra to only include those present in ISDB
    spectra = filter_massspecgym_spectra(spectra, isdb)

    os.makedirs("data/massspecgym", exist_ok=True)
    save_as_mgf(spectra, "data/massspecgym/massspecgym.mgf", file_mode="w")
    spectra_embedding = load_dreams_embedding("data/massspecgym/massspecgym.mgf")

    sims = cosine_similarity(spectra_embedding, isdb_embedding)

    # now we sort
    sorted_indices = np.argsort(sims, axis=1)[:, ::-1]

    scores_of_true_inchikey = []
    positions_of_true_inchikey = []
    inichikeys_of_true_mol = []
    smiles_of_true_inchikey = []

    for i in trange(sorted_indices.shape[0], desc="Processing spectra"):
        spectrum = spectra[i]
        for j in range(sorted_indices.shape[1]):
            if spectrum.get("inchikey") != isdb[sorted_indices[i, j]].get(
                "compound_name"
            ):
                continue

            scores_of_true_inchikey.append(sims[i, sorted_indices[i, j]])
            positions_of_true_inchikey.append(j + 1)
            inichikeys_of_true_mol.append(spectrum.get("inchikey"))
            smiles_of_true_inchikey.append(spectrum.get("smiles"))
            break

    df = pd.DataFrame(
        {
            "score": scores_of_true_inchikey,
            "top_n": positions_of_true_inchikey,
            "inchikey": inichikeys_of_true_mol,
            "smiles": smiles_of_true_inchikey,
        }
    )
    df.to_csv("lotus_cfmid_dreams_scores.csv", index=False)


if __name__ == "__main__":
    main()
