import typing as T

from cache_decorator import Cache
from downloaders import BaseDownloader
from matchms.filtering import default_filters
from matchms.importing import load_from_mgf
from tqdm import tqdm

from ms2mol_evaluation.spectrum import Spectrum


def download_isdb() -> None:
    downloader = BaseDownloader(auto_extract=False)
    _ = downloader.download(
        "https://zenodo.org/records/14887271/files/isdb_lotus_pos_energySum.mgf",
        "data/isdb/isdb_lotus_pos_energySum.mgf",
    )


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
    massspecgym_spectra: T.List[Spectrum],
    isdb_spectra: T.List[Spectrum],
    hydrogen_adduct_only: bool = False,
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
    if hydrogen_adduct_only:
        filtered_spectra = [s for s in filtered_spectra if s.get("adduct") == "[M+H]+"]
    return filtered_spectra
