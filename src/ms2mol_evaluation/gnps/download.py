from cache_decorator import Cache
import pandas as pd
from pathlib import Path
from downloaders import BaseDownloader
from tqdm.auto import tqdm
from matchms.importing import load_from_mgf
from ..spectrum import Spectrum

VALIDITY_DURATION = "30d"
GNPS_URL: str = "https://zenodo.org/records/19217442/files/clean_spectra.mgf"
GNPS_FILENAME = "GNPS.mgf"


@Cache(
    validity_duration=VALIDITY_DURATION,
)
def download_gnps(output_dir: str) -> pd.DataFrame:
    output = Path(output_dir)
    if output.exists():
        output.unlink()

    downloader = BaseDownloader(auto_extract=False)
    return downloader.download(
        GNPS_URL,
        str(output),
    )


@Cache(validity_duration=VALIDITY_DURATION)
def load_gnps(file_name: str) -> list[Spectrum]:
    _ = download_gnps(file_name)
    spectra = []
    for spectrum in tqdm(
        load_from_mgf(file_name),
        desc="Loading GNPS spectra",
        leave=False,
    ):
        spectrum = Spectrum(
            mz=spectrum.mz,
            intensities=spectrum.intensities,
            metadata=spectrum.metadata,
        )
        spectra.append(spectrum)
    return spectra
