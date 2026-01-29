from pathlib import Path

import pandas as pd
from downloaders import BaseDownloader
from dreams.utils.data import MSData

from ms2mol_evaluation.evaluation import Evaluation


class GNPSEvaluation(Evaluation):
    def __init__(self, output_dir: Path | str) -> None:
        super().__init__(output_dir)

    @staticmethod
    def load_gnps(in_memory: bool = True, **kwargs) -> MSData:
        path: Path = Path(GNPSEvaluation.download_gnps().destination.values[0])
        h5_path = path.with_suffix(".hdf5")
        if h5_path.exists():
            # If the HDF5 already exists, load it directly (this is fastest)
            return MSData.from_hdf5(h5_path, in_mem=in_memory)
        else:
            return MSData.load(path, in_mem=in_memory, **kwargs)

    @staticmethod
    def download_gnps(auto_extract: bool = False) -> pd.DataFrame:
        return BaseDownloader(auto_extract=auto_extract).download(
            "https://external.gnps2.org/processed_gnps_data/matchms.mgf",
            str((Path("downloads/gnps") / f"gnps_matchms_filtered.mgf").resolve()),
        )
