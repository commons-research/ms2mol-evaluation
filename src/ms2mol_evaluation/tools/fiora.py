import subprocess
from pathlib import Path
from typing import List

import pandas as pd
from cache_decorator import Cache
from matchms.filtering import (
    default_filters,
    derive_inchi_from_smiles,
    derive_inchikey_from_inchi,
)
from matchms.importing import load_from_mgf
from tqdm.auto import tqdm

from ms2mol_evaluation.evaluation import Evaluation
from ms2mol_evaluation.spectrum import Spectrum


class FioraEvaluation(Evaluation):
    def __init__(self, output_dir: Path | str) -> None:
        super().__init__(output_dir)
        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=True)
        self.fiora_df = self._create_fiora_df()
        self.isdb_spectra = FioraEvaluation._create_fiora_isdb(
            self.fiora_df, str(self.output_dir)
        )

    def _create_fiora_df(self) -> pd.DataFrame:
        smiles = [s.get("smiles") for s in self.isdb_spectra]
        name = [s.get("compound_name") for s in self.isdb_spectra]
        precursor_type = ["[M+H]+"] * len(self.isdb_spectra)
        collision_energy = [25.0] * len(self.isdb_spectra)
        instrument_type = ["HCD"] * len(self.isdb_spectra)

        fiora_df = pd.DataFrame(
            {
                "Name": name,
                "SMILES": smiles,
                "Precursor_type": precursor_type,
                "CE": collision_energy,
                "Instrument_type": instrument_type,
            }
        )

        return fiora_df

    @staticmethod
    @Cache(use_approximated_hash=True)
    def _create_fiora_isdb(df: pd.DataFrame, output_dir: str) -> List[Spectrum]:
        dir = Path(output_dir)
        df_path = dir / "fiora_input.csv"
        df.to_csv(df_path, index=False)
        mgf_filename = df_path.resolve().with_suffix(".mgf")

        subprocess.run(
            [
                "uv",
                "run",
                "fiora-predict",
                "-i",
                str(df_path.resolve()),
                "-o",
                str(mgf_filename),
            ]
        )

        spectra = []
        for spectrum in tqdm(
            load_from_mgf(mgf_filename),
            desc="Loading ISDB spectra",
            leave=False,
        ):
            spectrum = default_filters(spectrum)
            spectrum = derive_inchi_from_smiles(spectrum)
            spectrum = derive_inchikey_from_inchi(spectrum)
            spectrum.set("inchi", spectrum.get("title"))
            spectrum = Spectrum(
                mz=spectrum.mz,
                intensities=spectrum.intensities,
                metadata=spectrum.metadata,
            )
            spectra.append(spectrum)

        return spectra
