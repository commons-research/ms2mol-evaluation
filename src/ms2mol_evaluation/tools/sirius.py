import os
import subprocess
import typing as T
from pathlib import Path
from typing import cast

import pandas as pd
from dotenv import load_dotenv
from matchms import Spectrum as MatchMSSpectrum
from matchms.exporting import save_as_mgf

from ms2mol_evaluation.evaluation import Evaluation
from ms2mol_evaluation.spectrum import Spectrum

load_dotenv()


class SiriusEvaluation(Evaluation):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.sirius_executable = os.getenv("SIRIUS_PATH")
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=True)
        self._add_required_metadata_for_sirius()
        self._split_orbitrap_qtof()
        self._save_mgf_file(self.msg_orbitrap, self.output_dir / "sirius_orbitrap.mgf")
        self._save_mgf_file(self.msg_qtof, self.output_dir / "sirius_qtof.mgf")
        self._write_custom_db()
        self._create_custom_db()

    def _save_mgf_file(self, spectra: T.List[Spectrum], file_path: Path) -> None:
        save_as_mgf(spectra, str(file_path), file_mode="w")

    def _add_required_metadata_for_sirius(self) -> None:
        for s in self.msg_spectra:
            s.set("ms_level", 2)
            s.set("formula", None)
            s.set("precursor_formula", None)
            s.set("feature_id", s.get("identifier"))

    def _split_orbitrap_qtof(self) -> None:
        self.msg_orbitrap = [
            s for s in self.msg_spectra if s.get("instrument_type") == "Orbitrap"
        ]
        self.msg_qtof = [
            s for s in self.msg_spectra if s.get("instrument_type") == "QTOF"
        ]

    def _create_dataframe_from_spectra_list(
        self,
    ) -> pd.DataFrame:
        smiles = [s.get("smiles") for s in self.isdb_spectra]
        inchikey = [s.get("compound_name") for s in self.isdb_spectra]

        df = pd.DataFrame({"smiles": smiles, "name": inchikey})
        return df

    def _write_custom_db(self) -> None:
        df_for_custom_db = self._create_dataframe_from_spectra_list()
        self.df_path = self.output_dir / "sirius_custom_db.tsv"
        df_for_custom_db.to_csv(
            self.df_path,
            index=False,
            sep="\t",
        )

    def _create_custom_db(self) -> None:
        create_command = [
            self.sirius_executable,
            "custom-db",
            "create",
            "--name=lotusISDB",
            f"--location={str((self.output_dir / "lotusISDB.siriusdb").resolve())}",
        ]

        import_command = [
            self.sirius_executable,
            "custom-db",
            "import",
            "--db=lotusISDB",
            f"{self.df_path.resolve()}",
        ]
        subprocess.run(create_command, check=True)
        subprocess.run(import_command, check=True)

    def run_eval(self) -> pd.DataFrame:
        raise NotImplementedError("SiriusEvaluation is not implemented yet.")
