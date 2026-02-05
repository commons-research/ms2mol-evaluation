import os
import subprocess
import typing as T
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matchms import Spectrum as MatchMSSpectrum
from matchms.exporting import save_as_mgf
from tqdm.auto import tqdm

from ms2mol_evaluation.evaluation import Evaluation
from ms2mol_evaluation.sirius.constants import (
    SIRIUS_ORBITRAP_COMMAND,
    SIRIUS_QTOF_COMMAND,
)
from ms2mol_evaluation.spectrum import Spectrum

load_dotenv()


class SiriusEvaluation(Evaluation):
    def __init__(self, output_dir: Union[Path, str]) -> None:
        super().__init__(output_dir)
        self.sirius_executable = os.getenv("SIRIUS_PATH")
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=False)
        self._add_required_metadata_for_sirius()
        self._split_orbitrap_qtof()
        self.orbitrap_mgf_path = self.output_dir / "sirius_orbitrap.mgf"
        self.qtof_mgf_path = self.output_dir / "sirius_qtof.mgf"
        self._save_mgf_file(self.msg_orbitrap, self.orbitrap_mgf_path)
        self._save_mgf_file(self.msg_qtof, self.qtof_mgf_path)
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
            # only positive charges in current version of MassSpecGym
            s.set("charge", "1+")
            s.set("pepmass", s.get("precursor_mz"))

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
            f"--location={str((self.output_dir / 'lotusISDB.siriusdb').resolve())}",
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

    def _create_command(self, mgf_path: Path, is_orbitrap: bool) -> T.List[str]:
        sirius_command = []
        sirius_command.append(self.sirius_executable)
        sirius_command.append("--input")
        sirius_command.append(str(mgf_path.resolve()))
        sirius_command.append("--output")
        sirius_command.append(str(mgf_path.resolve().with_suffix(".sirius")))
        # we take the sirius orbitrap constant and split it by spaces
        if is_orbitrap:
            sirius_command.extend(SIRIUS_ORBITRAP_COMMAND.split())
        else:
            sirius_command.extend(SIRIUS_QTOF_COMMAND.split())
        return sirius_command

    def run_eval(self) -> pd.DataFrame:
        orbi_command = self._create_command(self.orbitrap_mgf_path, True)
        qtof_command = self._create_command(self.qtof_mgf_path, False)

        subprocess.run(orbi_command, check=True)
        subprocess.run(qtof_command, check=True)

        sirius_orbi = pd.read_csv(
            self.orbitrap_mgf_path.resolve().with_suffix("")
            / "structure_identifications_all.tsv",
            sep="\t",
        )
        sirius_orbi["instrument_type"] = "Orbitrap"
        sirius_qtof = pd.read_csv(
            self.qtof_mgf_path.resolve().with_suffix("")
            / "structure_identifications_all.tsv",
            sep="\t",
        )
        sirius_qtof["instrument_type"] = "QTOF"
        return pd.concat([sirius_orbi, sirius_qtof], ignore_index=True)

    def _create_scores_array(
        self,
        df: pd.DataFrame,
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        mass_spec_gym = self._load_massspecgym()
        identifier_to_inchikey = {}
        for msg_id, msg_inchikey in zip(mass_spec_gym.index, mass_spec_gym.inchikey):
            identifier_to_inchikey[msg_id] = msg_inchikey

        del mass_spec_gym

        df["true_inchikey"] = df["mappingFeatureId"].map(identifier_to_inchikey)
        index: T.List[str] = [s.get("identifier") for s in self.msg_spectra]
        identifier_to_inchikey = {
            s.get("identifier"): s.get("inchikey") for s in self.msg_spectra
        }
        id_to_int = {s.get("identifier"): i for i, s in enumerate(self.msg_spectra)}
        all_inchikeys = sorted(set(s.get("compound_name") for s in self.isdb_spectra))
        inchi_to_int = {inchk: i for i, inchk in enumerate(all_inchikeys)}

        self.scores = np.empty(
            (len(index), len(all_inchikeys)),
            dtype=np.float16,
        )
        self.scores.fill(np.nan)

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Filling scores"):
            identifier = row["mappingFeatureId"]
            inchikey = row["InChIkey2D"]
            if identifier not in id_to_int or inchikey not in inchi_to_int:
                continue
            self.scores[id_to_int[identifier], inchi_to_int[inchikey]] = row[
                "CSI:FingerIDScore"
            ]

        return (
            all_inchikeys,
            index,
            identifier_to_inchikey,
        )

    def get_fraction_results(
        self,
        df: pd.DataFrame,
        interval: Iterable[float] = [
            -2000,
            -1000,
            -500,
            -200,
            -100,
            -50,
            -20,
            -10,
            -5,
            0,
        ],
    ) -> None:
        (
            all_inchikeys,
            index,
            identifier_to_inchikey,
        ) = self._create_scores_array(df)

        y_fraction, x_fraction = self.evaluate_fraction(
            all_inchikeys=all_inchikeys,
            identifiers=index,
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
            index,
            identifier_to_inchikey,
        ) = self._create_scores_array(df)
        y_top_n, x_top_n = self.evaluate_top_n(
            all_inchikeys=all_inchikeys,
            identifiers=index,
            identifier_to_inchikey=identifier_to_inchikey,
            interval=interval,
        )
        self.plot_results(x_top_n, y_top_n, interval, image_name="top_n.png")
