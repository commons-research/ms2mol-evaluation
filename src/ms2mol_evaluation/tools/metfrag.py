import os
import subprocess
import typing as T
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import psycopg
from dotenv import load_dotenv
from downloaders import BaseDownloader
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from ms2mol_evaluation.databases.lotus import (
    create_lotus_table_query,
    generate_index_query,
    generate_insert_query,
)
from ms2mol_evaluation.evaluation import Evaluation
from ms2mol_evaluation.metfrag.metfrag_config import MetFragConfig
from ms2mol_evaluation.spectrum import Spectrum

load_dotenv()


class MetFragEvaluation(Evaluation):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.metfrag_executable: Path = Path(
            MetFragEvaluation.download_metfrag_exec().destination.values[0]
        )
        self.msg_spectra = self._filter_massspecgym_spectra(hydrogen_adduct_only=False)

    @staticmethod
    def download_metfrag_exec(
        version: str = "2.6.6",
        auto_extract: bool = False,
    ) -> pd.DataFrame:
        return BaseDownloader(auto_extract=auto_extract).download(
            f"https://github.com/ipb-halle/MetFragRelaunched/releases/download/v{version}/MetFragCommandLine-{version}.jar",
            str(
                (
                    Path("downloads/metfrag") / f"MetFragCommandLine-{version}.jar"
                ).resolve()
            ),
        )

    def run_metfrag(
        self,
        spectrum: Spectrum,
        config_params: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> T.Tuple[Path, "MetFragConfig", pd.DataFrame]:
        """
        Run MetFrag on a given spectrum with the provided configuration, or load results if they already exist.

        Args:
            spectrum (Spectrum): The spectrum to analyze.
            config_params (dict, optional): Additional configuration parameters for MetFrag.

        Returns:
            tuple: A tuple containing the path to the MetFrag configuration file, the MetFragConfig object, and the results DataFrame.
        """
        config_file, config = self.create_metfrag_config(spectrum, config_params)

        # Determine expected results CSV path
        results_csv = (
            Path(config.get_results_path()) / f"{config.get_results_file()}.csv"
        )
        if results_csv.exists() and not pd.read_csv(results_csv).empty:
            # Results already exist, skip running MetFrag
            config_file.unlink(missing_ok=True)
            return config_file, config, pd.read_csv(results_csv)

        command = [
            "java",
            "-jar",
            self.metfrag_executable.resolve(),
            str(config_file),
        ]

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
        )

        # once the process is done, we can delete the config file
        config_file.unlink(missing_ok=True)
        return config_file, config, pd.read_csv(results_csv)

    def write_metfrag_config(self, config: "MetFragConfig") -> Path:
        """
        Run MetFrag on a given spectrum with the provided configuration.

        Args:
            spectrum (Spectrum): The spectrum to analyze.
            config (MetFragConfig): Configuration for MetFrag analysis.
        """

        config_file_name = config.consistent_hash(use_approximation=False) + ".cgf"
        config_file_name = self.output_dir / config_file_name
        with open(str(config_file_name), "w") as config_file:
            config_file.write(config.to_config_string())

        return config_file_name

    def get_spectrum_hash(
        self,
        spectrum: Spectrum,
        use_approximation=False,
    ) -> str:
        return spectrum.consistent_hash(use_approximation=use_approximation)

    def create_metfrag_config(
        self,
        spectrum: Spectrum,
        config_params: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> T.Tuple[Path, "MetFragConfig"]:
        # Step 1: Compute spectrum hash
        spectrum_hash = self.get_spectrum_hash(spectrum, use_approximation=False)

        # Step 2: Create a temporary config to compute config hash
        temp_peak_list_file = Path(
            f"cache/peak_list_{spectrum_hash}.txt"
        )  # dummy path for hash computation
        temp_config = MetFragConfig(
            spectrum.get("precursor_mz"),
            spectrum.get("adduct"),
            peak_list_file=temp_peak_list_file,
            results_path="cache",  # dummy path
            results_file="results",
            config_params=config_params,
        )
        config_hash = temp_config.consistent_hash(use_approximation=False)

        # Step 3: Combine hashes for directory
        combined_dir = self.output_dir / f"metfrag_cache/{spectrum_hash}_{config_hash}"
        combined_dir.mkdir(parents=True, exist_ok=True)
        peak_list_file = combined_dir / "peak_list.txt"

        # Step 4: Write peak list to the new directory
        pd.DataFrame(spectrum.peaks.to_numpy).to_csv(
            str(peak_list_file),
            sep="\t",
            header=False,
            index=False,
        )

        # Step 5: Create the final config with correct paths
        config = MetFragConfig(
            spectrum.get("precursor_mz"),
            spectrum.get("adduct"),
            peak_list_file=peak_list_file,
            results_path=combined_dir,
            results_file="results",
            config_params=config_params,
        )

        config_file = self.write_metfrag_config(config)
        return config_file, config

    def run_eval(self, n_jobs: int) -> List[pd.DataFrame]:
        self._create_postgres_db()
        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(self.run_metfrag)(spectrum)
            for spectrum in tqdm(self.msg_spectra, desc="Running metfrag")
        )

        resulting_dataframes = [i[2] for i in results]
        del results
        return resulting_dataframes

    def concatenate_results(self, df_list: List[pd.DataFrame]) -> None:
        assert len(df_list) == len(self.msg_spectra), "Length of df_list must match number of spectra"
        for df,s in zip(df_list, self.msg_spectra):
            if df.empty:
                continue
            df["identifier"] = s.get("identifier")
            df["true_inchikey"] = s.get("inchikey")
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(self.output_dir / "combined_results.csv", index=False)

    def _create_scores_array(
        self,
        df_list: List[pd.DataFrame],
    ) -> Tuple[List[str], List[str], Dict[str, str]]:
        index: List[str] = [s.get("identifier") for s in self.msg_spectra]
        identifier_to_inchikey = {
            s.get("identifier"): s.get("inchikey") for s in self.msg_spectra
        }
        all_inchikeys = sorted(set(s.get("compound_name") for s in self.isdb_spectra))
        inchi_to_int = {inchk: i for i, inchk in enumerate(all_inchikeys)}
        self.scores = np.empty(
            (len(self.msg_spectra), len(all_inchikeys)),
            dtype=np.float16,
        )
        self.scores.fill(np.nan)
        for i, result in tqdm(
            enumerate(df_list),
            total=len(self.msg_spectra),
            desc="Filling scores",
        ):
            if result.empty:
                continue

            for inchikey, score in zip(
                result["InChIKey1"].values, result["Score"].values
            ):
                if inchikey in inchi_to_int:
                    self.scores[i][inchi_to_int[inchikey]] = score

        return (
            all_inchikeys,
            index,
            identifier_to_inchikey,
        )

    def _isdb_as_df(self) -> pd.DataFrame:
        """
        Loads the ISDB dataset formated as a DataFrame suitable for MetFrag.

        Returns:
            pd.DataFrame: DataFrame containing ISDB data.
        """
        identifier = [s.get("compound_name") for s in self.isdb_spectra]
        inchi = [s.get("inchi") for s in self.isdb_spectra]
        exact_mass = [s.get("parent_mass") for s in self.isdb_spectra]
        molecular_formula = [s.get("molecular_formula") for s in self.isdb_spectra]
        inchikey_1 = [s.get("compound_name") for s in self.isdb_spectra]
        inchikey_2 = [s.get("inchikey").split("-")[1] for s in self.isdb_spectra]
        inchikey_3 = [s.get("inchikey").split("-")[2] for s in self.isdb_spectra]
        smiles = [s.get("smiles") for s in self.isdb_spectra]
        name = [s.get("compound_name") for s in self.isdb_spectra]

        lotus_db = (
            pd.DataFrame(
                {
                    "Identifier": identifier,
                    "InChI": inchi,
                    "MonoisotopicMass": exact_mass,
                    "MolecularFormula": molecular_formula,
                    "InChIKey1": inchikey_1,
                    "InChIKey2": inchikey_2,
                    "SMILES": smiles,
                    "Name": name,
                    "InChIKey3": inchikey_3,
                }
            )
            .drop_duplicates("InChIKey1")
            .reset_index(drop=True)
        )

        return lotus_db

    def _create_postgres_db(self) -> None:
        # we want to run docker compose up -d to start the postgres database
        subprocess.run(
            ["docker", "compose", "-f", "docker-compose.yml", "up", "-d"],
            check=True,
        )

        conn = psycopg.connect(
            dbname=os.getenv("LOTUS_DB_PGDATABASE"),
            host=os.getenv("LOTUS_DB_PGHOST"),
            port=os.getenv("LOTUS_DB_PGPORT"),
            user=os.getenv("LOTUS_DB_POSTGRES_USER"),
            password=os.getenv("LOTUS_DB_POSTGRES_PASSWORD"),
        )
        conn.autocommit = True
        cursor = conn.cursor()
        create_table_query = create_lotus_table_query()
        cursor.execute(create_table_query)
        insert_query = generate_insert_query()
        df = self._isdb_as_df()
        data = df.values.tolist()
        batch_size = 10000
        for i in tqdm(
            range(0, len(data), batch_size),
            desc="Inserting data into PostgreSQL",
        ):
            batch = data[i : i + batch_size]
            cursor.executemany(insert_query, batch)

        index_query = generate_index_query()
        cursor.execute(index_query)

    def get_fraction_results(
        self,
        df_list: List[pd.DataFrame],
        interval: Iterable[float] = np.arange(0.0, 1.0, 0.05),
    ) -> None:
        (
            all_inchikeys,
            index,
            identifier_to_inchikey,
        ) = self._create_scores_array(df_list=df_list)

        y_fraction, x_fraction = self.evaluate_fraction(
            all_inchikeys=all_inchikeys,
            identifiers=index,
            identifier_to_inchikey=identifier_to_inchikey,
            interval=interval,
        )

        self.plot_results(x_fraction, y_fraction, interval, image_name="fraction.png")

    def get_top_n_results(
        self,
        df_list: List[pd.DataFrame],
        interval: Iterable[int] = [1, 2, 5, 10, 20, 50, 100, 200, 500],
    ) -> None:
        (
            all_inchikeys,
            index,
            identifier_to_inchikey,
        ) = self._create_scores_array(df_list)
        y_top_n, x_top_n = self.evaluate_top_n(
            all_inchikeys=all_inchikeys,
            identifiers=index,
            identifier_to_inchikey=identifier_to_inchikey,
            interval=interval,
        )
        self.plot_results(x_top_n, y_top_n, interval, image_name="top_n.png")
