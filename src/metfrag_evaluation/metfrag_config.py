import os
import typing as T
from pathlib import Path

from dict_hash import Hashable, sha256
from dotenv import load_dotenv

from metfrag_evaluation.spectrum import Spectrum

load_dotenv()
ADDUCTS_TO_VALUE = {"[M+H]+": 1, "[M+Na]+": 23}


class MetFragConfig(Hashable):
    def __init__(
        self,
        precursor_mass: float,
        adduct_type: str,
        peak_list_file: T.Union[str, Path],
        results_path: T.Union[str, Path],
        results_file: T.Union[str, Path],
        database_type="Postgres",
        config_params: T.Optional[T.Dict[str, T.Any]] = None,
    ):
        if adduct_type not in ADDUCTS_TO_VALUE:
            raise ValueError(
                f"Invalid adduct type: {adduct_type}. Must be one of {list(ADDUCTS_TO_VALUE.keys())}."
            )

        adduct_to_int = ADDUCTS_TO_VALUE[adduct_type]
        self._database_type = database_type
        self._universal_params = {
            "PrecursorIonMode": adduct_to_int,
            "FragmentPeakMatchRelativeMassDeviation": 5.0,
            "SampleName": str(results_file),
            "MetFragCandidateWriter": "CSV",
            "DatabaseSearchRelativeMassDeviation": 10.0,
            "FragmentPeakMatchAbsoluteMassDeviation": 0.001,
            "ResultsPath": "metfrag_results",
            "IonizedPrecursorMass": precursor_mass,
            "MetFragScoreTypes": "FragmenterScore",
            "MetFragScoreWeights": 1.0,
            "MetFragPreProcessingCandidateFilter": "UnconnectedCompoundFilter",
            "MetFragPostProcessingCandidateFilter": "InChIKeyFilter",
            "IsPositiveIonMode": True,
            "MaximumTreeDepth": 2,
            "NumberThreads": 1,
            "UseSmiles": True,
            "PeakListPath": str(peak_list_file),
            "ResultsPath": str(results_path),
        }
        self._db_specific_params = {}
        self.set_database_specific_defaults()
        if config_params:
            if not isinstance(config_params, dict):
                raise TypeError("config_params must be a dictionary.")
            for key, value in config_params.items():
                if key in self._universal_params:
                    self._universal_params[key] = value
                elif key in self._db_specific_params:
                    self._db_specific_params[key] = value
                else:
                    raise KeyError(f"Unknown config key: {key}")

    @property
    def database_type(self):
        return self._database_type

    @database_type.setter
    def database_type(self, value):
        self._database_type = value
        self.set_database_specific_defaults()

    def set_database_specific_defaults(self):
        if self._database_type == "Postgres":
            self._db_specific_params = {
                "LocalDatabase": os.getenv("LOTUS_DB_PGDATABASE"),
                "LocalDatabaseCompoundsTable": "lotus",
                "LocalDatabasePortNumber": os.getenv("LOTUS_DB_PGPORT"),
                "LocalDatabaseServerIp": os.getenv("LOTUS_DB_PGHOST"),
                "LocalDatabaseUser": os.getenv("LOTUS_DB_POSTGRES_USER"),
                "LocalDatabasePassword": os.getenv("LOTUS_DB_POSTGRES_PASSWORD"),
                "LocalDatabaseMassColumn": "monoisotopic_mass",
                "LocalDatabaseFormulaColumn": "formula",
                "LocalDatabaseInChIColumn": "inchi",
                "LocalDatabaseInChIKey1Column": "inchikey_1",
                "LocalDatabaseInChIKey2Column": "inchikey_2",
                "LocalDatabaseCidColumn": "identifier",
                "LocalDatabaseSmilesColumn": "smiles",
                "LocalDatabaseCompoundNameColumn": "name",
            }
        else:
            raise NotImplementedError(
                f"Database type '{self._database_type}' is not implemented."
            )

    def to_config_string(self):
        lines = [f"MetFragDatabaseType = {self._database_type}"]
        for key, val in self._universal_params.items():
            lines.append(f"{key} = {val}")
        for key, val in self._db_specific_params.items():
            lines.append(f"{key} = {val}")
        return "\n".join(lines)

    def __str__(self):
        return self.to_config_string()

    def get_param(self, key):
        return self._universal_params.get(key) or self._db_specific_params.get(key)

    def set_param(self, key, value):
        if key in self._universal_params:
            self._universal_params[key] = value
        elif key in self._db_specific_params:
            self._db_specific_params[key] = value
        else:
            raise KeyError(f"Unknown config key: {key}")

    @staticmethod
    def _merge_dicts(*dict_args) -> dict:
        """
        Given any number of dictionaries, shallow copy and merge into a new dict,
        precedence goes to key-value pairs in latter dictionaries.
        """
        result = {}
        for dictionary in dict_args:
            result.update(dictionary)
        return result

    def consistent_hash(self, use_approximation=False) -> str:
        return sha256(
            {
                key: value
                for key, value in MetFragConfig._merge_dicts(
                    self._universal_params, self._db_specific_params
                ).items()
            },
            use_approximation=use_approximation,
        )
