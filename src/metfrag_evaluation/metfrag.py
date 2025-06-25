import subprocess
import typing as T
from pathlib import Path

import pandas as pd
from cache_decorator import Cache

from metfrag_evaluation.metfrag_config import MetFragConfig
from metfrag_evaluation.spectrum import Spectrum


def write_metfrag_config(config: "MetFragConfig") -> str:
    """
    Run MetFrag on a given spectrum with the provided configuration.

    Args:
        spectrum (Spectrum): The spectrum to analyze.
        config (MetFragConfig): Configuration for MetFrag analysis.
    """

    config_file_name = config.consistent_hash(use_approximation=False) + ".cgf"
    with open(config_file_name, "w") as config_file:
        config_file.write(config.to_config_string())

    return config_file_name


@Cache(
    "./data/metfrag_cache/{_hash}/spectrum_hash.txt",
)
def get_spectrum_hash(spectrum: Spectrum, use_approximation=True) -> str:
    return spectrum.consistent_hash(use_approximation=use_approximation)


def spectrum_to_metfrag(
    spectrum: Spectrum,
    config_params: T.Optional[T.Dict[str, T.Any]] = None,
) -> T.Tuple[str, "MetFragConfig"]:
    peak_list_file = (
        Path(
            Cache.compute_path(get_spectrum_hash, spectrum, use_approximation=True)
        ).parent
        / "peak_list.txt"
    )
    peak_list_file.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(spectrum.peaks.to_numpy).to_csv(
        str(peak_list_file),
        sep="\t",
        header=False,
        index=False,
    )
    config = MetFragConfig(
        spectrum.get("precursor_mz"),
        spectrum.get("adduct"),
        peak_list_file=peak_list_file,
        results_path=peak_list_file.parent,
        results_file="results",
        config_params=config_params,
    )

    config_file = write_metfrag_config(config)
    return config_file, config


def run_metfrag(
    spectrum: Spectrum,
    config_params: T.Optional[T.Dict[str, T.Any]] = None,
) -> T.Tuple[str, "MetFragConfig"]:
    """
    Run MetFrag on a given spectrum with the provided configuration.

    Args:
        spectrum (Spectrum): The spectrum to analyze.
        config_params (dict, optional): Additional configuration parameters for MetFrag.

    Returns:
        tuple: A tuple containing the path to the MetFrag configuration file and the MetFragConfig object.
    """
    config_file, config = spectrum_to_metfrag(spectrum, config_params)

    command = [
        "java",
        "-jar",
        "MetFragCommandLine-2.6.6.jar",
        config_file,
    ]

    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
    )

    # once the process is done, we can delete the config file
    Path(config_file).unlink(missing_ok=True)
    return config_file, config
