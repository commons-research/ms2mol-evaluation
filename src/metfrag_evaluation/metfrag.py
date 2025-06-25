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


def get_spectrum_hash(spectrum: Spectrum, use_approximation=False) -> str:
    return spectrum.consistent_hash(use_approximation=use_approximation)


def create_metfrag_config(
    spectrum: Spectrum,
    config_params: T.Optional[T.Dict[str, T.Any]] = None,
) -> T.Tuple[str, "MetFragConfig"]:
    # Step 1: Compute spectrum hash
    spectrum_hash = get_spectrum_hash(spectrum, use_approximation=False)

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
    combined_dir = Path(f"data/metfrag_cache/{spectrum_hash}_{config_hash}")
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

    config_file = write_metfrag_config(config)
    return config_file, config


def run_metfrag(
    spectrum: Spectrum,
    config_params: T.Optional[T.Dict[str, T.Any]] = None,
) -> T.Tuple[str, "MetFragConfig", pd.DataFrame]:
    """
    Run MetFrag on a given spectrum with the provided configuration, or load results if they already exist.

    Args:
        spectrum (Spectrum): The spectrum to analyze.
        config_params (dict, optional): Additional configuration parameters for MetFrag.

    Returns:
        tuple: A tuple containing the path to the MetFrag configuration file, the MetFragConfig object, and the results DataFrame.
    """
    config_file, config = create_metfrag_config(spectrum, config_params)

    # Determine expected results CSV path
    results_csv = Path(config.get_results_path()) / f"{config.get_results_file()}.csv"
    if results_csv.exists() and not pd.read_csv(results_csv).empty:
        # Results already exist, skip running MetFrag
        Path(config_file).unlink(missing_ok=True)
        return config_file, config, pd.read_csv(results_csv)

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
    return config_file, config, pd.read_csv(results_csv)
