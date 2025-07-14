import argparse
import typing as T

import pandas as pd
from downloaders import BaseDownloader
from joblib import Parallel, delayed
from tqdm import tqdm

from ms2mol_evaluation.isdb import download_isdb, filter_massspecgym_spectra, load_isdb
from ms2mol_evaluation.massspecgym import load_massspecgym, to_spectra
from ms2mol_evaluation.metfrag import run_metfrag
from ms2mol_evaluation.spectrum import Spectrum
from ms2mol_evaluation.utils import (
    analyze_results,
    convert_evaluation_results,
    generate_full_results,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run MetFrag evaluation with configurable CPU usage."
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPUs to use (default: all available)",
    )
    args = parser.parse_args()
    _ = BaseDownloader(auto_extract=False).download(
        "https://github.com/ipb-halle/MetFragRelaunched/releases/download/v2.6.6/MetFragCommandLine-2.6.6.jar",
        "MetFragCommandLine-2.6.6.jar",
    )

    massspecgym = load_massspecgym()
    spectra: T.List[Spectrum] = to_spectra(massspecgym)
    download_isdb()
    isdb: T.List[Spectrum] = load_isdb()
    spectra = filter_massspecgym_spectra(spectra, isdb, hydrogen_adduct_only=False)

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_metfrag)(spectrum) for spectrum in tqdm(spectra)
    )

    # we now check the top 1, 5, 10 and 20 results
    # we also want to check if there is a difference between H adduct or Na adduct
    # we also want to check if there is a difference between orbitrap and qtof
    resulting_dataframes = [i[2] for i in results]

    res = analyze_results(spectra, resulting_dataframes)
    metrics: T.Dict[str, int] = res.pop("metrics")
    for key, value in metrics.items():
        if "_h" in key:
            metrics[key] = metrics[key] / res["n_spectrum_h"]
        elif "_na" in key:
            metrics[key] = metrics[key] / res["n_spectrum_na"]
        elif "_orbitrap" in key:
            metrics[key] = metrics[key] / res["n_spectrum_orbitrap"]
        elif "_qtof" in key:
            metrics[key] = metrics[key] / res["n_spectrum_qtof"]
        else:
            metrics[key] = metrics[key] / res["n_total"]

    out_df = pd.DataFrame.from_dict(
        metrics,
        orient="index",
    ).T

    convert_evaluation_results(out_df).to_csv(
        "lotus_metfrag_top_n.csv",
    )

    df = generate_full_results(spectra, resulting_dataframes)
    df.to_csv("lotus_metfrag_scores.csv", index=False)


if __name__ == "__main__":
    main()
