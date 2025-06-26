import argparse
import random
import typing as T
import pandas as pd
from downloaders import BaseDownloader
from joblib import Parallel, delayed
from tqdm import tqdm

from metfrag_evaluation.massspecgym import load_massspecgym, to_spectra
from metfrag_evaluation.metfrag import run_metfrag
from metfrag_evaluation.spectrum import Spectrum


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
    spectra: Spectrum = to_spectra(massspecgym)
    lotus = pd.read_csv("data/lotus/230106_frozen_metadata.csv.gz", compression="gzip")
    lotus["structure_inchikey_1"] = lotus["structure_inchikey"].apply(
        lambda x: x.split("-")[0]
    )
    lotus.drop_duplicates(subset=["structure_inchikey_1"], inplace=True)

    inchikeys = set(lotus["structure_inchikey_1"].values)

    spectra = [i for i in tqdm(spectra, leave=False) if i.get("inchikey") in inchikeys]
    random.Random(42).shuffle(spectra)

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_metfrag)(spectrum) for spectrum in tqdm(spectra)
    )

    # we now check the top 1, 5, 10 and 20 results
    # we also want to check if there is a difference between H adduct or Na adduct
    # we also want to check if there is a difference between orbitrap and qtof
    resulting_dataframes = [i[2] for i in results]

    res = analyze_results(spectra, resulting_dataframes)
    metrics = res.pop("metrics")


def analyze_results(
    spectra: T.List[Spectrum],
    results: T.List[pd.DataFrame],
) -> T.Dict[str, T.Union[int, T.Dict[str, int]]]:
    metrics = {
        "top_1": 0,
        "top_5": 0,
        "top_10": 0,
        "top_20": 0,
        "top_1_h": 0,
        "top_5_h": 0,
        "top_10_h": 0,
        "top_20_h": 0,
        "top_1_na": 0,
        "top_5_na": 0,
        "top_10_na": 0,
        "top_20_na": 0,
        "top_1_orbitrap": 0,
        "top_5_orbitrap": 0,
        "top_10_orbitrap": 0,
        "top_20_orbitrap": 0,
        "top_1_qtof": 0,
        "top_5_qtof": 0,
        "top_10_qtof": 0,
        "top_20_qtof": 0,
    }

    n_spectrum_orbitrap = 0
    n_spectrum_qtof = 0
    n_spectrum_h = 0
    n_spectrum_na = 0
    for spectrum, res in tqdm(zip(spectra, results), desc="Analyzing results"):
        # get the metadata from the spectrum
        inchikey:str = spectrum.get("inchikey")
        instrument_type = spectrum.get("instrument_type")
        adduct = spectrum.get("adduct")

        # add to the total counts
        if instrument_type == "Orbitrap":
            n_spectrum_orbitrap += 1
        elif instrument_type == "QTOF":
            n_spectrum_qtof += 1
        if adduct == "[M+H]+":
            n_spectrum_h += 1
        elif adduct == "[M+Na]+":
            n_spectrum_na += 1

        if res.empty:
            continue

        df_1 = res.iloc[:1]
        df_5 = res.iloc[:5]
        df_10 = res.iloc[:10]
        df_20 = res.iloc[:20]

        if inchikey in df_1["InChIKey1"].values:
            metrics["top_1"] += 1
            if adduct == "[M+H]+":
                metrics["top_1_h"] += 1
            elif adduct == "[M+Na]+":
                metrics["top_1_na"] += 1
            if instrument_type == "Orbitrap":
                metrics["top_1_orbitrap"] += 1
            elif instrument_type == "QTOF":
                metrics["top_1_qtof"] += 1

        if inchikey in df_5["InChIKey1"].values:
            metrics["top_5"] += 1
            if adduct == "[M+H]+":
                metrics["top_5_h"] += 1
            elif adduct == "[M+Na]+":
                metrics["top_5_na"] += 1
            if instrument_type == "Orbitrap":
                metrics["top_5_orbitrap"] += 1
            elif instrument_type == "QTOF":
                metrics["top_5_qtof"] += 1

        if inchikey in df_10["InChIKey1"].values:
            metrics["top_10"] += 1
            if adduct == "[M+H]+":
                metrics["top_10_h"] += 1
            elif adduct == "[M+Na]+":
                metrics["top_10_na"] += 1
            if instrument_type == "Orbitrap":
                metrics["top_10_orbitrap"] += 1
            elif instrument_type == "QTOF":
                metrics["top_10_qtof"] += 1

        if inchikey in df_20["InChIKey1"].values:
            metrics["top_20"] += 1
            if adduct == "[M+H]+":
                metrics["top_20_h"] += 1
            elif adduct == "[M+Na]+":
                metrics["top_20_na"] += 1
            if instrument_type == "Orbitrap":
                metrics["top_20_orbitrap"] += 1
            elif instrument_type == "QTOF":
                metrics["top_20_qtof"] += 1

    return {
        "metrics": metrics,
        "n_total": len(spectra),
        "n_spectrum_orbitrap": n_spectrum_orbitrap,
        "n_spectrum_qtof": n_spectrum_qtof,
        "n_spectrum_h": n_spectrum_h,
        "n_spectrum_na": n_spectrum_na,
    }


if __name__ == "__main__":
    main()
