import typing as T

import pandas as pd
from tqdm import tqdm

from ms2mol_evaluation.spectrum import Spectrum


def convert_evaluation_results(df: pd.DataFrame) -> pd.DataFrame:
    # extract the category (e.g., "orbitrap", "qtof") from the variable name
    df_melted = pd.melt(df)
    df_melted["category"] = df_melted["variable"].apply(
        lambda x: x.split("_")[-1] if "_" in x else None
    )
    df_melted["value_name"] = df_melted["variable"].apply(
        lambda x: "_".join(x.split("_")[:-1]) if "_" in x else x
    )

    # pivot the DataFrame to get the desired format
    df_pivoted = df_melted.pivot(index="category", columns="value_name", values="value")
    top_all = df_pivoted.pop("top")
    top_all.dropna(inplace=True)
    df_pivoted.dropna(inplace=True)
    df_pivoted.loc["overall"] = top_all.values

    return df_pivoted[
        [
            "top_1",
            "top_5",
            "top_10",
            "top_20",
        ]
    ].copy()


def generate_full_results(
    spectra: T.List[Spectrum],
    dataframes: T.List[pd.DataFrame],
) -> pd.DataFrame:
    n_empty = 0
    scores_of_true_inchikey = []
    positions_of_true_inchikey = []
    inichikeys_of_true_mol = []
    smiles_of_true_inchikey = []
    adducts_of_true_mol = []
    instrument_of_true_mol = []
    identifiers_of_true_mol = []
    for spectrum, spec_df in tqdm(
        zip(spectra, dataframes), desc="Saving results", total=len(spectra)
    ):
        if spec_df.empty:
            n_empty += 1
            continue
        inchikey = spectrum.get("inchikey")
        spec_df = spec_df[spec_df["InChIKey1"] == inchikey]
        if spec_df.empty:
            n_empty += 1
            continue
        scores_of_true_inchikey.append(spec_df["Score"].values[0])
        positions_of_true_inchikey.append(spec_df.index[0] + 1)
        inichikeys_of_true_mol.append(spectrum.get("inchikey"))
        smiles_of_true_inchikey.append(spectrum.get("smiles"))
        adducts_of_true_mol.append(spectrum.get("adduct"))
        instrument_of_true_mol.append(spectrum.get("instrument_type"))
        identifiers_of_true_mol.append(spectrum.get("identifier"))

    df = pd.DataFrame(
        {
            "n_empty": [n_empty] * len(scores_of_true_inchikey),
            "score": scores_of_true_inchikey,
            "top_n": positions_of_true_inchikey,
            "inchikey": inichikeys_of_true_mol,
            "smiles": smiles_of_true_inchikey,
            "adduct": adducts_of_true_mol,
            "instrument_type": instrument_of_true_mol,
            "identifier": identifiers_of_true_mol,
        }
    )
    return df


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
    for spectrum, res in tqdm(
        zip(spectra, results), desc="Analyzing results", total=len(spectra)
    ):
        # get the metadata from the spectrum
        inchikey: T.Optional[str] = spectrum.get("inchikey")
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
