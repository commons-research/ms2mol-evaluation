import argparse
import random

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from metfrag_evaluation import massspecgym
from metfrag_evaluation.massspecgym import load_massspecgym, to_spectra
from metfrag_evaluation.metfrag import run_metfrag


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

    massspecgym = load_massspecgym()
    spectra = to_spectra(massspecgym)
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


if __name__ == "__main__":
    main()
