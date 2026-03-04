from pathlib import Path
import pandas as pd

from ms2mol_evaluation import MetFragEvaluation


def main():
    eval = MetFragEvaluation(
        Path("evaluations/metfrag_lotus"),
    )

    res = eval.run_eval(n_jobs=8)
    eval.concatenate_results(res)
    eval.get_fraction_results(res)
    eval.get_top_n_results(res)


if __name__ == "__main__":
    main()
