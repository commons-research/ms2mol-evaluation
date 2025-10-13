from pathlib import Path

from ms2mol_evaluation import SiriusEvaluation


def main():
    eval = SiriusEvaluation(
        Path("evaluations/sirius_lotus"),
    )

    res = eval.run_eval()
    eval.get_fraction_results(res)
    eval.get_top_n_results(res)


if __name__ == "__main__":
    main()
