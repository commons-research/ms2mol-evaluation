from pathlib import Path

from matchms.similarity import ModifiedCosine

from ms2mol_evaluation import GNPSEvaluation


def main():
    eval = GNPSEvaluation(
        Path("evaluations/gnps_lotus"),
        precursor_mz_tolerance=20.0,
        precursor_mz_tolerance_type="ppm",
    )
    ms2_similarity = ModifiedCosine(tolerance=0.01)
    eval.set_ms2_similarity(ms2_similarity)
    res = eval.run_eval()
    eval.get_fraction_results(res)
    eval.get_top_n_results(res)


if __name__ == "__main__":
    main()
