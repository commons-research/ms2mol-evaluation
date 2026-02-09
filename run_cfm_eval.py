from pathlib import Path

from matchms.similarity import ModifiedCosine

from ms2mol_evaluation import CFMEvaluation


def main():
    eval = CFMEvaluation(
        Path("evaluations/cfmid_lotus"),
        precursor_mz_tolerance=20.0,
        precursor_mz_tolerance_type="ppm",
    )
    ms2_similarity = ModifiedCosine(tolerance=0.01)
    eval.set_ms2_similarity(ms2_similarity)
    res = eval.run_eval()
    eval.write_top_k_proba_to_json(res)
    eval.get_fraction_results(res)
    eval.get_top_n_results(res)


if __name__ == "__main__":
    main()
