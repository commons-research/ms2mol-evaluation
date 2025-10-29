from pathlib import Path

from matchms.similarity import FlashSimilarity

from ms2mol_evaluation import IcebergEvaluation


def main():
    eval = IcebergEvaluation(
        Path("evaluations/iceberg_lotus"),
        precursor_mz_tolerance=10.0,
        precursor_mz_tolerance_type="ppm",
    )

    ms2_similarity = FlashSimilarity(
        score_type="cosine", matching_mode="hybrid", tolerance=0.01
    )
    eval.set_ms2_similarity(ms2_similarity)
    res = eval.run_eval()
    eval.get_fraction_results(res)
    eval.get_top_n_results(res)


if __name__ == "__main__":
    main()
