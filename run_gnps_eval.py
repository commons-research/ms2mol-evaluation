from pathlib import Path

from matchms.similarity import ModifiedCosineGreedy

from ms2mol_evaluation import GNPSEvaluation


def main():
    eval = GNPSEvaluation(
        Path("evaluations/gnps_lotus"),
        precursor_mz_tolerance=20.0,
        precursor_mz_tolerance_type="ppm",
    )
    ms2_similarity = ModifiedCosineGreedy()
    eval.set_ms2_similarity(ms2_similarity)
    res = eval.run_eval()


if __name__ == "__main__":
    main()
