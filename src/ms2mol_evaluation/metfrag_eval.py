from pathlib import Path

import pandas as pd
from downloaders import BaseDownloader

from ms2mol_evaluation.evaluation import Evaluation


class MetFragEvaluation(Evaluation):
    def __init__(self, output_dir: Path) -> None:
        super().__init__(output_dir)
        self.metfrag_exec: str = (
            MetFragEvaluation.download_metfrag_exec().destination.values[0]
        )

    @staticmethod
    def download_metfrag_exec(
        version: str = "2.6.6",
        auto_extract: bool = False,
    ) -> pd.DataFrame:
        return BaseDownloader(auto_extract=auto_extract).download(
            f"https://github.com/ipb-halle/MetFragRelaunched/releases/download/v{version}/MetFragCommandLine-{version}.jar",
            str(
                (
                    Path("downloads/metfrag") / f"MetFragCommandLine-{version}.jar"
                ).resolve()
            ),
        )
