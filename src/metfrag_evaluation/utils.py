import pandas as pd


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
