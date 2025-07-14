import pandas as pd
from downloaders import BaseDownloader


def create_lotus_table_query():
    query = """
DROP TABLE IF EXISTS lotus;
DROP INDEX IF EXISTS idx_lotus_mass;
CREATE TABLE IF NOT EXISTS lotus (
    id SERIAL PRIMARY KEY,
    identifier TEXT NOT NULL,
    inchi TEXT NOT NULL,
    monoisotopic_mass FLOAT NOT NULL,
    formula VARCHAR(255) NOT NULL,
    inchikey_1 CHAR(14) NOT NULL,
    inchikey_2 VARCHAR(10) NOT NULL,
    smiles TEXT NOT NULL,
    name CHAR(27) NOT NULL,
    inchikey_3 CHAR(1) NOT NULL
);
"""
    return query


def load_lotus_for_metfrag() -> pd.DataFrame:
    """
    Loads the LOTUS dataset formated as a DataFrame suitable for MetFrag.

    Returns:
        pd.DataFrame: DataFrame containing LOTUS data.
    """

    lotus_path = "data/lotus/230106_frozen_metadata.csv.gz"
    _ = BaseDownloader(auto_extract=False).download(
        urls="https://zenodo.org/records/7534071/files/230106_frozen_metadata.csv.gz",
        paths=lotus_path,
    )

    lotus = pd.read_csv(lotus_path, compression="gzip")

    lotus_db = (
        pd.DataFrame(
            {
                "Identifier": lotus["structure_wikidata"],
                "InChI": lotus["structure_inchi"],
                "MonoisotopicMass": lotus["structure_exact_mass"],
                "MolecularFormula": lotus["structure_molecular_formula"],
                "InChIKey1": lotus["structure_inchikey"].apply(
                    lambda x: x.split("-")[0]
                ),
                "InChIKey2": lotus["structure_inchikey"].apply(
                    lambda x: x.split("-")[1]
                ),
                "SMILES": lotus["structure_smiles_2D"],
                "Name": lotus["structure_inchikey"],
                "InChIKey3": lotus["structure_inchikey"].apply(
                    lambda x: x.split("-")[2]
                ),
            }
        )
        .drop_duplicates("InChIKey1")
        .reset_index(drop=True)
    )

    return lotus_db


def generate_insert_query():
    insert_query = """
INSERT INTO lotus (
    identifier, inchi, monoisotopic_mass, formula,
    inchikey_1, inchikey_2, smiles, name, inchikey_3
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
    return insert_query


def generate_index_query():
    index_query = """
CREATE INDEX IF NOT EXISTS idx_lotus_mass ON lotus (monoisotopic_mass);
"""
    return index_query
