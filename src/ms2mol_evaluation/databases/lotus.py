import pandas as pd


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
    name CHAR(14) NOT NULL,
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

    download_isdb()
    isdb = load_isdb()
    identifier = [s.get("compound_name") for s in isdb]
    inchi = [s.get("inchi") for s in isdb]
    exact_mass = [s.get("parent_mass") for s in isdb]
    molecular_formula = [s.get("molecular_formula") for s in isdb]
    inchikey_1 = [s.get("compound_name") for s in isdb]
    inchikey_2 = [s.get("inchikey").split("-")[1] for s in isdb]
    inchikey_3 = [s.get("inchikey").split("-")[2] for s in isdb]
    smiles = [s.get("smiles") for s in isdb]
    name = [s.get("compound_name") for s in isdb]

    lotus_db = (
        pd.DataFrame(
            {
                "Identifier": identifier,
                "InChI": inchi,
                "MonoisotopicMass": exact_mass,
                "MolecularFormula": molecular_formula,
                "InChIKey1": inchikey_1,
                "InChIKey2": inchikey_2,
                "SMILES": smiles,
                "Name": name,
                "InChIKey3": inchikey_3,
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
