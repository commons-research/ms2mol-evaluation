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
