TABLE_NAME = "lotus_expanded"


def create_table_query():
    query = f"""
DROP TABLE IF EXISTS {TABLE_NAME};
DROP INDEX IF EXISTS idx_{TABLE_NAME}_mass;
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
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


def create_insert_query():
    insert_query = f"""
INSERT INTO {TABLE_NAME} (
    identifier, inchi, monoisotopic_mass, formula,
    inchikey_1, inchikey_2, smiles, name, inchikey_3
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
    return insert_query


def create_index_query():
    index_query = f"""
CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_mass ON {TABLE_NAME} (monoisotopic_mass);
"""
    return index_query
