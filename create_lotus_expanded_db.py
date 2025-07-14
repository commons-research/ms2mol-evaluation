import gc
import os
import typing as T

import pandas as pd
import polars as pl
import psycopg2
from dotenv import load_dotenv
from joblib import Parallel, delayed
from pymongo import MongoClient
from rdkit.Chem import Mol, MolToInchiKey
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from skfp.preprocessing import (
    MolFromSmilesTransformer,
    MolToInchiTransformer,
    MolToSmilesTransformer,
)
from tqdm import tqdm

from ms2mol_evaluation.lotus_expanded import create_insert_query, create_table_query

load_dotenv()


def fetch_lotus_expanded_from_mongodb() -> pl.DataFrame:
    client = MongoClient()
    db = client.get_database("lotus_mines")
    collection = db.get_collection("compounds")
    df = pl.from_dicts(collection.find(), infer_schema_length=500)
    return df


def convert_smiles_to_mol(
    smiles: T.List[str],
    n_jobs: int,
    valid_only: bool = True,
) -> T.List[Mol]:
    transformer = MolFromSmilesTransformer(
        n_jobs=n_jobs, verbose=True, valid_only=valid_only
    )
    return transformer.transform(smiles)


def get_exact_masses(
    mols: T.List[Mol],
) -> T.List[float]:
    """
    Get exact masses for a list of RDKit Mol objects.
    """
    return [
        ExactMolWt(mol)
        for mol in tqdm(mols, desc="Calculating monoisotopic masses", leave=False)
        if mol is not None
    ]


def get_mol_formulas(
    mols: T.List[Mol],
) -> T.List[str]:
    """
    Get molecular formulas for a list of RDKit Mol objects.
    """
    return [
        CalcMolFormula(mol)
        for mol in tqdm(mols, desc="Calculating mol formulas", leave=False)
        if mol is not None
    ]


def get_inchis(
    mols: T.List[Mol],
    n_jobs: int = -1,
    batch_size: int = 5000,
) -> T.List[str]:
    """
    Get InChI strings for a list of RDKit Mol objects.
    """
    transformer = MolToInchiTransformer(
        n_jobs=n_jobs, batch_size=batch_size, verbose=True
    )
    return transformer.transform(mols)


def mol_to_inchikey(
    mol: Mol,
) -> str:
    """
    Convert a single RDKit Mol object to its InChIKey.
    """
    if mol is None:
        raise ValueError("Input molecule is None.")
    inchikey = MolToInchiKey(mol)
    return inchikey


def get_inchikeys(
    mols: T.List[Mol],
    n_jobs: int = -1,
) -> T.List[str]:
    """
    Get InChIKey strings for a list of RDKit Mol objects.
    """
    inchikeys = Parallel(n_jobs=n_jobs)(
        delayed(mol_to_inchikey)(mol) for mol in tqdm(mols)
    )
    return inchikeys


def get_smiles(
    mols: T.List[Mol],
    n_jobs: int = -1,
) -> T.List[str]:
    """
    Get SMILES strings for a list of RDKit Mol objects.
    """
    transformer = MolToSmilesTransformer(verbose=True, n_jobs=n_jobs)
    smiles = transformer.transform(mols)
    return smiles


def create_dataframe_for_db(mols: T.List[Mol]) -> pd.DataFrame:
    monoisotopic_masses = get_exact_masses(mols)
    formulas = get_mol_formulas(mols)
    inchis = get_inchis(mols)
    inchikeys = get_inchikeys(mols)
    smiles = get_smiles(mols)

    data = (
        pd.DataFrame(
            {
                "Identifier": inchikeys,
                "InChI": inchis,
                "MonoisotopicMass": monoisotopic_masses,
                "MolecularFormula": formulas,
                "InChIKey1": list(map(lambda x: x.split("-")[0], inchikeys)),
                "InChIKey2": list(map(lambda x: x.split("-")[1], inchikeys)),
                "SMILES": smiles,
                "Name": inchikeys,
                "InChIKey3": list(map(lambda x: x.split("-")[2], inchikeys)),
            }
        )
        .drop_duplicates("InChIKey1")
        .reset_index(drop=True)
    )
    return data


def main():
    df = fetch_lotus_expanded_from_mongodb()
    mols = convert_smiles_to_mol(df["SMILES"].to_list(), n_jobs=-1, valid_only=True)
    del df
    df = create_dataframe_for_db(mols)
    del mols
    gc.collect()

    conn = psycopg2.connect(
        database=os.getenv("LOTUS_DB_PGDATABASE"),
        host=os.getenv("LOTUS_DB_PGHOST"),
        port=os.getenv("LOTUS_DB_PGPORT"),
        user=os.getenv("LOTUS_DB_POSTGRES_USER"),
        password=os.getenv("LOTUS_DB_POSTGRES_PASSWORD"),
    )
    conn.autocommit = True
    cursor = conn.cursor()

    table_query = create_table_query()
    cursor.execute(table_query)

    insert_query = create_insert_query()
    data = df.values.tolist()
    batch_size = 1000
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        cursor.executemany(insert_query, batch)


if __name__ == "__main__":
    main()
