import os

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

from ms2mol_evaluation.lotus import (
    create_lotus_table_query,
    generate_index_query,
    generate_insert_query,
    load_lotus_for_metfrag,
)

load_dotenv()


def main():
    df = load_lotus_for_metfrag()
    conn = psycopg2.connect(
        database=os.getenv("LOTUS_DB_PGDATABASE"),
        host=os.getenv("LOTUS_DB_PGHOST"),
        port=os.getenv("LOTUS_DB_PGPORT"),
        user=os.getenv("LOTUS_DB_POSTGRES_USER"),
        password=os.getenv("LOTUS_DB_POSTGRES_PASSWORD"),
    )
    conn.autocommit = True
    cursor = conn.cursor()

    create_table_query = create_lotus_table_query()
    cursor.execute(create_table_query)

    insert_query = generate_insert_query()
    data = df.values.tolist()

    batch_size = 10000
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i : i + batch_size]
        cursor.executemany(insert_query, batch)

    index_query = generate_index_query()
    cursor.execute(index_query)


if __name__ == "__main__":
    main()
