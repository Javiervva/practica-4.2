import os
import sqlite3

import pandas as pd

from app import ROOT_DIR


def load_dataset_to_db():
    """
    Función para cargar un dataset desde un archivo CSV a una base de datos SQLite.

    Returns:
       int.  Número total de registros cargados.
    """
    # Cargar los datos desde el archivo CSV
    df_path = os.path.join(ROOT_DIR, "data/data.csv")
    df = pd.read_csv(df_path)

    # Crear una conexión a la base de datos SQLite
    conn = sqlite3.connect('titanic.db')

    # Escribir el DataFrame a una tabla en SQLite
    df.to_sql('titanic', conn, if_exists='replace', index=False)

    # Obtener el número total de registros cargados
    total_records = df.shape[0]

    return total_records


def check_data_in_db():
    """
    Función para consultar la base de datos SQLite para comprobar si hay datos.

    Returns:
       int.  Número total de registros en la base de datos.
    """
    # Crear una conexión a la base de datos SQLite
    conn = sqlite3.connect('titanic.db')

    # Consultar la base de datos para comprobar si hay datos
    query = "SELECT COUNT(*) FROM titanic"
    result = pd.read_sql_query(query, conn)

    # Obtener el número total de registros en la base de datos
    total_records = result.iloc[0, 0]

    return total_records


def get_raw_data_from_db():
    """
    Función para obtener los datos originales desde SQLite local

    Returns:
       DataFrame. Dataset con los datos de entrada.
    """
    conn = sqlite3.connect('titanic.db')
    query = "SELECT * FROM titanic"
    df = pd.read_sql_query(query, conn)
    return df.copy()
