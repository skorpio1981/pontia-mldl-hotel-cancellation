"""Carga, limpieza y división del dataset de reservas hoteleras."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


COLUMNA_OBJETIVO = "is_canceled"
COLUMNAS_FUGA_DATOS = ["reservation_status", "reservation_status_date"]


# ============================================================================
# 1. CARGA Y LIMPIEZA INICIAL
# Esta función reúne los pasos que siempre deben ejecutarse antes de entrenar:
# leer el CSV, eliminar duplicados, quitar las columnas que producen fuga de
# datos y separar las variables predictoras (X) de la variable objetivo (y).
# ============================================================================
def cargar_datos(ruta_dataset):
    """Lee el CSV, elimina duplicados y separa variables y objetivo."""
    ruta_dataset = Path(ruta_dataset)

    if not ruta_dataset.exists():
        raise FileNotFoundError(f"No se encuentra el dataset: {ruta_dataset}")

    df_reservas = pd.read_csv(ruta_dataset)

    if COLUMNA_OBJETIVO not in df_reservas.columns:
        raise ValueError(f"Falta la columna objetivo: {COLUMNA_OBJETIVO}")

    numero_duplicados = df_reservas.duplicated().sum()
    df_reservas = df_reservas.drop_duplicates().reset_index(drop=True)

    X = df_reservas.drop(
        columns=[COLUMNA_OBJETIVO, *COLUMNAS_FUGA_DATOS],
        errors="ignore",
    )
    y = df_reservas[COLUMNA_OBJETIVO].astype(int)

    return df_reservas, X, y, numero_duplicados


# ============================================================================
# 2. DIVISIÓN EN ENTRENAMIENTO, VALIDACIÓN Y TEST
# La estratificación mantiene una proporción semejante de cancelaciones en los
# tres conjuntos. Primero se reserva un 30 % y luego se divide en dos mitades.
# Resultado final: 70 % train, 15 % validación y 15 % test.
# ============================================================================
def dividir_datos(X, y, random_state=42):
    """Crea conjuntos estratificados de train, validación y test (70/15/15)."""
    X_train, X_temporal, y_train, y_temporal = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=random_state,
    )

    X_validacion, X_test, y_validacion, y_test = train_test_split(
        X_temporal,
        y_temporal,
        test_size=0.50,
        stratify=y_temporal,
        random_state=random_state,
    )

    return X_train, X_validacion, X_test, y_train, y_validacion, y_test
