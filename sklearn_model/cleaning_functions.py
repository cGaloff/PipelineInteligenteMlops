import pandas as pd
import numpy as np

numeric_cols = [
    "Year_of_Release",
    "Critic_Score", "Critic_Count",
    "User_Score", "User_Count"
]

cat_cols = ["Platform", "Genre", "Publisher", "Developer", "Rating"]



def replace_tbd_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
    "Year_of_Release",
    "Critic_Score", "Critic_Count",
    "User_Score", "User_Count"
    ]

    """
    Reemplaza el texto 'tbd' por NaN en las columnas numéricas definidas
    en numeric_cols.
    """
    df[numeric_cols] = df[numeric_cols].replace("tbd", np.nan)
    return df



def ensure_numeric(df: pd.DataFrame, cols=numeric_cols) -> pd.DataFrame:

    """
    Convierte las columnas indicadas a tipo numérico.
    Cualquier valor no convertible se vuelve NaN (errors='coerce').
    """
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df

def fill_missing_publisher_developer(df: pd.DataFrame, cols = ["Publisher", "Developer"]) -> pd.DataFrame:

    """
    Rellena los valores faltantes de las columnas categóricas
    'Publisher' y 'Developer' con 'Unknown'.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df

def fill_cats_with_mode(df: pd.DataFrame, cols=["Genre", "Platform", "Rating"]) -> pd.DataFrame:

    """
    Rellena valores faltantes en columnas categóricas con su valor más frecuente (moda).
    Pensado para columnas como Genre, Platform y Rating.
    """
    for col in cols:
        if col in df.columns:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
    return df

def normalize_categories(df: pd.DataFrame, cols = cat_cols) -> pd.DataFrame:

    for col in cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)      # asegura tipo string
                .str.strip()      # quita espacios al inicio y final
                .str.lower()      # pasa a minúsculas
            )
            
    return df


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    return df

def drop_rows_without_name_year(df):
    df = df[~df["Name"].isna()]
    df = df[~df["Year_of_Release"].isna()]   # o df[df["Name"].notna()]
    return df


def impute_numeric_missing(df: pd.DataFrame,
                           cols=["Critic_Score", "Critic_Count", "User_Score", "User_Count"],
                           strategy: str = "median") -> pd.DataFrame:
    """
    Imputa valores faltantes en columnas numéricas usando media o mediana.

    strategy: "median" (por defecto) o "mean".
    """
    for col in cols:
        if col in df.columns:
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            else:
                raise ValueError("strategy debe ser 'median' o 'mean'")

            df[col] = df[col].fillna(fill_value)

    return df