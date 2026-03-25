# ============================================================
# preprocess.py -- Data loading, cleaning, and preparation
# Completely independent of modelling (Separation of Concerns)
# ============================================================

import pandas as pd
import numpy as np
from Config import DATA_FILES, COLUMN_MAP, TEXT_COLS


def load_data(file_paths=None):
    """
    Load one or more CSV files and concatenate into a single DataFrame.
    Renames Type 1-4 columns to y1-y4.
    """
    if file_paths is None:
        file_paths = DATA_FILES

    frames = []
    for path in file_paths:
        try:
            df = pd.read_csv(path, encoding='utf-8-sig')  # utf-8-sig strips BOM
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin-1')

        # Strip whitespace from column names and all string cell values
        df.columns = df.columns.str.strip()
        df = df.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)

        # Drop unnamed/empty trailing columns
        df = df.loc[:, ~df.columns.str.startswith('Unnamed')]

        # Add group identifier (filename without extension)
        df['group'] = path.replace('.csv', '')

        # Rename Type 1-4 to y1-y4
        df = df.rename(columns=COLUMN_MAP)

        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"[preprocess] Loaded {len(combined)} records from {len(file_paths)} file(s).")
    return combined


def de_duplication(df):
    """
    Remove exact duplicate rows based on text columns.
    """
    before = len(df)
    df = df.drop_duplicates(subset=TEXT_COLS, keep='first')
    after = len(df)
    print(f"[preprocess] De-duplication: {before} -> {after} rows ({before - after} removed).")
    return df.reset_index(drop=True)


def noise_remover(df):
    """
    Remove noise from the dataset:
      - Rows where any text column is null or empty
      - Rows where any label column (y1-y4) is null or empty
    """
    before = len(df)

    # Drop rows with missing text
    for col in TEXT_COLS:
        if col in df.columns:
            df = df[df[col].notna()]
            df = df[df[col].str.strip() != '']

    # Drop rows with missing labels
    label_cols = [c for c in ['y1', 'y2', 'y3', 'y4'] if c in df.columns]
    for col in label_cols:
        df = df[df[col].notna()]
        df = df[df[col].str.strip() != '']

    after = len(df)
    print(f"[preprocess] Noise removal: {before} -> {after} rows ({before - after} removed).")
    return df.reset_index(drop=True)


def preprocess(file_paths=None):
    """
    Full preprocessing pipeline: load -> deduplicate -> remove noise.
    """
    df = load_data(file_paths)
    df = de_duplication(df)
    df = noise_remover(df)
    print(f"[preprocess] Final dataset: {len(df)} records.\n")
    return df
