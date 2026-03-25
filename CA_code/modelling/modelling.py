# ============================================================
# modelling/modelling.py — Design Choice 1: Chained Multi-Output
# ONE model instance per chaining level (3 levels total)
# Labels are concatenated: y2 → y2+y3 → y2+y3+y4
# ============================================================

import numpy as np
from modelling.data_model import Data
from model.randomforest import RandomForest
from Config import CHAIN_COLS


def _build_chained_label(df, cols):
    """
    Concatenate label columns into a single combined label string.

    Example:
        cols = ['y2', 'y3']
        row:  y2='Suggestion', y3='Payment'
        result: 'Suggestion + Payment'

    Args:
        df:   DataFrame containing the label columns.
        cols: List of column names to concatenate.

    Returns:
        pandas Series of combined label strings.
    """
    combined = df[cols[0]].astype(str)
    for col in cols[1:]:
        combined = combined + ' + ' + df[col].astype(str)
    return combined


def model_predict_chained(X, df, group_name, chain_cols=None):
    """
    Design Choice 1: Chained Multi-Output Classification.

    Trains ONE RandomForest instance per chaining level:
      Level 1 — classifies Type 2 only
      Level 2 — classifies Type 2 + Type 3 (combined label)
      Level 3 — classifies Type 2 + Type 3 + Type 4 (combined label)

    Accuracy cascades: L3 ≤ L2 ≤ L1, because a wrong Type 2
    prediction at Level 1 makes all subsequent combined labels wrong.

    Args:
        X:          Full TF-IDF embedding matrix (n_samples, n_features)
        df:         Cleaned DataFrame with y2, y3, y4 columns
        group_name: Name of the data group (for display purposes)
        chain_cols: List of columns to chain (default: Config.CHAIN_COLS)
    """
    if chain_cols is None:
        chain_cols = CHAIN_COLS

    print(f"\n{'='*60}")
    print(f"  DESIGN CHOICE 1 — CHAINED MULTI-OUTPUT")
    print(f"  Group: {group_name}")
    print(f"{'='*60}")

    for level in range(1, len(chain_cols) + 1):
        cols_at_level = chain_cols[:level]
        level_label = ' + '.join([c.upper() for c in cols_at_level])

        print(f"\n--- Level {level}: {level_label} ---")

        # Build combined label for this level
        df_copy = df.copy()
        df_copy['y'] = _build_chained_label(df_copy, cols_at_level)

        # Encapsulate data
        data = Data(X, df_copy, label_col='y')

        if not data.is_valid():
            print(f"  [SKIP] Insufficient data at Level {level} "
                  f"(fewer than 2 classes or too few samples).")
            continue

        print(f"  Classes ({len(data.classes)}): {list(data.classes)}")
        print(f"  Train samples: {len(data.X_train)} | Test samples: {len(data.X_test)}")

        # Instantiate model, train, predict, report
        model = RandomForest()
        model.train(data)
        model.predict(data.X_test)
        model.print_results(data)


def model_predict(X, df, group_name, label_col='y2'):
    """
    Original single-label prediction (Exercise 3 compatible).
    Kept for backward compatibility.

    Args:
        X:          TF-IDF embedding matrix
        df:         DataFrame with label columns
        group_name: Group identifier
        label_col:  Single label column to classify
    """
    print(f"\n{'='*60}")
    print(f"  SINGLE-LABEL PREDICTION — {label_col.upper()}")
    print(f"  Group: {group_name}")
    print(f"{'='*60}")

    data = Data(X, df, label_col=label_col)

    if not data.is_valid():
        print(f"  [SKIP] Insufficient data for {label_col}.")
        return

    print(f"  Classes ({len(data.classes)}): {list(data.classes)}")
    print(f"  Train samples: {len(data.X_train)} | Test samples: {len(data.X_test)}")

    model = RandomForest()
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)
