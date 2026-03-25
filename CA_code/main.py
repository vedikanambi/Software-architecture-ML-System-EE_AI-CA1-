# ============================================================
# main.py — Controller for Design Choice 1: Chained Multi-Output
#
# Pipeline:
#   1. preprocess.py  → load, deduplicate, clean all CSV files
#   2. embeddings.py  → convert text to TF-IDF numeric matrix
#   3. modelling.py   → chained multi-output classification (3 levels)
#
# The controller only calls high-level functions — it has zero
# knowledge of model internals or data cleaning specifics.
# ============================================================

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess
from embeddings import get_tfidf_embd
from modelling.modelling import model_predict_chained
from Config import DATA_FILES


def perform_chained_modelling(df, X):
    """
    Run Design Choice 1 (Chained Multi-Output) for each data group.

    Splits the combined DataFrame by 'group' column (one per source CSV)
    and runs the chained pipeline independently on each group.

    Args:
        df: Cleaned DataFrame from preprocessing.
        X:  Full TF-IDF embedding matrix.
    """
    groups = df['group'].unique()
    print(f"\n[main] Found {len(groups)} group(s): {list(groups)}")

    for group_name in groups:
        # Filter rows and corresponding embedding rows for this group
        mask = df['group'] == group_name
        df_group = df[mask].reset_index(drop=True)
        X_group = X[mask.values]

        model_predict_chained(X_group, df_group, group_name)


def main():
    print("=" * 60)
    print("  EMAIL CLASSIFIER — Design Choice 1: Chained Multi-Output")
    print("=" * 60)

    # Step 1: Preprocessing
    print("\n[Step 1] Preprocessing...")
    df = preprocess(DATA_FILES)

    # Step 2: Embedding
    print("[Step 2] Generating TF-IDF embeddings...")
    X, _ = get_tfidf_embd(df)

    # Step 3: Chained multi-output classification
    print("\n[Step 3] Running chained multi-output classification...")
    perform_chained_modelling(df, X)

    print("\n" + "=" * 60)
    print("  DONE — Design Choice 1 complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
