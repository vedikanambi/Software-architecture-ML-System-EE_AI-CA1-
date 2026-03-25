# ============================================================
# main_hierarchical.py — Controller for Design Choice 2: Hierarchical Modelling
#
# Pipeline:
#   1. preprocess.py      → load, deduplicate, clean all CSV files
#   2. embeddings.py      → convert text to TF-IDF numeric matrix
#   3. hierarchical.py    → hierarchical classification (3 levels, tree of models)
#
# Same SoC principle as main.py — controller calls only high-level functions.
# preprocess.py and embeddings.py are UNCHANGED from Design Choice 1.
# ============================================================

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocess import preprocess
from embeddings import get_tfidf_embd
from modelling.hierarchical import hierarchical_predict
from Config import DATA_FILES


def main():
    print("=" * 60)
    print("  EMAIL CLASSIFIER — Design Choice 2: Hierarchical Modelling")
    print("=" * 60)

    # Step 1: Preprocessing (identical to DC1 — demonstrates SoC)
    print("\n[Step 1] Preprocessing...")
    df = preprocess(DATA_FILES)

    # Step 2: Embedding (identical to DC1 — demonstrates SoC)
    print("[Step 2] Generating TF-IDF embeddings...")
    X, _ = get_tfidf_embd(df)

    # Step 3: Hierarchical classification
    print("\n[Step 3] Running hierarchical classification...")
    groups = df['group'].unique()
    print(f"[main] Found {len(groups)} group(s): {list(groups)}")

    for group_name in groups:
        mask = df['group'] == group_name
        df_group = df[mask].reset_index(drop=True)
        X_group = X[mask.values]

        hierarchical_predict(X_group, df_group, group_name)

    print("\n" + "=" * 60)
    print("  DONE — Design Choice 2 complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
