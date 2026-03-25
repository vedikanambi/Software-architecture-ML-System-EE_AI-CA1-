# ============================================================
# modelling/hierarchical.py — Design Choice 2: Hierarchical Modelling
# Model INSTANCES are chained, not labels.
# Output of each model filters the data for the next level.
#
# Level 0: 1 RF model → classifies Type 2 on full dataset
# Level 1: N RF models → one per Type 2 class, classifies Type 3
# Level 2: M RF models → one per (Type2 class × Type3 class), classifies Type 4
# ============================================================

import numpy as np
from modelling.data_model import Data
from model.randomforest import RandomForest
from Config import MIN_CLASS_SAMPLES


def hierarchical_predict(X, df, group_name):
    """
    Design Choice 2: Hierarchical Modelling.

    Three-level hierarchy:
      Level 0 — classify Type 2 on the full dataset (1 model)
      Level 1 — for each Type 2 class, filter data and classify Type 3
      Level 2 — for each (Type2, Type3) pair, filter further and classify Type 4

    Benefits:
      - Each sub-model sees only class-specific data (focused decision boundary)
      - Accuracy can be assessed independently per parent class
      - Mirrors real-world hierarchical taxonomies

    Trade-off:
      - Number of models grows as N(y2) × N(y3) — exponential with classes
      - Small subsets may have insufficient data to train

    Args:
        X:          Full TF-IDF embedding matrix (n_samples, n_features)
        df:         Cleaned DataFrame with y2, y3, y4 columns
        group_name: Name of the data group (for display purposes)
    """
    print(f"\n{'='*60}")
    print(f"  DESIGN CHOICE 2 — HIERARCHICAL MODELLING")
    print(f"  Group: {group_name}")
    print(f"{'='*60}")

    # ----------------------------------------------------------------
    # LEVEL 0 — Classify Type 2 on the full dataset
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print(f"  LEVEL 0: Classifying TYPE 2 (full dataset)")
    print(f"{'─'*50}")

    data_l0 = Data(X, df, label_col='y2')

    if not data_l0.is_valid():
        print("  [SKIP] Insufficient data for Level 0 (Type 2).")
        return

    print(f"  Type 2 classes ({len(data_l0.classes)}): {list(data_l0.classes)}")
    print(f"  Train: {len(data_l0.X_train)} | Test: {len(data_l0.X_test)}")

    model_l0 = RandomForest()
    model_l0.train(data_l0)
    model_l0.predict(data_l0.X_test)
    model_l0.print_results(data_l0)

    # Get valid Type 2 classes from the full dataset for Level 1 filtering
    y2_classes = _get_valid_classes(df['y2'], MIN_CLASS_SAMPLES)

    # ----------------------------------------------------------------
    # LEVEL 1 — For each Type 2 class: filter → classify Type 3
    # ----------------------------------------------------------------
    print(f"\n{'─'*50}")
    print(f"  LEVEL 1: Classifying TYPE 3 | filtered by TYPE 2 class")
    print(f"  Number of models to create: {len(y2_classes)}")
    print(f"{'─'*50}")

    for y2_class in y2_classes:
        print(f"\n  [Filter Set A] Type 2 == '{y2_class}'")

        # Filter data to rows where y2 == y2_class
        mask_a = df['y2'] == y2_class
        df_a = df[mask_a].copy()
        X_a = X[mask_a.values]

        if len(df_a) == 0:
            print(f"    [SKIP] No data for y2 == '{y2_class}'")
            continue

        data_l1 = Data(X_a, df_a, label_col='y3')

        if not data_l1.is_valid():
            print(f"    [SKIP] Insufficient data for Type 3 "
                  f"(subset size: {len(df_a)}, needs ≥2 classes with ≥{MIN_CLASS_SAMPLES} samples).")
            continue

        print(f"    Type 3 classes ({len(data_l1.classes)}): {list(data_l1.classes)}")
        print(f"    Train: {len(data_l1.X_train)} | Test: {len(data_l1.X_test)}")

        model_l1 = RandomForest()
        model_l1.train(data_l1)
        model_l1.predict(data_l1.X_test)
        model_l1.print_results(data_l1)

        # Get valid Type 3 classes within this Filter Set A
        y3_classes = _get_valid_classes(df_a['y3'], MIN_CLASS_SAMPLES)

        # ------------------------------------------------------------
        # LEVEL 2 — For each (y2, y3) pair: filter → classify Type 4
        # ------------------------------------------------------------
        print(f"\n    LEVEL 2: Classifying TYPE 4 | filtered by "
              f"Type 2='{y2_class}' AND Type 3")
        print(f"    Number of Level 2 models: {len(y3_classes)}")

        for y3_class in y3_classes:
            print(f"\n    [Filter Set B] Type 2 == '{y2_class}' AND Type 3 == '{y3_class}'")

            # Filter Set B: further filter on y3 within Filter Set A
            mask_b = df_a['y3'] == y3_class
            df_b = df_a[mask_b].copy()
            X_b = X_a[mask_b.values]

            if len(df_b) == 0:
                print(f"      [SKIP] No data for y3 == '{y3_class}'")
                continue

            data_l2 = Data(X_b, df_b, label_col='y4')

            if not data_l2.is_valid():
                print(f"      [SKIP] Insufficient data for Type 4 "
                      f"(subset size: {len(df_b)}, needs ≥2 classes with ≥{MIN_CLASS_SAMPLES} samples).")
                continue

            print(f"      Type 4 classes ({len(data_l2.classes)}): {list(data_l2.classes)}")
            print(f"      Train: {len(data_l2.X_train)} | Test: {len(data_l2.X_test)}")

            model_l2 = RandomForest()
            model_l2.train(data_l2)
            model_l2.predict(data_l2.X_test)
            model_l2.print_results(data_l2)


def _get_valid_classes(series, min_samples):
    """
    Return class values that appear at least min_samples times.

    Args:
        series:      pandas Series of label values.
        min_samples: Minimum count threshold.

    Returns:
        List of valid class label strings.
    """
    counts = series.value_counts()
    return list(counts[counts >= min_samples].index)
