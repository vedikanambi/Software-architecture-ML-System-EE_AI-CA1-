# ============================================================
# Config.py — Shared constants across all components
# ============================================================

# Raw CSV column names → renamed to y1–y4 after loading
# Note: actual CSVs have trailing spaces on Type columns — stripped on load
COLUMN_MAP = {
    'Type 1': 'y1',
    'Type 2': 'y2',
    'Type 3': 'y3',
    'Type 4': 'y4',
}

# Text columns used for TF-IDF embedding
TEXT_COLS = ['Ticket Summary', 'Interaction content']

# Dependent variable columns to classify (Type 1 excluded — only 1 class per file)
TYPE_COLS = ['y2', 'y3', 'y4']

# Columns used for chained multi-output (Design Choice 1)
CHAIN_COLS = ['y2', 'y3', 'y4']

# Minimum number of samples a class must have to be included
MIN_CLASS_SAMPLES = 3

# TF-IDF max features
TFIDF_MAX_FEATURES = 2000

# Random Forest parameters
RF_N_ESTIMATORS = 1000
RF_CLASS_WEIGHT = 'balanced_subsample'
RF_RANDOM_STATE = 42

# Train/test split
TEST_SIZE = 0.2
SPLIT_RANDOM_STATE = 42

# Data file paths (update if needed)
DATA_FILES = [
    'AppGallery.csv',
    'Purchasing.csv',
]
