# ============================================================
# embeddings.py — Text to numeric feature extraction (TF-IDF)
# Independent of all modelling code (Separation of Concerns)
# ============================================================

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from Config import TEXT_COLS, TFIDF_MAX_FEATURES


def get_tfidf_embd(df, max_features=TFIDF_MAX_FEATURES):
    """
    Combine text columns and convert to a TF-IDF numeric matrix.

    Concatenates 'Ticket Summary' and 'Interaction content' for each row,
    then fits and transforms using TfidfVectorizer.

    Args:
        df: Cleaned DataFrame containing TEXT_COLS columns.
        max_features: Maximum number of TF-IDF features (default: 2000).

    Returns:
        X: numpy ndarray of shape (n_samples, max_features).
        vectorizer: Fitted TfidfVectorizer (kept for inspection/reuse).
    """
    # Combine all text columns into one string per row
    combined_text = df[TEXT_COLS[0]].fillna('') + ' ' + df[TEXT_COLS[1]].fillna('')
    combined_text = combined_text.str.lower().str.strip()

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,       # Apply log normalization
        strip_accents='unicode',
        analyzer='word',
        stop_words='english',
    )

    X = vectorizer.fit_transform(combined_text).toarray()
    print(f"[embeddings] TF-IDF matrix shape: {X.shape}")
    return X, vectorizer
