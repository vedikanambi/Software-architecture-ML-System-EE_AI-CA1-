# Multi-Label Email Classification Architecture

MSc AI Engineering — Evaluating Artificial Intelligence  
Continuous Assessment — Vedika Thirumalai Nambi(24222437), Diwakar Gunasekaran (24201723)


## Project Structure


email_classifier
│
├── Config.py                    # Shared constants (column names, RF params, etc.)
├── utils.py                     # Reusable helper functions
├── preprocess.py                # Data loading, deduplication, noise removal
├── embeddings.py                # TF-IDF text-to-numeric conversion
│
├── model/
│   ├── base.py                  # Abstract BaseModel (train, predict, print_results)
│   └── randomforest.py          # RandomForest implementing BaseModel
│
├── modelling/
│   ├── data_model.py            # Data encapsulation class (X_train/test, y_train/test)
│   ├── modelling.py             # Design Choice 1: Chained Multi-Output logic
│   └── hierarchical.py         # Design Choice 2: Hierarchical Modelling logic
│
├── main.py                      # Controller — Design Choice 1 (run this)
├── main_hierarchical.py         # Controller — Design Choice 2 (run this)
├── requirements.txt
└── README.md



## Setup

data files are placed in the project root:
- `AppGallery.csv`
- `Purchasing.csv`


### Design Choice 1: Chained Multi-Output
- ONE RandomForest instance per chaining level (3 total)
- Labels are concatenated: `Type2` → `Type2+Type3` → `Type2+Type3+Type4`
- Accuracy cascades downward: `L1 ≥ L2 ≥ L3`

### Design Choice 2: Hierarchical Modelling
- MULTIPLE RandomForest instances in a tree structure
- Level 0: 1 model classifies Type 2 on full data
- Level 1: N models (one per Type 2 class) classify Type 3 on filtered subsets
- Level 2: M models (one per Type2×Type3 pair) classify Type 4 on further filtered subsets


## Key Architecture Principles

| Principle | Implementation |
|---|---|
| Separation of Concerns | `preprocess.py` and `embeddings.py` are fully independent of modelling |
| Data Encapsulation | `Data` class wraps all train/test arrays — same object for every model |
| Abstraction | `BaseModel` enforces `train()`, `predict()`, `print_results()` on all models |
| Configuration | `Config.py` is the single source of truth for all constants |
