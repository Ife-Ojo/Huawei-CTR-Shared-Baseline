# Huawei TechArena CTR Model: Shared Baseline Repository

**Official Leaderboard ROC AUC:** `0.810459` (Phase 15C Champion)  
**Target Goal:** `0.830000+`  
**Platform Compatibility:** Windows 11 / Windows 10 (Linux / macOS also supported)  
**Hardware Profile:** Intel Core i5-13500H, 8GB–16GB RAM  

---

## 1. Project Purpose

This repository provides a self-contained, fully reproducible distribution of our **Phase 15C Champion CTR Model**. 

Teammates can clone this repository to:
1. Immediately reproduce the exact Phase 15C leaderboard predictions with a single command.
2. Inspect the production model architecture, feature extraction pipelines, and event calibration mechanisms.
3. Rapidly experiment with new features, loss functions, and ensemble strategies on dedicated feature branches.
4. Validate improvements against our verified local benchmarks (Fold A, Fold B, Frozen Day 2, and Two-Day Pooled backtest).

---

## 2. Architecture Overview

Phase 15C achieves **0.810459 Global ROC AUC** by unifying a 3-way diverse tree ensemble with within-event discrete choice modeling and SVD-informed slate intent prediction:

```
[Candidate Features (98F)] ---> LightGBM (45%)  \
                           ---> XGBoost (30%)   ---> Core 3-Way Blend (p_core)
                           ---> CatBoost (25%)  /
                                     |
                          [Specialist Routing] (u_history_length <= 15: 70% p_core + 30% Specialist)
                                     |
                          [Method C Event Calibration] (gamma = 0.40 with 36F Event Classifier)
                                     |
                            p_Phase12c (50%)
                                     |
                                     +-------------------> FINAL BLEND (50% / 50%)
                                     |                     p_final = 0.50 * p_Phase12c + 0.50 * p_cond_up
                            p_cond_up (50%)
                                     |
             +-----------------------+-----------------------+
             |                                               |
[Phase 14 Conditional Choice]                 [Phase 15 SVD Slate Event Intent]
Custom Categorical Softmax                     50-Feature LightGBM Event Classifier
on 10-candidate slates                         (36 context + 14 collaborative SVD geometry)
--> p_choice (within-event item share)         --> q_event_up (P(at least one click in slate))
             \                                               /
              +---------------> p_cond_up <-----------------+
                                (q_event_up * p_choice)
```

---

## 3. Environment & Installation

### Requirements:
- Python 3.10, 3.11, or 3.12 (Python 3.12 recommended)
- Git & Git LFS (for large binary models and Parquet caches)

### Quick Setup (Windows PowerShell):
```powershell
# 1. Clone the repository
git clone <REPO_URL> Huawei-CTR-Shared-Baseline
cd Huawei-CTR-Shared-Baseline

# 2. Pull Git LFS artifacts
git lfs install
git lfs pull

# 3. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Directory Structure

```
Huawei-CTR-Shared-Baseline/
├── README.md                          # Documentation and quickstart
├── requirements.txt                   # Dependency specifications
├── .gitignore                         # Standard exclusion rules
├── .gitattributes                     # Git LFS tracking configuration
├── models/
│   ├── phase8c_full_warm_matched_98f.txt  # Canonical 98 feature names
│   └── checkpoints/
│       ├── phase8n_production_lightgbm_98f.txt              # Candidate LightGBM (2.2 MB)
│       ├── phase8n_production_xgboost_98f.json              # Candidate XGBoost (1.9 MB)
│       ├── phase12c_production_catboost_98f.cbm             # Candidate CatBoost (0.9 MB)
│       ├── phase8s_production_specialist_short_med_lgb.txt # Specialist Router (1.3 MB)
│       ├── phase8k_production_event_classifier_lgb.txt      # Method C Event Model (0.4 MB)
│       ├── phase14_production_conditional_choice_lgb.txt   # Conditional Choice Model (3.0 MB)
│       ├── phase15_production_svd_event_classifier_lgb.txt  # SVD Slate Event Intent (0.9 MB)
│       └── metric_svd64_jan25_s42.pt                        # Bilinear Metric Weights (7 KB)
├── scripts/
│   ├── predict_phase15c.py            # Single-command inference & reproduction
│   ├── train_phase15c_models.py       # Modular training pipeline
│   ├── generate_eval_features.py      # Feature cache generation check
│   └── verify_submission_integrity.py # Integrity and format checker
├── src/
│   ├── utils.py                       # Path resolution and metrics
│   ├── refined_svd.py                 # Bilinear Procrustes metric model
│   ├── embeddings.py                  # SVD snapshot loaders
│   ├── features.py                    # Feature computation utilities
│   ├── event_model.py                 # Event-level aggregation logic
│   └── validate.py                    # Grouped and global AUC metrics
├── data/
│   ├── raw/
│   │   └── evaluation.csv             # Test evaluation metadata (529,390 rows)
│   └── processed/
│       └── features/
│           ├── eval_98f_features.parquet   # Precomputed 98 candidate features
│           ├── eval_features.parquet       # Precomputed base context features
│           └── svd_snapshots/
│               ├── svd_asof_1737936000_k64_dmin2p0_s42.npz # Eval SVD snapshot
│               └── svd_asof_1737849600_k64_dmin2p0_s42.npz # Jan 26 snapshot
└── outputs/
    ├── phase12c_final_3model_champion_submission.csv        # Phase 12C base submission
    └── phase15c_final_50pct_svd_event_intent_submission.csv # Official Champion Benchmark
```

---

## 5. One-Command Prediction & Reproduction

To reproduce the official champion submission file directly:

```powershell
python scripts/predict_phase15c.py
```

### Verification Output:
The script executes in ~10 seconds and verifies the SHA-256 hash byte-for-byte:
```
================================================================================
PHASE 15C OFFICIAL CHAMPION PREDICTION PIPELINE (OFFICIAL SCORE: 0.810459)
================================================================================
1. Canonical feature schema loaded (98 features).
2. Loaded evaluation metadata (529,390 rows, 52,939 events).
3. Loaded Phase 12C Champion base predictions (Mean: 0.072353).
4. Computing Event Intent features (36 base + 14 slate SVD = 50 features)...
   Event intent predicted: Mean=0.654508, Std=0.134258
5. Predicting within-event conditional item choice (Softmax)...
6. Blending 50% Phase 12C + 50% (q_event_up * p_choice)...
7. Submission written to outputs/phase15c_final_50pct_svd_event_intent_submission.csv.
--------------------------------------------------------------------------------
Generated SHA-256: 0e992d347d2b01e3e17ad05280f69491ce8e411ec4008cbd4fa0d23d46e4da9c
Expected SHA-256:  0e992d347d2b01e3e17ad05280f69491ce8e411ec4008cbd4fa0d23d46e4da9c
Exact Byte-for-Byte Match: True
--------------------------------------------------------------------------------
>> SUCCESS: Phase 15C Champion perfectly reproduced in 10.53s! <<
```

To run integrity checks on any submission file:
```powershell
python scripts/verify_submission_integrity.py outputs/phase15c_final_50pct_svd_event_intent_submission.csv
```

---

## 6. Development Workflow & Git Best Practices

To keep the repository clean and avoid accidental overwrites of baseline checkpoints:

1. **Always develop on a separate branch:**
   ```powershell
   git checkout -b feature/your-experiment-name
   ```
2. **Never commit directly to `main`:**
   Push your feature branch and open a Pull Request (PR) for review.
3. **Evaluating New Models:**
   - Always measure performance against our local benchmarks:
     - **Fold A (Jan 25)**: Baseline AUC `0.81754284`
     - **Frozen Day 2 (Jan 26)**: Baseline AUC `0.79976634`
     - **Two-Day Pooled Backtest**: Baseline AUC `0.81411177`
   - Test whether your model provides complementary signal when blended with Phase 15C before submitting to the official leaderboard.

---

## 7. Hardware & Memory Guidelines

- **Peak Memory**: ~1.5 GB RAM during inference; ~4.0 GB during model training.
- **Inference Runtime**: ~10–12 seconds on an Intel i5 CPU with 4 worker threads.
- **Data Safety**: All paths are resolved dynamically using `src/utils.py:resolve_data_paths()`, ensuring zero hardcoded local directory paths.
