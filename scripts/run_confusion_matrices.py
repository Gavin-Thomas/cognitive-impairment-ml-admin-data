# fmt: off
"""
CONFUSION MATRIX EXTRACTION SCRIPT
Trains only specific models on specific tasks to extract confusion matrices.
"""

import os

# Project root for data and output paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["PYTHONUNBUFFERED"] = "1"

# Same threading settings as original
N_THREADS_PER_MODEL = 1
os.environ["OMP_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["XGB_NUM_THREADS"] = str(N_THREADS_PER_MODEL)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
import time

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: XGBoost not available")

# ============================================================================
# EXACT SAME SETTINGS AS ORIGINAL_PARALLEL.PY
# ============================================================================
N_CV_SPLITS_GRIDSEARCH = 3
TEST_SET_SIZE = 0.3
RANDOM_STATE = 42
GRIDSEARCH_SCORING_METRIC = 'balanced_accuracy'

# --- Same exclusion columns as original ---
EXCLUDE_COLUMNS_FOR_X = sorted(list(set([
    "id", "prompt_id", "prompt_visitdate", "visit_year", "jaak_hosp_code", "dob",
    "dementia", "cognitive_label", "dem/norm/oth", "jaak_Dem", "cognitive_status",
    "jaak_rx_filled", "jaak_3claims+_2yr_30days",
    "race", "education_years", "education_level", "sex", "age",
    "moca_total", "mmsetotal",
    "living_arrangement",
])))

# --- Data preparation functions (same as original) ---
# NOTE: Imputation and encoding moved to preprocess_after_split() to prevent data leakage.
def prepare_data_common(subset_df, label_definition, exclude_cols):
    """Drops excluded columns and extracts features/labels. No imputation or encoding."""
    if subset_df is None or subset_df.empty:
        return None, None, None

    cols_to_drop_existing = [col for col in exclude_cols if col in subset_df.columns]
    X_df = subset_df.drop(columns=cols_to_drop_existing, errors='ignore')
    y_series = subset_df["cognitive_status"]

    y_series = y_series.loc[X_df.index]
    return X_df, y_series, None


def preprocess_after_split(X_train_df, X_test_df, label_definition):
    """Impute and one-hot encode AFTER split. Fits on training data only.

    Returns: (X_train_encoded, X_test_encoded, feature_names_encoded) or (None, None, None)
    """
    if X_train_df is None or X_test_df is None:
        return None, None, None

    X_train = X_train_df.copy()
    X_test = X_test_df.copy()

    # Numeric imputation (fit medians on TRAINING data only)
    numeric_cols = X_train.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        for col in numeric_cols:
            median_val = X_train[col].median()
            if pd.isna(median_val):
                median_val = 0
            if X_train[col].isnull().any():
                X_train[col] = X_train[col].fillna(median_val)
            if X_test[col].isnull().any():
                X_test[col] = X_test[col].fillna(median_val)

    # Categorical imputation
    for df_part in [X_train, X_test]:
        categorical_cols = df_part.select_dtypes(include=['object', 'category']).columns
        if not categorical_cols.empty:
            for col in categorical_cols:
                df_part[col] = df_part[col].fillna('missing').astype(str)
                df_part[col] = df_part[col].replace(['nan', 'NaN', 'None', '', 'None'], 'missing')

    # One-hot encode (fit on training categories, align test to match)
    try:
        cols_to_encode_train = X_train.select_dtypes(include=['object', 'category']).columns
        cols_to_encode_test = X_test.select_dtypes(include=['object', 'category']).columns

        if not cols_to_encode_train.empty or not cols_to_encode_test.empty:
            X_train_encoded = pd.get_dummies(X_train, columns=cols_to_encode_train,
                                              drop_first=True, dummy_na=False, dtype=int)
            X_test_encoded = pd.get_dummies(X_test, columns=cols_to_encode_test,
                                             drop_first=True, dummy_na=False, dtype=int)
            # Align test columns to match training columns exactly
            train_cols = X_train_encoded.columns
            for col in train_cols:
                if col not in X_test_encoded.columns:
                    X_test_encoded[col] = 0
            X_test_encoded = X_test_encoded[train_cols]
        else:
            X_train_encoded = X_train
            X_test_encoded = X_test

        feature_names_encoded = list(X_train_encoded.columns)
    except Exception as e:
        print(f"    ERROR during encoding for '{label_definition}': {e}")
        return None, None, None

    return X_train_encoded, X_test_encoded, feature_names_encoded


def finalize_binary_data(X_encoded, y_original_series, label_positive_list):
    if X_encoded is None or y_original_series is None:
        return None, None
    label_positive_list_clean = [s.lower().strip().replace(' ', '_') for s in label_positive_list]
    y_binary = y_original_series.apply(lambda x: 1 if x in label_positive_list_clean else 0).values
    return X_encoded, y_binary


def finalize_multiclass_data(X_encoded, y_original_series):
    if X_encoded is None or y_original_series is None:
        return None, None, None

    def map_to_three_class(status):
        if "dementia" in status: return 2
        elif "mci" in status: return 1
        elif "normal" in status: return 0
        else: return -1

    y_multi = y_original_series.apply(map_to_three_class).values
    valid_indices = (y_multi != -1)

    if np.sum(~valid_indices) > 0:
        X_encoded_filtered = X_encoded[valid_indices]
        y_multi_filtered = y_multi[valid_indices]
        if X_encoded_filtered.empty:
            return None, None, None
        X_final = X_encoded_filtered
        y_final = y_multi_filtered
    else:
        X_final = X_encoded
        y_final = y_multi

    all_orig_numeric_labels = sorted(list(set(y_final)))
    return X_final, y_final, all_orig_numeric_labels


def main():
    """Train specific models on specific tasks and extract confusion matrices."""
    print("=" * 60)
    print("CONFUSION MATRIX EXTRACTION")
    print("Training only selected models for CM extraction")
    print("=" * 60)

    # --- Load Data (same as original) ---
    data_dir = os.path.join(PROJECT_ROOT, "data")
    possible_data_paths = [
        os.path.join(data_dir, "MAIN_new.csv"),
        os.path.join(data_dir, "MAIN.csv"),
    ]
    data_path = None
    for p in possible_data_paths:
        if os.path.exists(p):
            data_path = p
            break

    if data_path is None:
        print(f"ERROR: Data file not found in {data_dir}")
        return

    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    df['cognitive_status'] = df['cognitive_status'].fillna('unknown').astype(str)
    df['cognitive_status'] = df['cognitive_status'].str.lower().str.strip().str.replace(' ', '_', regex=False)

    # --- Define the 3 specific tasks we need ---
    tasks_to_run = {
        "1. Def Normal vs. Def Dementia": {
            "filter": ["definite_normal", "definite_dementia"],
            "is_multi": False,
            "pos_labels": ["definite_dementia"],
            "class_names": ['Normal', 'Dementia'],
            "models": ["Random Forest (Entropy)"]
        },
        "3. Def Normal vs. Def MCI": {
            "filter": ["definite_normal", "definite_mci"],
            "is_multi": False,
            "pos_labels": ["definite_mci"],
            "class_names": ['Normal', 'MCI'],
            "models": ["Random Forest (Entropy)"]
        },
        "5. Multi: Def Normal vs. Def MCI vs. Def Dementia": {
            "filter": ["definite_normal", "definite_mci", "definite_dementia"],
            "is_multi": True,
            "pos_labels": None,
            "class_names": ['Normal', 'MCI', 'Dementia'],
            "models": ["XGBoost"]
        }
    }

    # --- Output directory ---
    output_dir = os.path.join(PROJECT_ROOT, "output", "confusion_matrices")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # --- Process each task ---
    results_summary = []

    for task_name, task_config in tasks_to_run.items():
        print(f"\n{'='*60}")
        print(f"TASK: {task_name}")
        print('='*60)

        # Filter data
        df_subset = df[df["cognitive_status"].isin(task_config["filter"])].copy()
        print(f"  Samples after filter: {len(df_subset)}")

        # Step 1: Prepare data (no imputation/encoding yet)
        X_raw_df, y_ser, _ = prepare_data_common(df_subset, task_name, EXCLUDE_COLUMNS_FOR_X)
        if X_raw_df is None:
            print(f"  ERROR: Data preparation failed")
            continue

        # Step 2: Finalize labels
        if task_config["is_multi"]:
            X_final_df, y_final_np, all_orig_labels = finalize_multiclass_data(X_raw_df, y_ser)
            n_classes = 3
        else:
            X_final_df, y_final_np = finalize_binary_data(X_raw_df, y_ser, task_config["pos_labels"])
            all_orig_labels = [0, 1]
            n_classes = 2

        if X_final_df is None:
            print(f"  ERROR: Label finalization failed")
            continue

        # Step 3: Split RAW DataFrames - SAME RANDOM_STATE ensures identical split
        X_train_raw_df, X_test_raw_df, y_train, y_test = train_test_split(
            X_final_df, y_final_np,
            test_size=TEST_SET_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_final_np
        )

        # Step 4: Impute and encode AFTER split (fit on training data only)
        X_train_enc_df, X_test_enc_df, feat_names = preprocess_after_split(
            X_train_raw_df, X_test_raw_df, task_name
        )
        if X_train_enc_df is None:
            print(f"  ERROR: Preprocessing after split failed")
            continue

        X_train = X_train_enc_df.values
        X_test = X_test_enc_df.values
        print(f"  Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, Features: {X_train.shape[1]}")

        # Train each model for this task
        for model_name in task_config["models"]:
            print(f"\n  Training: {model_name}")
            start_time = time.time()

            if model_name == "Random Forest (Entropy)":
                # EXACT same pipeline and param_grid as original
                model_pipeline = Pipeline([
                    ('randomforestclassifier', RandomForestClassifier(
                        random_state=RANDOM_STATE,
                        class_weight='balanced_subsample',
                        criterion='entropy',
                        n_estimators=300,
                        max_depth=20,
                        min_samples_leaf=3,
                        min_samples_split=5,
                        max_features='sqrt',
                        n_jobs=N_THREADS_PER_MODEL,
                        oob_score=True
                    ))
                ])
                param_grid = {
                    'randomforestclassifier__n_estimators': [200, 400],
                    'randomforestclassifier__max_depth': [15, 25, None],
                    'randomforestclassifier__min_samples_leaf': [2, 5],
                    'randomforestclassifier__min_samples_split': [3, 5, 10]
                }

            elif model_name == "XGBoost" and XGB_AVAILABLE:
                # EXACT same pipeline and param_grid as original (multiclass)
                model_pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('xgbclassifier', XGBClassifier(
                        objective='multi:softprob',
                        num_class=n_classes,
                        random_state=RANDOM_STATE,
                        eval_metric='mlogloss',
                        n_jobs=N_THREADS_PER_MODEL,
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_alpha=0.1,
                        reg_lambda=1.0,
                        min_child_weight=3
                    ))
                ])
                param_grid = {
                    'xgbclassifier__n_estimators': [150, 250],
                    'xgbclassifier__learning_rate': [0.02, 0.05, 0.1],
                    'xgbclassifier__max_depth': [3, 5, 7],
                    'xgbclassifier__min_child_weight': [1, 3, 5],
                    'xgbclassifier__subsample': [0.7, 0.85],
                    'xgbclassifier__colsample_bytree': [0.7, 0.85]
                }
            else:
                print(f"    Skipping {model_name} (not available)")
                continue

            # GridSearchCV - SAME settings as original
            cv_grid = StratifiedKFold(n_splits=N_CV_SPLITS_GRIDSEARCH, shuffle=True, random_state=RANDOM_STATE)
            grid_search = GridSearchCV(
                estimator=clone(model_pipeline),
                param_grid=param_grid,
                scoring=GRIDSEARCH_SCORING_METRIC,
                cv=cv_grid,
                n_jobs=1,
                refit=True
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            elapsed = time.time() - start_time

            print(f"    Training complete in {elapsed:.1f}s")
            print(f"    Best params: {grid_search.best_params_}")

            # Get predictions
            y_pred = best_model.predict(X_test)

            # Generate confusion matrix (raw counts)
            labels = list(range(n_classes))
            cm = confusion_matrix(y_test, y_pred, labels=labels)

            # Generate normalized confusion matrix (by true label / row)
            cm_normalized = confusion_matrix(y_test, y_pred, labels=labels, normalize='true')

            # Print confusion matrices
            class_names = task_config["class_names"]
            cm_df = pd.DataFrame(cm,
                                index=[f"True_{n}" for n in class_names],
                                columns=[f"Pred_{n}" for n in class_names])
            cm_norm_df = pd.DataFrame(cm_normalized,
                                index=[f"True_{n}" for n in class_names],
                                columns=[f"Pred_{n}" for n in class_names])

            print(f"\n    Confusion Matrix (Raw Counts):")
            print(cm_df.to_string())
            print(f"\n    Confusion Matrix (Normalized by True Label):")
            print(cm_norm_df.round(3).to_string())

            # Calculate metrics from CM
            if n_classes == 2:
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                print(f"\n    Sensitivity: {sensitivity:.3f}")
                print(f"    Specificity: {specificity:.3f}")

            # Save CSVs
            safe_name = f"{model_name.replace(' ', '_').replace('(', '').replace(')', '')}_{task_name.replace(' ', '_').replace(':', '').replace('.', '')}"
            csv_path = os.path.join(output_dir, f"cm_{safe_name}_counts.csv")
            csv_norm_path = os.path.join(output_dir, f"cm_{safe_name}_normalized.csv")
            cm_df.to_csv(csv_path)
            cm_norm_df.round(4).to_csv(csv_norm_path)

            # Save normalized confusion matrix image
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
            ax.figure.colorbar(im, ax=ax)
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_names, yticklabels=class_names,
                   title=f'Normalized Confusion Matrix\n{model_name}\n{task_name}',
                   ylabel='True Label',
                   xlabel='Predicted Label')
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            # Add annotations (show percentage AND count)
            thresh = 0.5
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    pct = cm_normalized[i, j]
                    count = cm[i, j]
                    ax.text(j, i, f'{pct:.1%}\n(n={count})',
                            ha="center", va="center",
                            color="white" if pct > thresh else "black",
                            fontsize=11, fontweight='bold')
            fig.tight_layout()

            img_path = os.path.join(output_dir, f"cm_{safe_name}_normalized.png")
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"\n    Saved: {csv_path}")
            print(f"    Saved: {csv_norm_path}")
            print(f"    Saved: {img_path}")

            results_summary.append({
                "Task": task_name,
                "Model": model_name,
                "Time (s)": f"{elapsed:.1f}",
                "Best Params": str(grid_search.best_params_)
            })

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nOutput saved to: {output_dir}")
    print("\nResults:")
    for r in results_summary:
        print(f"  - {r['Model']} on {r['Task']}: {r['Time (s)']}s")

    print("\nFiles generated:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")


if __name__ == "__main__":
    main()
# fmt: on
