# fmt: off
"""
SECONDARY ANALYSIS: Including Age and Sex as Predictors

"""

import os
import sys

# Add project root to Python path so scripts can import from utils/ and scripts/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

os.environ["PYTHONUNBUFFERED"] = "1"

# HIGH-PERFORMANCE SERVER OPTIMIZATION (32 CPUs / 100GB RAM)
# Must match run_primary_analysis.py settings
N_PARALLEL_MODELS = 8
N_THREADS_PER_MODEL = 2

os.environ["OMP_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["MKL_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["XGB_NUM_THREADS"] = str(N_THREADS_PER_MODEL)
os.environ["LGBM_NUM_THREADS"] = str(N_THREADS_PER_MODEL)

import numpy as np
import pandas as pd
import collections
import time
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')

from sklearn.model_selection import train_test_split
from sklearn.base import clone
from joblib import Parallel, delayed, dump as joblib_dump

# Import everything from the main pipeline
from scripts.run_primary_analysis import (
    # Configuration constants
    N_BOOTSTRAPS, N_CV_SPLITS_GRIDSEARCH, N_CV_SPLITS_THRESH,
    N_CV_REPEATS_THRESH, TEST_SET_SIZE, RANDOM_STATE, MCNEMAR_ALPHA,
    N_JOBS_GRIDSEARCH, N_JOBS_BOOTSTRAP, N_JOBS_MODELS,
    OPTIMIZE_THRESHOLD_METRIC, GRIDSEARCH_SCORING_METRIC,
    # Data preparation
    prepare_data_common, preprocess_after_split,
    finalize_binary_data, finalize_multiclass_data,
    # Model evaluation
    evaluate_single_model_task, evaluate_benchmark_rule,
    detect_class_imbalance,
    # Results processing
    create_comprehensive_metrics_table,
    generate_auc_charts, generate_comparison_charts,
)

# Optional imports (same as original)
try:
    from utils.publication_visualizations import run_all_visualizations, SHAP_AVAILABLE
    VISUALIZATIONS_AVAILABLE = True
except ImportError:
    VISUALIZATIONS_AVAILABLE = False

try:
    from utils.statistical_tests import run_statistical_comparisons, extract_all_feature_importances
    STATISTICAL_TESTS_AVAILABLE = True
except ImportError:
    STATISTICAL_TESTS_AVAILABLE = False

# Import model definitions from original (we need the pipeline/grid objects)
from scripts.run_primary_analysis import (
    XGBClassifier, LGBMClassifier,
    LGBM_AVAILABLE, SMOTE_AVAILABLE,
)

# We need to rebuild model pipelines since they reference local variables in main()
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

if SMOTE_AVAILABLE:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    from scripts.run_primary_analysis import create_smote_pipeline, create_adaptive_smote


def main():
    """
    Secondary analysis: same pipeline as original but WITH age and sex as features.
    """
    print("=" * 70)
    print("SECONDARY ANALYSIS: Including Age and Sex as Predictors")
    print("=" * 70)
    print(f"Using {N_JOBS_MODELS} parallel model evaluations")
    print(f"Using {N_JOBS_BOOTSTRAP} parallel bootstrap workers per model")
    print()

    analysis_start_time = time.time()

    # --- Find Data ---
    data_dir = os.path.join(PROJECT_ROOT, "data")
    possible_data_paths = [
        os.path.join(data_dir, "MAIN_new.csv"),
        os.path.join(data_dir, "MAIN.csv"),
    ]
    data_path = None
    for p in possible_data_paths:
        if os.path.exists(p):
            data_path = p
            print(f"Found data file at: {os.path.abspath(p)}")
            break

    if data_path is None:
        print(f"CRITICAL ERROR: CSV data file not found in {data_dir}")
        print("Place MAIN_new.csv or MAIN.csv in the data/ directory.")
        return

    # --- Output Directories ---
    base_output_dir = os.path.join(PROJECT_ROOT, "output", "secondary")
    charts_dir = os.path.join(base_output_dir, "charts_final_eval")
    results_dir = os.path.join(base_output_dir, "results_final_eval")

    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    print(f"Output will be saved in: {base_output_dir}")

    # --- Load Data ---
    print("\n=== Loading Data ===")
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        df['cognitive_status'] = df['cognitive_status'].fillna('unknown').astype(str)
        df['cognitive_status'] = df['cognitive_status'].str.lower().str.strip().str.replace(' ', '_', regex=False)
        print("Value counts for 'cognitive_status':")
        print(df['cognitive_status'].value_counts())
    except Exception as e:
        print(f"CRITICAL ERROR loading data: {e}")
        return

    # --- Define Exclusions (KEY DIFFERENCE: age and sex are NOT excluded) ---
    exclude_columns_for_X = sorted(list(set([
        # Identifiers and dates
        "id", "prompt_id", "prompt_visitdate", "visit_year", "jaak_hosp_code", "dob",
        # Label/outcome columns (leakage)
        "dementia", "cognitive_label", "dem/norm/oth", "jaak_Dem", "cognitive_status",
        "jaak_rx_filled", "jaak_3claims+_2yr_30days",
        # Demographics PARTIALLY excluded (age and sex NOW INCLUDED as predictors)
        "race", "education_years", "education_level",
        # "sex",   <-- NOW INCLUDED
        # "age",   <-- NOW INCLUDED
        # Cognitive test scores (label leakage - used to determine cognitive_status)
        "moca_total", "mmsetotal",
        # Other non-predictive variables
        "living_arrangement",
    ])))

    print("\n*** SECONDARY ANALYSIS: age and sex INCLUDED as predictors ***")
    print(f"Excluded columns ({len(exclude_columns_for_X)}): {exclude_columns_for_X}")

    # --- Prepare Task Subsets (same 6 tasks as primary analysis) ---
    print("\n=== Preparing Task Subsets ===")
    task_dfs = collections.OrderedDict()
    task_dfs["1. Def Normal vs. Def Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "definite_dementia"])].copy()
    task_dfs["2. Def+Pos Normal vs. Def+Pos Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_dementia", "possible_dementia"])].copy()
    task_dfs["3. Def Normal vs. Def MCI"] = df[df["cognitive_status"].isin(["definite_normal", "definite_mci"])].copy()
    task_dfs["4. Def+Pos Normal vs. Def+Pos MCI"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_mci", "possible_mci"])].copy()
    task_dfs["5. Multi: Def Normal vs. Def MCI vs. Def Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "definite_mci", "definite_dementia"])].copy()
    task_dfs["6. Multi: Def+Pos Normal vs. Def+Pos MCI vs. Def+Pos Dementia"] = df[df["cognitive_status"].isin(["definite_normal", "possible_normal", "definite_mci", "possible_mci", "definite_dementia", "possible_dementia"])].copy()

    # --- Process and Split Data ---
    print("\n=== Preprocessing and Splitting Data ===")
    task_data = {}
    feature_names_dict_final = {}

    JAAK_COLS = ['jaak_hosp_code', 'jaak_rx_filled', 'jaak_3claims+_2yr_30days']

    for label_def, df_subset in task_dfs.items():
        print(f"\n  Processing: {label_def}")
        if df_subset.empty:
            continue

        is_dementia_vs_normal = ("Dementia" in label_def and "Normal" in label_def
                                  and "Multi:" not in label_def and "MCI" not in label_def)

        if is_dementia_vs_normal:
            result = prepare_data_common(df_subset, label_def, exclude_columns_for_X,
                                         jaak_cols=JAAK_COLS)
            if result[0] is None:
                continue
            X_raw_df, y_ser, _, Z_df = result
        else:
            result = prepare_data_common(df_subset, label_def, exclude_columns_for_X)
            if result[0] is None:
                continue
            X_raw_df, y_ser, _ = result
            Z_df = None

        is_multi = "Multi:" in label_def
        all_orig_numeric_labels_task = None

        if is_multi:
            X_final_df, y_final_np, all_orig_numeric_labels_task = finalize_multiclass_data(X_raw_df, y_ser)
            Z_final_df = None
        else:
            pos_labels_str = []
            if "Dementia" in label_def and "Normal" in label_def:
                pos_labels_str = ["definite_dementia", "possible_dementia"]
            elif "MCI" in label_def and "Normal" in label_def:
                pos_labels_str = ["definite_mci", "possible_mci"]
            if not pos_labels_str:
                continue
            X_final_df, y_final_np = finalize_binary_data(X_raw_df, y_ser, pos_labels_str)

            if Z_df is not None and X_final_df is not None:
                Z_final_df = Z_df.loc[X_final_df.index]
            else:
                Z_final_df = None

        if X_final_df is None or y_final_np is None:
            continue

        try:
            if Z_final_df is not None:
                X_train_raw_df, X_test_raw_df, y_train, y_test, Z_train_df, Z_test_df = train_test_split(
                    X_final_df, y_final_np, Z_final_df,
                    test_size=TEST_SET_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_final_np
                )
                Z_test = Z_test_df.values
            else:
                X_train_raw_df, X_test_raw_df, y_train, y_test = train_test_split(
                    X_final_df, y_final_np,
                    test_size=TEST_SET_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_final_np
                )
                Z_test = None

            X_train_enc_df, X_test_enc_df, feat_names = preprocess_after_split(
                X_train_raw_df, X_test_raw_df, label_def
            )
            if X_train_enc_df is None:
                print(f"    ERROR: Preprocessing after split failed for '{label_def}'")
                continue

            feature_names_dict_final[label_def] = feat_names

            X_train = X_train_enc_df.values
            X_test = X_test_enc_df.values

            task_data[label_def] = (X_train, y_train, X_test, y_test, all_orig_numeric_labels_task, Z_test)
            print(f"    Split complete: Train={X_train.shape[0]}, Test={X_test.shape[0]}, Features={X_train.shape[1]}")
            if Z_test is not None:
                print(f"    Jaakkimainen benchmark data extracted: Z_test shape={Z_test.shape}")

            # Verify age/sex are present in feature names
            age_present = any('age' in f.lower() for f in feat_names)
            sex_present = any('sex' in f.lower() for f in feat_names)
            print(f"    Age in features: {age_present}, Sex in features: {sex_present}")

        except Exception as e:
            print(f"    ERROR splitting/preprocessing: {e}")
            continue

    if not task_data:
        print("CRITICAL ERROR: No task datasets prepared. Exiting.")
        return

    print(f"\nSuccessfully prepared {len(task_data)} tasks.")

    # --- Define Models (same as original_parallel.py) ---
    print("\n=== Defining Models ===")
    max_iter_lr = 3000

    lr_pipeline = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='liblinear', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced', penalty='l1'))])
    rf_pipeline = Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))])
    gbm_pipeline = Pipeline([('gradientboostingclassifier', GradientBoostingClassifier(random_state=RANDOM_STATE))])
    svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))])

    if XGBClassifier:
        xgb_pipeline = Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=N_THREADS_PER_MODEL))])
    else:
        xgb_pipeline = None

    hist_gbm_binary_pipeline = Pipeline([
        ('histgradientboostingclassifier', HistGradientBoostingClassifier(
            random_state=RANDOM_STATE, class_weight='balanced', max_iter=200,
            max_depth=6, min_samples_leaf=10, learning_rate=0.05,
            l2_regularization=0.1, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15
        ))
    ])

    if LGBM_AVAILABLE and LGBMClassifier:
        lgbm_binary_pipeline = Pipeline([
            ('lgbmclassifier', LGBMClassifier(
                objective='binary', random_state=RANDOM_STATE,
                class_weight='balanced', n_estimators=200, max_depth=6,
                learning_rate=0.05, num_leaves=31, min_child_samples=10,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=0.1, n_jobs=N_THREADS_PER_MODEL, verbose=-1
            ))
        ])
    else:
        lgbm_binary_pipeline = None

    rf_entropy_binary_pipeline = Pipeline([
        ('randomforestclassifier', RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight='balanced_subsample',
            criterion='entropy', n_estimators=300, max_depth=20,
            min_samples_leaf=3, min_samples_split=5, max_features='sqrt',
            n_jobs=N_THREADS_PER_MODEL, oob_score=True
        ))
    ])

    # --- Hyperparameter Grids (same as original) ---
    lr_param_grid = {'logisticregression__C': [0.1, 0.5, 1.0, 5.0]}
    rf_param_grid = {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_depth': [10, 20, None],
        'randomforestclassifier__min_samples_leaf': [3, 5],
        'randomforestclassifier__max_features': ['sqrt', 0.5]
    }
    rf_entropy_param_grid = {
        'randomforestclassifier__n_estimators': [200, 400],
        'randomforestclassifier__max_depth': [15, 25, None],
        'randomforestclassifier__min_samples_leaf': [2, 5],
        'randomforestclassifier__min_samples_split': [3, 5, 10]
    }
    gbm_param_grid = {
        'gradientboostingclassifier__n_estimators': [100, 150],
        'gradientboostingclassifier__learning_rate': [0.05, 0.1],
        'gradientboostingclassifier__max_depth': [3, 5],
        'gradientboostingclassifier__subsample': [0.7, 1.0]
    }
    hist_gbm_param_grid = {
        'histgradientboostingclassifier__max_iter': [150, 250],
        'histgradientboostingclassifier__learning_rate': [0.02, 0.05, 0.1],
        'histgradientboostingclassifier__max_depth': [4, 6, 8],
        'histgradientboostingclassifier__min_samples_leaf': [5, 10, 20],
        'histgradientboostingclassifier__l2_regularization': [0.0, 0.1, 1.0]
    }
    svm_param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__gamma': ['scale', 'auto', 0.1, 1]
    }

    xgb_param_grid = {}
    if XGBClassifier:
        xgb_param_grid = {
            'xgbclassifier__n_estimators': [100, 200],
            'xgbclassifier__learning_rate': [0.05, 0.1],
            'xgbclassifier__max_depth': [3, 5],
            'xgbclassifier__subsample': [0.7, 1.0],
            'xgbclassifier__colsample_bytree': [0.7, 1.0]
        }

    lgbm_param_grid = {}
    if LGBM_AVAILABLE and LGBMClassifier:
        lgbm_param_grid = {
            'lgbmclassifier__n_estimators': [150, 250],
            'lgbmclassifier__learning_rate': [0.02, 0.05, 0.1],
            'lgbmclassifier__max_depth': [4, 6, 8],
            'lgbmclassifier__num_leaves': [15, 31, 63],
            'lgbmclassifier__min_child_samples': [5, 10, 20],
            'lgbmclassifier__subsample': [0.7, 0.85]
        }

    # SMOTE wrappers (same as original)
    if SMOTE_AVAILABLE:
        lr_pipeline = create_smote_pipeline(lr_pipeline, RANDOM_STATE)
        rf_pipeline = create_smote_pipeline(rf_pipeline, RANDOM_STATE)
        rf_entropy_binary_pipeline = create_smote_pipeline(rf_entropy_binary_pipeline, RANDOM_STATE)
        hist_gbm_binary_pipeline = create_smote_pipeline(hist_gbm_binary_pipeline, RANDOM_STATE)
        svm_pipeline = create_smote_pipeline(svm_pipeline, RANDOM_STATE)
        if lgbm_binary_pipeline is not None:
            lgbm_binary_pipeline = create_smote_pipeline(lgbm_binary_pipeline, RANDOM_STATE)

    # Binary models list
    binary_models = [
        ("Logistic Regression (L1)", lr_pipeline, lr_param_grid),
        ("Random Forest (Gini)", rf_pipeline, rf_param_grid),
        ("Random Forest (Entropy)", rf_entropy_binary_pipeline, rf_entropy_param_grid),
        ("Gradient Boosting", gbm_pipeline, gbm_param_grid),
        ("HistGradientBoosting", hist_gbm_binary_pipeline, hist_gbm_param_grid),
        ("SVM (RBF Kernel)", svm_pipeline, svm_param_grid),
    ]

    if xgb_pipeline is not None:
        binary_models.append(("XGBoost", xgb_pipeline, xgb_param_grid))
    if lgbm_binary_pipeline is not None:
        binary_models.append(("LightGBM", lgbm_binary_pipeline, lgbm_param_grid))

    # Voting ensemble
    voting_estimators = [
        ('lr', clone(Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='liblinear', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced', penalty='l1'))]))),
        ('rf', clone(Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))]))),
        ('hist_gbm', clone(Pipeline([('histgradientboostingclassifier', HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight='balanced'))]))),
        ('svm', clone(Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))]))),
    ]
    if lgbm_binary_pipeline is not None:
        voting_estimators.append(('lgbm', clone(Pipeline([('lgbmclassifier', LGBMClassifier(objective='binary', random_state=RANDOM_STATE, class_weight='balanced', verbose=-1, n_jobs=N_THREADS_PER_MODEL))]))))
    if xgb_pipeline is not None:
        voting_estimators.append(('xgb', clone(Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=N_THREADS_PER_MODEL))]))))

    voting_pipeline = Pipeline([('votingclassifier', VotingClassifier(estimators=voting_estimators, voting='soft', n_jobs=1))])
    binary_models.append(("Voting Ensemble", voting_pipeline, {}))

    # Stacking ensemble
    stacking_base_estimators = [
        ('lr', Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='liblinear', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced'))])),
        ('rf', Pipeline([('randomforestclassifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))])),
        ('gbm', Pipeline([('gradientboostingclassifier', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))])),
        ('svm', Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))])),
    ]
    if xgb_pipeline is not None:
        stacking_base_estimators.append(('xgb', Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=N_THREADS_PER_MODEL))])))

    stacking_meta_learner = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=max_iter_lr))])
    stacking_pipeline = Pipeline([('stackingclassifier', StackingClassifier(estimators=stacking_base_estimators, final_estimator=stacking_meta_learner, cv=3, n_jobs=1))])

    stacking_param_grid = {
        'stackingclassifier__lr__logisticregression__C': [0.1, 1.0],
        'stackingclassifier__rf__randomforestclassifier__max_depth': [10, None],
        'stackingclassifier__gbm__gradientboostingclassifier__learning_rate': [0.05, 0.1],
        'stackingclassifier__svm__svc__C': [0.1, 1],
    }
    if xgb_pipeline is not None:
        stacking_param_grid['stackingclassifier__xgb__xgbclassifier__max_depth'] = [3, 5]

    binary_models.append(("Stacking Ensemble", stacking_pipeline, stacking_param_grid))

    print(f"Defined {len(binary_models)} binary models.")

    # --- Multiclass model definitions ---
    lr_multi_pipeline = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='lbfgs', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced', penalty='l2'))])
    lr_multi_param_grid = {'logisticregression__C': [0.01, 0.1, 0.5, 1.0, 5.0]}

    svm_multi_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced', decision_function_shape='ovo'))])
    svm_multi_param_grid = {'svc__C': [0.1, 1, 10], 'svc__gamma': ['scale', 'auto', 0.01, 0.1]}

    gbm_multi_pipeline = Pipeline([('gradientboostingclassifier', GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=200, learning_rate=0.05))])
    gbm_multi_param_grid = {
        'gradientboostingclassifier__n_estimators': [150, 250],
        'gradientboostingclassifier__learning_rate': [0.02, 0.05, 0.1],
        'gradientboostingclassifier__max_depth': [3, 5, 7],
        'gradientboostingclassifier__subsample': [0.7, 0.85]
    }

    hist_gbm_multi_pipeline = Pipeline([('histgradientboostingclassifier', HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight='balanced', max_iter=200, max_depth=6, min_samples_leaf=10, learning_rate=0.05, l2_regularization=0.1, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15))])
    hist_gbm_multi_param_grid = hist_gbm_param_grid.copy()

    rf_multi_param_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300],
        'randomforestclassifier__max_depth': [10, 20, None],
        'randomforestclassifier__min_samples_leaf': [2, 5, 10],
        'randomforestclassifier__max_features': ['sqrt', 0.3, 0.5]
    }

    # SMOTE for multiclass
    if SMOTE_AVAILABLE:
        lr_multi_pipeline = create_smote_pipeline(lr_multi_pipeline, RANDOM_STATE)
        svm_multi_pipeline = create_smote_pipeline(svm_multi_pipeline, RANDOM_STATE)
        hist_gbm_multi_pipeline = create_smote_pipeline(hist_gbm_multi_pipeline, RANDOM_STATE)

    # --- Build Evaluation Jobs ---
    print("\n=== Building Parallel Evaluation Jobs ===")
    eval_jobs = []

    for label_def, task_specific_data in task_data.items():
        X_train, y_train, X_test, y_test, all_orig_labels, Z_test = task_specific_data
        current_feature_names = feature_names_dict_final.get(label_def)

        if current_feature_names is None or len(current_feature_names) != X_train.shape[1]:
            current_feature_names = [f"Feature_{i+1}" for i in range(X_train.shape[1])]

        is_multi = "Multi:" in label_def
        imbalance_info = detect_class_imbalance(y_train)

        if is_multi:
            n_classes = len(all_orig_labels) if all_orig_labels else len(np.unique(y_train))
            train_counts = np.bincount(y_train.astype(int), minlength=n_classes)
            test_counts = np.bincount(y_test.astype(int), minlength=n_classes)
            print(f"  {label_def}:")
            print(f"    Classes: {n_classes}, Train distribution: {dict(enumerate(train_counts))}")
            print(f"    Test distribution: {dict(enumerate(test_counts))}")
        else:
            print(f"  {label_def}:")
            print(f"    Train: {imbalance_info['class_counts']}, Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}:1")

        if is_multi:
            n_classes = len(all_orig_labels) if all_orig_labels else len(np.unique(y_train))

            # Build multiclass models dynamically (same as original)
            multi_models = [
                ("Logistic Regression (Multinomial)", lr_multi_pipeline, lr_multi_param_grid),
                ("Random Forest (Gini)", clone(Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))])) if not SMOTE_AVAILABLE else create_smote_pipeline(Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))]), RANDOM_STATE), rf_multi_param_grid),
                ("Random Forest (Entropy)", clone(rf_entropy_binary_pipeline) if not SMOTE_AVAILABLE else create_smote_pipeline(Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', criterion='entropy', n_estimators=300, max_depth=20, min_samples_leaf=3, min_samples_split=5, max_features='sqrt', n_jobs=N_THREADS_PER_MODEL, oob_score=True))]), RANDOM_STATE), rf_entropy_param_grid),
                ("Gradient Boosting", gbm_multi_pipeline, gbm_multi_param_grid),
                ("HistGradientBoosting", hist_gbm_multi_pipeline, hist_gbm_multi_param_grid),
                ("SVM (RBF Kernel)", svm_multi_pipeline, svm_multi_param_grid),
            ]

            # Add XGBoost for multiclass
            if XGBClassifier:
                xgb_multi_pipeline = Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(
                    objective='multi:softprob', num_class=n_classes,
                    random_state=RANDOM_STATE, eval_metric='mlogloss',
                    n_jobs=N_THREADS_PER_MODEL, n_estimators=200,
                    max_depth=4, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3
                ))])
                xgb_multi_param_grid = {
                    'xgbclassifier__n_estimators': [150, 250],
                    'xgbclassifier__learning_rate': [0.02, 0.05, 0.1],
                    'xgbclassifier__max_depth': [3, 5, 7],
                    'xgbclassifier__min_child_weight': [1, 3, 5],
                    'xgbclassifier__subsample': [0.7, 0.85],
                    'xgbclassifier__colsample_bytree': [0.7, 0.85]
                }
                multi_models.append(("XGBoost", xgb_multi_pipeline, xgb_multi_param_grid))

            # Add LightGBM for multiclass
            if LGBM_AVAILABLE and LGBMClassifier:
                lgbm_multi_base = Pipeline([('lgbmclassifier', LGBMClassifier(
                    objective='multiclass', num_class=n_classes,
                    random_state=RANDOM_STATE, class_weight='balanced',
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    num_leaves=31, min_child_samples=10, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
                    n_jobs=N_THREADS_PER_MODEL, verbose=-1
                ))])
                if SMOTE_AVAILABLE:
                    lgbm_multi_base = create_smote_pipeline(lgbm_multi_base, RANDOM_STATE)
                multi_models.append(("LightGBM", lgbm_multi_base, lgbm_param_grid))

            # Multiclass Voting
            multi_voting_estimators = [
                ('lr', clone(Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='lbfgs', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced'))]))),
                ('rf', clone(Pipeline([('randomforestclassifier', RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))]))),
                ('hist_gbm', clone(Pipeline([('histgradientboostingclassifier', HistGradientBoostingClassifier(random_state=RANDOM_STATE, class_weight='balanced'))]))),
                ('svm', clone(Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))]))),
            ]
            if LGBM_AVAILABLE and LGBMClassifier:
                multi_voting_estimators.append(('lgbm', clone(Pipeline([('lgbmclassifier', LGBMClassifier(objective='multiclass', num_class=n_classes, random_state=RANDOM_STATE, class_weight='balanced', verbose=-1, n_jobs=N_THREADS_PER_MODEL))]))))
            if XGBClassifier:
                multi_voting_estimators.append(('xgb', clone(Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(objective='multi:softprob', num_class=n_classes, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=N_THREADS_PER_MODEL))]))))

            multi_voting_pipeline = Pipeline([('votingclassifier', VotingClassifier(estimators=multi_voting_estimators, voting='soft', n_jobs=1))])
            multi_models.append(("Voting Ensemble", multi_voting_pipeline, {}))

            # Multiclass Stacking
            multi_stacking_base = [
                ('lr', Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(solver='lbfgs', max_iter=max_iter_lr, random_state=RANDOM_STATE, class_weight='balanced'))])),
                ('rf', Pipeline([('randomforestclassifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced_subsample', n_jobs=N_THREADS_PER_MODEL))])),
                ('gbm', Pipeline([('gradientboostingclassifier', GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))])),
                ('svm', Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced'))])),
            ]
            if XGBClassifier:
                multi_stacking_base.append(('xgb', Pipeline([('scaler', StandardScaler()), ('xgbclassifier', XGBClassifier(n_estimators=100, random_state=RANDOM_STATE, eval_metric='mlogloss', n_jobs=N_THREADS_PER_MODEL))])))

            multi_stacking_meta = Pipeline([('scaler', StandardScaler()), ('logisticregression', LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=max_iter_lr))])
            multi_stacking_pipeline = Pipeline([('stackingclassifier', StackingClassifier(estimators=multi_stacking_base, final_estimator=multi_stacking_meta, cv=3, n_jobs=1))])

            multi_stacking_param_grid = {
                'stackingclassifier__lr__logisticregression__C': [0.1, 1.0],
                'stackingclassifier__rf__randomforestclassifier__max_depth': [10, None],
                'stackingclassifier__gbm__gradientboostingclassifier__learning_rate': [0.05, 0.1],
                'stackingclassifier__svm__svc__C': [0.1, 1],
            }
            if XGBClassifier:
                multi_stacking_param_grid['stackingclassifier__xgb__xgbclassifier__max_depth'] = [3, 5]

            multi_models.append(("Stacking Ensemble", multi_stacking_pipeline, multi_stacking_param_grid))

            for model_name, model_instance, param_grid in multi_models:
                eval_jobs.append((
                    label_def, model_name, clone(model_instance), param_grid,
                    X_train, y_train, X_test, y_test,
                    current_feature_names, True, all_orig_labels
                ))
        else:
            # Binary models
            for model_name, model_instance, param_grid in binary_models:
                eval_jobs.append((
                    label_def, model_name, clone(model_instance), param_grid,
                    X_train, y_train, X_test, y_test,
                    current_feature_names, False, None
                ))

    print(f"Total evaluation jobs: {len(eval_jobs)}")
    print(f"Running with {N_JOBS_MODELS} parallel workers...")

    # --- RUN PARALLEL EVALUATION ---
    print("\n" + "=" * 60)
    print("STARTING PARALLEL MODEL EVALUATION (SECONDARY ANALYSIS)")
    print("=" * 60 + "\n")

    parallel_start = time.time()
    all_results = Parallel(n_jobs=N_JOBS_MODELS, verbose=10, backend='loky')(
        delayed(evaluate_single_model_task)(job) for job in eval_jobs
    )
    parallel_elapsed = time.time() - parallel_start
    print(f"\n=== Parallel Evaluation Complete in {parallel_elapsed:.1f} seconds ===")

    # --- Evaluate Jaakkimainen Benchmark ---
    print("\n=== Evaluating Jaakkimainen Benchmark ===")
    for label_def, task_specific_data in task_data.items():
        is_dementia_vs_normal = ("Dementia" in label_def and "Normal" in label_def
                                  and "Multi:" not in label_def and "MCI" not in label_def)
        if is_dementia_vs_normal:
            X_train, y_train, X_test, y_test, all_orig_labels, Z_test = task_specific_data
            if Z_test is not None:
                print(f"  Running Jaakkimainen benchmark for: {label_def}")
                try:
                    benchmark_result = evaluate_benchmark_rule(
                        Z_test=Z_test, y_test=y_test,
                        label_definition=label_def,
                        n_bootstraps=N_BOOTSTRAPS,
                        random_state=RANDOM_STATE
                    )
                    all_results.append(benchmark_result)
                    point_auc = benchmark_result["Test_Metrics_Point"].get("auc", np.nan)
                    point_sens = benchmark_result["Test_Metrics_Point"].get("sensitivity", np.nan)
                    point_spec = benchmark_result["Test_Metrics_Point"].get("specificity", np.nan)
                    print(f"    Jaakkimainen AUC: {point_auc:.3f}, Sens: {point_sens:.3f}, Spec: {point_spec:.3f}")
                except Exception as e:
                    print(f"    ERROR evaluating Jaakkimainen benchmark: {e}")

    # --- Process Results ---
    print("\n=== Processing Results ===")
    if not all_results:
        print("No model results generated. Exiting.")
        return

    metrics_df = create_comprehensive_metrics_table(all_results, results_dir)
    generate_auc_charts(all_results, charts_dir)
    generate_comparison_charts(all_results, charts_dir)

    # Save trained models
    print("\n=== Saving Trained Models ===")
    models_dir = os.path.join(base_output_dir, "trained_models")
    os.makedirs(models_dir, exist_ok=True)
    try:
        joblib_dump(all_results, os.path.join(models_dir, "all_results.joblib"))
        joblib_dump(task_data, os.path.join(models_dir, "task_data.joblib"))
        joblib_dump(feature_names_dict_final, os.path.join(models_dir, "feature_names.joblib"))
        print(f"  Saved {len(all_results)} model results to: {models_dir}")
    except Exception as e:
        print(f"  WARNING: Failed to save models: {e}")

    # Publication visualizations
    if VISUALIZATIONS_AVAILABLE:
        pub_viz_dir = os.path.join(base_output_dir, "publication_figures")
        os.makedirs(pub_viz_dir, exist_ok=True)
        try:
            run_all_visualizations(
                results_list=all_results,
                task_data=task_data,
                feature_names_dict=feature_names_dict_final,
                output_base_dir=pub_viz_dir
            )
        except Exception as e:
            print(f"  WARNING: Publication visualizations failed: {e}")

    # Statistical comparisons
    if STATISTICAL_TESTS_AVAILABLE:
        stats_dir = os.path.join(base_output_dir, "statistical_comparisons")
        os.makedirs(stats_dir, exist_ok=True)
        try:
            stat_results = run_statistical_comparisons(
                results_list=all_results,
                task_data=task_data,
                output_dir=stats_dir
            )
            print("\n=== Feature Importance Extraction ===")
            feature_importances = extract_all_feature_importances(all_results, feature_names_dict_final, task_data=task_data)
            for task_name, imp_df in feature_importances.items():
                safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
                imp_path = os.path.join(stats_dir, f"feature_importance_{safe_name}.csv")
                imp_df.to_csv(imp_path, index=False)
                print(f"  Saved feature importance for: {task_name}")
        except Exception as e:
            print(f"  WARNING: Statistical comparisons failed: {e}")

    # --- Summary ---
    total_elapsed = time.time() - analysis_start_time
    print("\n" + "=" * 70)
    print(f"SECONDARY ANALYSIS COMPLETE (Total time: {total_elapsed:.1f}s)")
    print(f"Results saved to: {base_output_dir}")
    print("=" * 70)
    print("\nCompare these results with the primary analysis in:")
    print("  output/primary/")
    print("to assess the contribution of age and sex to model performance.")


if __name__ == "__main__":
    main()
# fmt: on
