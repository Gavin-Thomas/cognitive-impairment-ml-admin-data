# fmt: off
"""
Statistical Tests for Model Comparison in Health Sciences
==========================================================
Implements:
1. DeLong test for comparing AUCs between models
2. McNemar's test for comparing classifier predictions
3. Bonferroni correction for multiple comparisons
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# TASK DATA HELPERS
# ==============================================================================

def _unpack_task_data(task_tuple):
    """
    Unpack task tuple from analysis pipeline.

    Supports both tuple formats:
    - (X_train, y_train, X_test, y_test, labels)
    - (X_train, y_train, X_test, y_test, labels, z_test)
    """
    if not isinstance(task_tuple, (tuple, list)):
        raise ValueError("Task data must be a tuple/list.")

    if len(task_tuple) == 6:
        X_train, y_train, X_test, y_test, labels, z_test = task_tuple
    elif len(task_tuple) == 5:
        X_train, y_train, X_test, y_test, labels = task_tuple
        z_test = None
    else:
        raise ValueError(f"Unexpected task tuple length: {len(task_tuple)}")

    return X_train, y_train, X_test, y_test, labels, z_test


# ==============================================================================
# MULTIPLE COMPARISON CORRECTIONS
# ==============================================================================

def benjamini_hochberg_fdr(p_values, alpha=0.05):
    """
    Apply Benjamini-Hochberg FDR correction.

    The Benjamini-Hochberg procedure controls the False Discovery Rate (FDR),
    which is less conservative than Bonferroni and more appropriate when
    making many comparisons.

    Parameters:
    -----------
    p_values : array-like
        Raw p-values from multiple tests
    alpha : float
        Desired FDR level (default 0.05)

    Returns:
    --------
    dict with:
        - p_adjusted: FDR-adjusted p-values
        - significant: Boolean array of significant results
        - threshold: The BH threshold used
    """
    p_values = np.asarray(p_values)
    n = len(p_values)

    if n == 0:
        return {'p_adjusted': np.array([]), 'significant': np.array([]), 'threshold': 0}

    # Handle NaN values
    valid_mask = ~np.isnan(p_values)
    valid_p = p_values[valid_mask]
    n_valid = len(valid_p)

    if n_valid == 0:
        return {
            'p_adjusted': np.full(n, np.nan),
            'significant': np.full(n, False),
            'threshold': 0
        }

    # Sort p-values and get ranks
    sorted_indices = np.argsort(valid_p)
    sorted_p = valid_p[sorted_indices]
    ranks = np.arange(1, n_valid + 1)

    # Calculate BH critical values: (rank / n) * alpha
    bh_critical = (ranks / n_valid) * alpha

    # Find largest p-value that is <= its critical value
    below_threshold = sorted_p <= bh_critical
    if np.any(below_threshold):
        max_significant_rank = np.max(np.where(below_threshold)[0]) + 1
        threshold = (max_significant_rank / n_valid) * alpha
    else:
        threshold = 0

    # Calculate adjusted p-values (Benjamini-Hochberg method)
    # p_adj[i] = min(p[i] * n / rank[i], 1)
    adjusted_sorted = np.minimum(sorted_p * n_valid / ranks, 1.0)

    # Enforce monotonicity: adjusted p-values should be non-decreasing
    for i in range(n_valid - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Unsort to original order
    adjusted_valid = np.empty(n_valid)
    adjusted_valid[sorted_indices] = adjusted_sorted

    # Map back to full array (with NaNs)
    p_adjusted = np.full(n, np.nan)
    p_adjusted[valid_mask] = adjusted_valid

    # Determine significance
    significant = p_adjusted < alpha

    return {
        'p_adjusted': p_adjusted,
        'significant': significant,
        'threshold': threshold
    }


def apply_multiple_comparison_corrections(df, p_value_col='P_Value', alpha=0.05):
    """
    Apply both Bonferroni and FDR corrections to a DataFrame.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing p-values
    p_value_col : str
        Name of the column containing p-values
    alpha : float
        Significance level

    Returns:
    --------
    DataFrame with added correction columns
    """
    if len(df) == 0 or p_value_col not in df.columns:
        return df

    n_comparisons = len(df)
    p_values = df[p_value_col].values

    # Bonferroni correction (conservative)
    df['P_Value_Bonferroni'] = (df[p_value_col] * n_comparisons).clip(upper=1.0)
    df['Significant_Bonferroni'] = df['P_Value_Bonferroni'] < alpha

    # Benjamini-Hochberg FDR correction (less conservative)
    fdr_result = benjamini_hochberg_fdr(p_values, alpha)
    df['P_Value_FDR'] = fdr_result['p_adjusted']
    df['Significant_FDR'] = fdr_result['significant']

    # Add interpretation column
    def interpret_significance(row):
        if pd.isna(row[p_value_col]):
            return 'N/A'
        elif row['Significant_Bonferroni']:
            return 'Significant (Bonferroni & FDR)'
        elif row['Significant_FDR']:
            return 'Significant (FDR only)'
        elif row[p_value_col] < alpha:
            return 'Nominally significant (uncorrected)'
        else:
            return 'Not significant'

    df['Significance_Interpretation'] = df.apply(interpret_significance, axis=1)

    return df


# ==============================================================================
# DELONG TEST FOR AUC COMPARISON
# ==============================================================================

def compute_midrank(x):
    """Compute midranks for tied values."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    Fast DeLong implementation.

    Parameters:
    -----------
    predictions_sorted_transposed : array
        Predictions sorted by ground truth labels (negatives first, then positives)
    label_1_count : int
        Number of positive samples

    Returns:
    --------
    aucs : array
        AUC values for each model
    delongcov : array
        Covariance matrix for AUC estimates
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)

    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
        tx[r, :] = tz[r, :m]
        ty[r, :] = tz[r, m:]

    aucs = (np.sum(tx, axis=1) / m - (m + 1.0) / 2.0) / n

    v01 = (tz[:, :m].T - tx.T / m).T
    v10 = 1.0 - (tz[:, m:].T - ty.T / n).T

    sx = np.cov(v01)
    sy = np.cov(v10)

    delongcov = sx / m + sy / n

    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Calculate p-value for AUC difference."""
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    pvalue = 2 * (1 - stats.norm.cdf(z))
    return pvalue[0, 0]


def delong_test(y_true, y_pred1, y_pred2):
    """
    DeLong test for comparing two AUCs.

    Parameters:
    -----------
    y_true : array
        Ground truth labels (0/1)
    y_pred1 : array
        Predicted probabilities from model 1
    y_pred2 : array
        Predicted probabilities from model 2

    Returns:
    --------
    dict with:
        - auc1: AUC of model 1
        - auc2: AUC of model 2
        - auc_diff: Difference in AUCs
        - z_stat: Z statistic
        - p_value: Two-sided p-value
        - ci_diff: 95% CI for AUC difference
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    # Check for valid input
    if len(np.unique(y_true)) < 2:
        return {
            'auc1': np.nan, 'auc2': np.nan, 'auc_diff': np.nan,
            'z_stat': np.nan, 'p_value': np.nan, 'ci_diff': (np.nan, np.nan),
            'significant': False
        }

    # Sort by ground truth
    order = np.argsort(y_true)[::-1]  # Positives first
    y_true_sorted = y_true[order]

    # Find number of positives
    label_1_count = int(np.sum(y_true))

    # Stack predictions
    predictions = np.vstack([y_pred1[order], y_pred2[order]])

    try:
        aucs, cov = fastDeLong(predictions, label_1_count)

        # Calculate difference and statistics
        auc_diff = aucs[0] - aucs[1]

        # Variance of difference
        var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        se_diff = np.sqrt(var_diff) if var_diff > 0 else 1e-10

        z_stat = auc_diff / se_diff
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # 95% CI for difference
        ci_lower = auc_diff - 1.96 * se_diff
        ci_upper = auc_diff + 1.96 * se_diff

        return {
            'auc1': aucs[0],
            'auc2': aucs[1],
            'auc_diff': auc_diff,
            'z_stat': z_stat,
            'p_value': p_value,
            'ci_diff': (ci_lower, ci_upper),
            'significant': p_value < 0.05
        }

    except Exception as e:
        return {
            'auc1': np.nan, 'auc2': np.nan, 'auc_diff': np.nan,
            'z_stat': np.nan, 'p_value': np.nan, 'ci_diff': (np.nan, np.nan),
            'significant': False, 'error': str(e)
        }


def delong_test_all_pairs(y_true, model_predictions, model_names):
    """
    Perform DeLong test for all pairs of models.

    Parameters:
    -----------
    y_true : array
        Ground truth labels
    model_predictions : dict
        {model_name: predicted_probabilities}
    model_names : list
        List of model names to compare

    Returns:
    --------
    DataFrame with pairwise comparison results
    """
    results = []
    n_models = len(model_names)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]

            if model1 in model_predictions and model2 in model_predictions:
                test_result = delong_test(
                    y_true,
                    model_predictions[model1],
                    model_predictions[model2]
                )

                results.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'AUC_1': test_result['auc1'],
                    'AUC_2': test_result['auc2'],
                    'AUC_Diff': test_result['auc_diff'],
                    'Z_Statistic': test_result['z_stat'],
                    'P_Value': test_result['p_value'],
                    'CI_Lower': test_result['ci_diff'][0],
                    'CI_Upper': test_result['ci_diff'][1],
                    'Significant_0.05': test_result['significant']
                })

    df = pd.DataFrame(results)

    # Add multiple comparison corrections (Bonferroni and FDR)
    if len(df) > 0 and 'P_Value' in df.columns:
        df = apply_multiple_comparison_corrections(df, 'P_Value')

    return df


# ==============================================================================
# MCNEMAR'S TEST FOR CLASSIFIER COMPARISON
# ==============================================================================

def mcnemar_test(y_true, y_pred1, y_pred2, correction=True):
    """
    McNemar's test for comparing two classifiers.

    Tests whether two classifiers have the same error rate.

    Parameters:
    -----------
    y_true : array
        Ground truth labels
    y_pred1 : array
        Predictions from classifier 1
    y_pred2 : array
        Predictions from classifier 2
    correction : bool
        Whether to apply continuity correction

    Returns:
    --------
    dict with:
        - b: Count where model1 correct, model2 wrong
        - c: Count where model1 wrong, model2 correct
        - chi2_stat: Chi-square statistic
        - p_value: P-value
        - significant: Whether difference is significant at 0.05
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    # Create contingency table
    # b = model1 correct, model2 wrong
    # c = model1 wrong, model2 correct
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct

    # McNemar's test statistic
    if b + c == 0:
        return {
            'b': b, 'c': c,
            'chi2_stat': 0.0,
            'p_value': 1.0,
            'significant': False,
            'better_model': 'equal'
        }

    if correction:
        # With continuity correction
        chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        chi2_stat = (b - c) ** 2 / (b + c)

    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    # Determine which model is better
    if b > c:
        better_model = 'model1'
    elif c > b:
        better_model = 'model2'
    else:
        better_model = 'equal'

    return {
        'b': int(b),
        'c': int(c),
        'chi2_stat': chi2_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'better_model': better_model
    }


def mcnemar_test_all_pairs(y_true, model_predictions, model_names):
    """
    Perform McNemar's test for all pairs of models.

    Parameters:
    -----------
    y_true : array
        Ground truth labels
    model_predictions : dict
        {model_name: predicted_labels}
    model_names : list
        List of model names to compare

    Returns:
    --------
    DataFrame with pairwise comparison results
    """
    results = []
    n_models = len(model_names)

    for i in range(n_models):
        for j in range(i + 1, n_models):
            model1 = model_names[i]
            model2 = model_names[j]

            if model1 in model_predictions and model2 in model_predictions:
                test_result = mcnemar_test(
                    y_true,
                    model_predictions[model1],
                    model_predictions[model2]
                )

                results.append({
                    'Model_1': model1,
                    'Model_2': model2,
                    'b_Model1_Correct_Model2_Wrong': test_result['b'],
                    'c_Model1_Wrong_Model2_Correct': test_result['c'],
                    'Chi2_Statistic': test_result['chi2_stat'],
                    'P_Value': test_result['p_value'],
                    'Significant_0.05': test_result['significant'],
                    'Better_Model': test_result['better_model']
                })

    df = pd.DataFrame(results)

    # Add multiple comparison corrections (Bonferroni and FDR)
    if len(df) > 0 and 'P_Value' in df.columns:
        df = apply_multiple_comparison_corrections(df, 'P_Value')

    return df


# ==============================================================================
# FEATURE IMPORTANCE EXTRACTION
# ==============================================================================

def extract_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from a trained model.

    Parameters:
    -----------
    model : sklearn model or pipeline
        Trained model
    feature_names : list
        List of feature names
    model_name : str
        Name of the model

    Returns:
    --------
    DataFrame with feature importances or None if not available
    """
    importance_values = None
    importance_type = None

    # Try to get the final estimator if it's a pipeline
    if hasattr(model, 'named_steps'):
        final_step_name = list(model.named_steps.keys())[-1]
        estimator = model.named_steps[final_step_name]
    else:
        estimator = model

    # Try different importance attributes
    if hasattr(estimator, 'feature_importances_'):
        # Tree-based models (RF, GBM, XGBoost)
        importance_values = estimator.feature_importances_
        importance_type = 'Gini/Gain Importance'
    elif hasattr(estimator, 'coef_'):
        # Linear models (Logistic Regression, SVM with linear kernel)
        coef = estimator.coef_
        if coef.ndim > 1:
            # For multi-class, take mean absolute across classes
            importance_values = np.mean(np.abs(coef), axis=0)
        else:
            importance_values = np.abs(coef).flatten()
        importance_type = 'Coefficient Magnitude'

    if importance_values is None:
        return None

    # Ensure feature_names matches importance_values length
    if len(feature_names) != len(importance_values):
        feature_names = [f"Feature_{i+1}" for i in range(len(importance_values))]

    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values,
        'Importance_Type': importance_type,
        'Model': model_name
    })

    # Sort by importance and add rank
    df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
    df['Rank'] = range(1, len(df) + 1)

    # Add normalized importance
    total_importance = df['Importance'].sum()
    if total_importance > 0:
        df['Importance_Normalized'] = df['Importance'] / total_importance
    else:
        df['Importance_Normalized'] = 0

    return df[['Rank', 'Feature', 'Importance', 'Importance_Normalized', 'Importance_Type', 'Model']]


def extract_permutation_importance(model, X_test, y_test, feature_names, model_name,
                                    n_repeats=10, random_state=42, scoring='balanced_accuracy'):
    """
    Compute permutation importance on the test set.

    Unlike model-native importance (Gini/Gain), permutation importance measures
    the decrease in model performance when each feature is randomly shuffled,
    making it model-agnostic and unbiased toward high-cardinality features.

    Parameters:
    -----------
    model : sklearn model or pipeline
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    n_repeats : int
        Number of times to permute each feature (default: 10)
    random_state : int
        Random seed for reproducibility
    scoring : str
        Scoring metric to evaluate importance (default: 'balanced_accuracy')

    Returns:
    --------
    DataFrame with permutation importances or None if computation fails
    """
    try:
        from sklearn.inspection import permutation_importance as sklearn_perm_importance
    except ImportError:
        print(f"    WARNING: sklearn.inspection.permutation_importance not available")
        return None

    try:
        perm_result = sklearn_perm_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring=scoring,
            n_jobs=1
        )

        importance_values = perm_result.importances_mean
        importance_std = perm_result.importances_std

        if len(feature_names) != len(importance_values):
            feature_names = [f"Feature_{i+1}" for i in range(len(importance_values))]

        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values,
            'Importance_Std': importance_std,
            'Importance_Type': 'Permutation Importance',
            'Model': model_name
        })

        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)

        total_importance = df['Importance'].clip(lower=0).sum()
        if total_importance > 0:
            df['Importance_Normalized'] = df['Importance'].clip(lower=0) / total_importance
        else:
            df['Importance_Normalized'] = 0

        return df[['Rank', 'Feature', 'Importance', 'Importance_Std', 'Importance_Normalized', 'Importance_Type', 'Model']]

    except Exception as e:
        print(f"    WARNING: Permutation importance failed for {model_name}: {e}")
        return None


def extract_all_feature_importances(results_list, feature_names_dict, task_data=None):
    """
    Extract feature importances from all models.

    Computes both model-native importance (Gini/Gain/Coefficients) and,
    if task_data is provided, permutation importance on the test set.

    Parameters:
    -----------
    results_list : list
        List of model results
    feature_names_dict : dict
        Dictionary mapping task names to feature names
    task_data : dict, optional
        Dictionary with task data (needed for permutation importance).
        If None, only model-native importance is computed.

    Returns:
    --------
    dict: {task_name: DataFrame with all models' importances}
    """
    all_importances = {}

    for result in results_list:
        task_name = result.get("LabelDefinition", "Unknown")
        model_name = result.get("ModelName", "Unknown")
        model = result.get("FinalModel")

        if model is None:
            continue

        feature_names = feature_names_dict.get(task_name, [])
        if not feature_names:
            continue

        # Model-native importance (Gini/Gain/Coefficients)
        importance_df = extract_feature_importance(model, feature_names, model_name)

        if importance_df is not None:
            if task_name not in all_importances:
                all_importances[task_name] = []
            all_importances[task_name].append(importance_df)

        # Permutation importance (if task_data provided)
        if task_data is not None and task_name in task_data:
            td = task_data[task_name]
            if len(td) >= 4:
                X_test = td[2]
                y_test = td[3]
                perm_df = extract_permutation_importance(
                    model, X_test, y_test, feature_names, model_name
                )
                if perm_df is not None:
                    all_importances[task_name].append(perm_df)

    # Combine importances per task
    combined = {}
    for task_name, dfs in all_importances.items():
        if dfs:
            combined[task_name] = pd.concat(dfs, ignore_index=True)

    return combined


# ==============================================================================
# PER-SAMPLE PREDICTIONS
# ==============================================================================

def generate_per_sample_predictions(results_list, task_data):
    """
    Generate per-sample predictions for all models.

    Parameters:
    -----------
    results_list : list
        List of model results
    task_data : dict
        Dictionary with task data

    Returns:
    --------
    dict: {task_name: DataFrame with all predictions}
    """
    all_predictions = {}

    # Group results by task
    task_results = {}
    for result in results_list:
        task_name = result.get("LabelDefinition", "Unknown")
        if task_name not in task_results:
            task_results[task_name] = []
        task_results[task_name].append(result)

    for task_name, results in task_results.items():
        if task_name not in task_data:
            continue

        X_train, y_train, X_test, y_test, _, z_test = _unpack_task_data(task_data[task_name])
        is_multi = "Multi:" in task_name

        # Start with ground truth
        pred_df = pd.DataFrame({
            'Sample_Index': range(len(y_test)),
            'True_Label': y_test
        })

        for result in results:
            model_name = result.get("ModelName", "Unknown")
            model = result.get("FinalModel")
            threshold = result.get("CV_Threshold", 0.5)

            # FIX: Handle NaN thresholds (rule-based classifiers store NaN).
            if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
                threshold = 0.5

            if model is None:
                continue

            try:
                # FIX: Jaakkimainen benchmark must use Z_test (3 indicator
                # columns), not X_test (full ML feature matrix).
                if "jaakkimainen" in model_name.lower():
                    if z_test is not None:
                        y_prob_jaak = model.predict_proba(z_test)
                        pred_df[f'{model_name}_Prob'] = y_prob_jaak[:, 1]
                        pred_df[f'{model_name}_Pred'] = model.predict(z_test)
                    else:
                        print(f"    WARNING: No Z_test for {model_name}, skipping")
                    continue

                # Get probability predictions
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)

                    if is_multi:
                        # For multiclass tasks, keep class prediction and confidence.
                        pred_df[f'{model_name}_Pred'] = np.argmax(y_prob, axis=1)
                        pred_df[f'{model_name}_Confidence'] = np.max(y_prob, axis=1)
                    elif y_prob.shape[1] >= 2:
                        pred_df[f'{model_name}_Prob'] = y_prob[:, 1]
                        pred_df[f'{model_name}_Pred'] = (y_prob[:, 1] >= threshold).astype(int)
                    else:
                        pred_df[f'{model_name}_Pred'] = model.predict(X_test)
                else:
                    pred_df[f'{model_name}_Pred'] = model.predict(X_test)

            except Exception as e:
                print(f"    Error getting predictions for {model_name}: {e}")
                continue

        # Add correct/incorrect flags
        for result in results:
            model_name = result.get("ModelName", "Unknown")
            pred_col = f'{model_name}_Pred'
            if pred_col in pred_df.columns:
                pred_df[f'{model_name}_Correct'] = (pred_df[pred_col] == pred_df['True_Label']).astype(int)

        all_predictions[task_name] = pred_df

    return all_predictions


# ==============================================================================
# CLASS IMBALANCE REPORTING
# ==============================================================================

def calculate_class_imbalance(y_train, y_test, class_names=None):
    """
    Calculate class imbalance metrics.

    Parameters:
    -----------
    y_train : array
        Training labels
    y_test : array
        Test labels
    class_names : list, optional
        Names for each class

    Returns:
    --------
    dict with imbalance statistics
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # Get unique classes
    classes = np.unique(np.concatenate([y_train, y_test]))
    n_classes = len(classes)

    if class_names is None:
        class_names = [f"Class_{c}" for c in classes]

    # Training set statistics
    train_counts = {c: np.sum(y_train == c) for c in classes}
    train_total = len(y_train)
    train_proportions = {c: count / train_total for c, count in train_counts.items()}

    # Test set statistics
    test_counts = {c: np.sum(y_test == c) for c in classes}
    test_total = len(y_test)
    test_proportions = {c: count / test_total for c, count in test_counts.items()}

    # Calculate imbalance ratio (majority / minority)
    train_max = max(train_counts.values())
    train_min = min(train_counts.values())
    imbalance_ratio_train = train_max / train_min if train_min > 0 else np.inf

    test_max = max(test_counts.values())
    test_min = min(test_counts.values())
    imbalance_ratio_test = test_max / test_min if test_min > 0 else np.inf

    # Create summary
    summary = {
        'n_classes': n_classes,
        'class_names': class_names,
        'train_total': train_total,
        'test_total': test_total,
        'train_counts': train_counts,
        'test_counts': test_counts,
        'train_proportions': train_proportions,
        'test_proportions': test_proportions,
        'imbalance_ratio_train': imbalance_ratio_train,
        'imbalance_ratio_test': imbalance_ratio_test
    }

    return summary


def generate_imbalance_report(task_data, output_path=None):
    """
    Generate a comprehensive class imbalance report.

    Parameters:
    -----------
    task_data : dict
        Dictionary with task data
    output_path : str, optional
        Path to save CSV report

    Returns:
    --------
    DataFrame with imbalance report
    """
    rows = []

    for task_name, data in task_data.items():
        X_train, y_train, X_test, y_test, labels, _ = _unpack_task_data(data)

        # Determine class names
        if "Multi:" in task_name or "vs. MCI vs." in task_name:
            class_names = ['Normal', 'MCI', 'Dementia']
        else:
            class_names = ['Normal/Negative', 'Impaired/Positive']

        imbalance = calculate_class_imbalance(y_train, y_test, class_names)

        # Create row for each class
        for i, (cls, name) in enumerate(zip(sorted(imbalance['train_counts'].keys()), class_names)):
            rows.append({
                'Task': task_name,
                'Class_Label': cls,
                'Class_Name': name,
                'Train_Count': imbalance['train_counts'][cls],
                'Train_Proportion': imbalance['train_proportions'][cls],
                'Test_Count': imbalance['test_counts'][cls],
                'Test_Proportion': imbalance['test_proportions'][cls],
                'Imbalance_Ratio_Train': imbalance['imbalance_ratio_train'],
                'Imbalance_Ratio_Test': imbalance['imbalance_ratio_test']
            })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Class imbalance report saved to: {output_path}")

    return df


# ==============================================================================
# MAIN STATISTICAL ANALYSIS RUNNER
# ==============================================================================

def run_statistical_comparisons(results_list, task_data, output_dir):
    """
    Run all statistical comparisons and save results.

    Parameters:
    -----------
    results_list : list
        List of model results
    task_data : dict
        Dictionary with task data
    output_dir : str
        Directory to save results

    Returns:
    --------
    dict with all comparison results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("RUNNING STATISTICAL COMPARISONS")
    print("=" * 70)

    all_results = {
        'delong': {},
        'mcnemar': {},
        'feature_importance': {},
        'per_sample': {},
        'imbalance': None
    }

    # Group results by task
    task_results_map = {}
    for result in results_list:
        task_name = result.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(result)

    # 1. DeLong tests for AUC comparison
    print("\n=== DeLong Tests (AUC Comparison) ===")
    for task_name, results in task_results_map.items():
        if task_name not in task_data:
            continue

        is_multi = "Multi:" in task_name
        if is_multi:
            continue  # Skip multi-class for DeLong

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, _, z_test = _unpack_task_data(task_data[task_name])

        # Collect predictions
        model_probs = {}
        model_names = []

        for result in results:
            model_name = result.get("ModelName", "Unknown")
            model = result.get("FinalModel")

            if model is not None and hasattr(model, "predict_proba"):
                try:
                    # FIX: Jaakkimainen benchmark must use Z_test (3 indicator
                    # columns), not X_test (full ML feature matrix). Its predict()
                    # does np.any(X > 0, axis=1) which fires incorrectly on 28 cols.
                    # Also: DeLong is unreliable for binary (0/1) predictions
                    # because the mid-rank computation becomes degenerate. We skip
                    # Jaakkimainen in DeLong and note this in the output.
                    if "jaakkimainen" in model_name.lower():
                        # DeLong is not valid for deterministic (0/1)
                        # classifiers — the mid-rank computation degenerates,
                        # producing unreliable variance estimates. Skip entirely.
                        print(f"    SKIP: {model_name} produces binary predictions; "
                              f"excluding from DeLong (McNemar used instead).")
                        continue
                    else:
                        y_prob = model.predict_proba(X_test)[:, 1]
                    model_probs[model_name] = y_prob
                    model_names.append(model_name)
                except:
                    continue

        if len(model_names) >= 2:
            delong_df = delong_test_all_pairs(y_test, model_probs, model_names)
            all_results['delong'][task_name] = delong_df

            # Save
            safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
            delong_df.to_csv(os.path.join(output_dir, f"delong_test_{safe_name}.csv"), index=False)
            print(f"    Saved DeLong test results ({len(delong_df)} comparisons)")

    # 2. McNemar's tests for classifier comparison
    print("\n=== McNemar's Tests (Classifier Comparison) ===")
    for task_name, results in task_results_map.items():
        if task_name not in task_data:
            continue

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, _, z_test = _unpack_task_data(task_data[task_name])
        is_multi = "Multi:" in task_name

        # Collect predictions
        model_preds = {}
        model_names = []

        for result in results:
            model_name = result.get("ModelName", "Unknown")
            model = result.get("FinalModel")
            threshold = result.get("CV_Threshold", 0.5)

            # FIX: Handle NaN thresholds (e.g., rule-based classifiers).
            # np.nan >= any_value returns False, so all predictions become 0.
            if threshold is None or (isinstance(threshold, float) and np.isnan(threshold)):
                threshold = 0.5

            if model is not None:
                try:
                    # FIX: Jaakkimainen benchmark must use Z_test, not X_test.
                    # Its predict() does np.any(X > 0, axis=1) which fires on
                    # all 28 ML features, producing wrong predictions.
                    if "jaakkimainen" in model_name.lower():
                        if z_test is not None:
                            y_pred = model.predict(z_test)
                        else:
                            print(f"    WARNING: No Z_test for {model_name}, skipping McNemar")
                            continue
                    elif hasattr(model, "predict_proba") and not is_multi:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        y_pred = (y_prob >= threshold).astype(int)
                    else:
                        y_pred = model.predict(X_test)
                    model_preds[model_name] = y_pred
                    model_names.append(model_name)
                except:
                    continue

        if len(model_names) >= 2:
            mcnemar_df = mcnemar_test_all_pairs(y_test, model_preds, model_names)
            all_results['mcnemar'][task_name] = mcnemar_df

            # Save
            safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
            mcnemar_df.to_csv(os.path.join(output_dir, f"mcnemar_test_{safe_name}.csv"), index=False)
            print(f"    Saved McNemar's test results ({len(mcnemar_df)} comparisons)")

    # 3. Class imbalance report
    print("\n=== Class Imbalance Report ===")
    imbalance_df = generate_imbalance_report(
        task_data,
        os.path.join(output_dir, "class_imbalance_report.csv")
    )
    all_results['imbalance'] = imbalance_df

    # 4. Per-sample predictions
    print("\n=== Per-Sample Predictions ===")
    per_sample = generate_per_sample_predictions(results_list, task_data)
    for task_name, pred_df in per_sample.items():
        safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
        pred_df.to_csv(os.path.join(output_dir, f"predictions_{safe_name}.csv"), index=False)
        print(f"  Saved predictions for: {task_name} ({len(pred_df)} samples)")
    all_results['per_sample'] = per_sample

    print("\n" + "=" * 70)
    print("STATISTICAL COMPARISONS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    print("Statistical tests module for model comparison.")
    print("\nAvailable functions:")
    print("  - delong_test(y_true, y_pred1, y_pred2)")
    print("  - mcnemar_test(y_true, y_pred1, y_pred2)")
    print("  - extract_feature_importance(model, feature_names, model_name)")
    print("  - run_statistical_comparisons(results_list, task_data, output_dir)")
# fmt: on
