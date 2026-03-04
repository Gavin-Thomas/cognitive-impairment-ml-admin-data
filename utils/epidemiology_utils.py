# fmt: off
"""
Epidemiology Utilities for Clinical Prediction Model Validation
================================================================
Implements epidemiological best practices for health sciences ML:

1. Missing Data Analysis (Little's MCAR test, missingness patterns)
2. Decision Curve Analysis (clinical utility assessment)
3. Calibration Metrics (slope, intercept, Hosmer-Lemeshow)
4. Sample Size Justification (events-per-variable)
5. BCa Bootstrap Confidence Intervals
6. Subgroup Analysis
7. TRIPOD Compliance Checklist

References:
- Collins GS et al. TRIPOD Statement (2015) Ann Intern Med
- Vickers AJ et al. Decision Curve Analysis (2006) Med Decis Making
- Steyerberg EW. Clinical Prediction Models (2019) Springer
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

    Supports:
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
# 1. MISSING DATA ANALYSIS
# ==============================================================================

def littles_mcar_test(data):
    """
    Little's MCAR (Missing Completely at Random) Test.

    Tests the null hypothesis that data is MCAR vs MAR/MNAR.
    A significant p-value suggests data is NOT MCAR.

    Parameters:
    -----------
    data : DataFrame
        Data with potential missing values

    Returns:
    --------
    dict with:
        - chi2_stat: Chi-square statistic
        - df: Degrees of freedom
        - p_value: P-value (significant = NOT MCAR)
        - n_patterns: Number of missing patterns
        - interpretation: Text interpretation
    """
    df = data.copy()

    # Only consider numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        return {
            'chi2_stat': np.nan,
            'df': np.nan,
            'p_value': np.nan,
            'n_patterns': 0,
            'interpretation': 'No numeric columns to analyze'
        }

    df_numeric = df[numeric_cols]

    # Check if there's any missing data
    if not df_numeric.isnull().any().any():
        return {
            'chi2_stat': 0.0,
            'df': 0,
            'p_value': 1.0,
            'n_patterns': 1,
            'interpretation': 'No missing data detected'
        }

    # Create missing data patterns
    missing_pattern = df_numeric.isnull().astype(int)
    pattern_strings = missing_pattern.apply(lambda x: ''.join(x.astype(str)), axis=1)
    unique_patterns = pattern_strings.unique()
    n_patterns = len(unique_patterns)

    if n_patterns <= 1:
        return {
            'chi2_stat': 0.0,
            'df': 0,
            'p_value': 1.0,
            'n_patterns': n_patterns,
            'interpretation': 'Only one missing pattern - test not applicable'
        }

    # Calculate overall means and covariance (complete cases)
    complete_data = df_numeric.dropna()
    if len(complete_data) < 2:
        return {
            'chi2_stat': np.nan,
            'df': np.nan,
            'p_value': np.nan,
            'n_patterns': n_patterns,
            'interpretation': 'Insufficient complete cases for test'
        }

    try:
        overall_mean = complete_data.mean()
        overall_cov = complete_data.cov()

        # Calculate chi-square statistic
        chi2_stat = 0.0
        df_total = 0

        for pattern in unique_patterns:
            pattern_mask = pattern_strings == pattern
            pattern_data = df_numeric[pattern_mask]

            # Get observed (non-missing) variables for this pattern
            observed_vars = [col for col in numeric_cols
                           if not pattern_data[col].isnull().all()]

            if len(observed_vars) == 0:
                continue

            pattern_complete = pattern_data[observed_vars].dropna()
            n_pattern = len(pattern_complete)

            if n_pattern < 2:
                continue

            # Pattern mean
            pattern_mean = pattern_complete.mean()

            # Subset of overall covariance matrix
            cov_subset = overall_cov.loc[observed_vars, observed_vars]

            # Check if covariance matrix is invertible
            try:
                cov_inv = np.linalg.inv(cov_subset.values)
            except np.linalg.LinAlgError:
                continue

            # Mean difference
            mean_diff = (pattern_mean - overall_mean[observed_vars]).values

            # Contribution to chi-square
            chi2_contrib = n_pattern * np.dot(np.dot(mean_diff.T, cov_inv), mean_diff)
            chi2_stat += chi2_contrib
            df_total += len(observed_vars)

        # Adjust degrees of freedom
        df_adjusted = max(1, df_total - len(numeric_cols))

        # Calculate p-value
        p_value = 1 - chi2.cdf(chi2_stat, df_adjusted)

        # Interpretation
        if p_value < 0.05:
            interpretation = f"SIGNIFICANT (p={p_value:.4f}): Data is likely NOT MCAR. Consider MAR/MNAR mechanisms."
        else:
            interpretation = f"Not significant (p={p_value:.4f}): Data is consistent with MCAR assumption."

        return {
            'chi2_stat': chi2_stat,
            'df': df_adjusted,
            'p_value': p_value,
            'n_patterns': n_patterns,
            'interpretation': interpretation
        }

    except Exception as e:
        return {
            'chi2_stat': np.nan,
            'df': np.nan,
            'p_value': np.nan,
            'n_patterns': n_patterns,
            'interpretation': f'Error in calculation: {str(e)}'
        }


def analyze_missing_patterns(data, output_path=None):
    """
    Comprehensive missing data pattern analysis.

    Parameters:
    -----------
    data : DataFrame
        Data to analyze
    output_path : str, optional
        Path to save report

    Returns:
    --------
    dict with missingness analysis results
    """
    df = data.copy()
    n_total = len(df)

    results = {
        'summary': {},
        'by_variable': [],
        'patterns': [],
        'mcar_test': None
    }

    # Overall summary
    n_complete = df.dropna().shape[0]
    n_with_missing = n_total - n_complete

    results['summary'] = {
        'n_total': n_total,
        'n_complete_cases': n_complete,
        'n_with_any_missing': n_with_missing,
        'pct_complete': 100 * n_complete / n_total if n_total > 0 else 0
    }

    # By variable
    for col in df.columns:
        n_missing = df[col].isnull().sum()
        pct_missing = 100 * n_missing / n_total if n_total > 0 else 0

        results['by_variable'].append({
            'Variable': col,
            'N_Missing': n_missing,
            'Pct_Missing': pct_missing,
            'N_Complete': n_total - n_missing
        })

    # Convert to DataFrame for easy viewing
    var_df = pd.DataFrame(results['by_variable'])
    var_df = var_df.sort_values('Pct_Missing', ascending=False)

    # Missing patterns (top 10)
    missing_pattern = df.isnull()
    pattern_counts = missing_pattern.groupby(list(missing_pattern.columns)).size()
    pattern_counts = pattern_counts.sort_values(ascending=False).head(10)

    for pattern, count in pattern_counts.items():
        missing_vars = [col for col, is_missing in zip(df.columns, pattern) if is_missing]
        results['patterns'].append({
            'Pattern': '|'.join(missing_vars) if missing_vars else 'Complete',
            'N': count,
            'Pct': 100 * count / n_total
        })

    # Little's MCAR test
    results['mcar_test'] = littles_mcar_test(df)

    # Save if path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("MISSING DATA ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            for key, val in results['summary'].items():
                f.write(f"  {key}: {val}\n")

            f.write("\n\nLITTLE'S MCAR TEST\n")
            f.write("-" * 40 + "\n")
            mcar = results['mcar_test']
            f.write(f"  Chi-square: {mcar['chi2_stat']:.4f}\n")
            f.write(f"  df: {mcar['df']}\n")
            f.write(f"  p-value: {mcar['p_value']:.4f}\n")
            f.write(f"  Interpretation: {mcar['interpretation']}\n")

            f.write("\n\nMISSINGNESS BY VARIABLE\n")
            f.write("-" * 40 + "\n")
            f.write(var_df.to_string(index=False))

            f.write("\n\n\nTOP MISSING PATTERNS\n")
            f.write("-" * 40 + "\n")
            pattern_df = pd.DataFrame(results['patterns'])
            f.write(pattern_df.to_string(index=False))

        print(f"  Missing data report saved to: {output_path}")

    return results


# ==============================================================================
# 2. DECISION CURVE ANALYSIS
# ==============================================================================

def calculate_net_benefit(y_true, y_prob, threshold):
    """
    Calculate net benefit at a given threshold.

    Net Benefit = (TP/N) - (FP/N) * (threshold / (1 - threshold))

    Parameters:
    -----------
    y_true : array
        True labels (0/1)
    y_prob : array
        Predicted probabilities
    threshold : float
        Decision threshold (0-1)

    Returns:
    --------
    float: Net benefit
    """
    if threshold <= 0 or threshold >= 1:
        return np.nan

    y_pred = (y_prob >= threshold).astype(int)
    n = len(y_true)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    # Net benefit formula
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))

    return net_benefit


def decision_curve_analysis(y_true, model_predictions, model_names,
                            thresholds=None, output_path=None):
    """
    Perform Decision Curve Analysis for clinical utility assessment.

    Parameters:
    -----------
    y_true : array
        True labels
    model_predictions : dict
        {model_name: predicted_probabilities}
    model_names : list
        List of model names
    thresholds : array, optional
        Threshold values to evaluate (default: 0.01 to 0.99)
    output_path : str, optional
        Path to save results

    Returns:
    --------
    DataFrame with net benefit at each threshold for each model
    """
    y_true = np.asarray(y_true)

    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    results = []
    prevalence = np.mean(y_true)

    for thresh in thresholds:
        row = {'Threshold': thresh}

        # Treat All strategy
        row['Treat_All'] = prevalence - (1 - prevalence) * (thresh / (1 - thresh))

        # Treat None strategy
        row['Treat_None'] = 0.0

        # Each model
        for model_name in model_names:
            if model_name in model_predictions:
                y_prob = np.asarray(model_predictions[model_name])
                nb = calculate_net_benefit(y_true, y_prob, thresh)
                row[model_name] = nb

        results.append(row)

    df = pd.DataFrame(results)

    # Calculate summary metrics
    summary = []
    for model_name in model_names:
        if model_name in df.columns:
            # Find threshold range where model beats treat-all and treat-none
            model_nb = df[model_name].values
            treat_all_nb = df['Treat_All'].values
            treat_none_nb = df['Treat_None'].values

            # Net benefit improvement over treat-all
            improvement = model_nb - np.maximum(treat_all_nb, treat_none_nb)

            # Integrated net benefit (area under net benefit curve)
            valid_mask = ~np.isnan(model_nb)
            if np.sum(valid_mask) > 1:
                integrated_nb = np.trapz(model_nb[valid_mask], thresholds[valid_mask])
            else:
                integrated_nb = np.nan

            summary.append({
                'Model': model_name,
                'Max_Net_Benefit': np.nanmax(model_nb),
                'Threshold_at_Max_NB': thresholds[np.nanargmax(model_nb)] if not np.all(np.isnan(model_nb)) else np.nan,
                'Integrated_Net_Benefit': integrated_nb,
                'Mean_Improvement_Over_Default': np.nanmean(improvement)
            })

    summary_df = pd.DataFrame(summary)

    if output_path:
        df.to_csv(output_path, index=False)
        summary_df.to_csv(output_path.replace('.csv', '_summary.csv'), index=False)
        print(f"  Decision curve data saved to: {output_path}")

    return df, summary_df


# ==============================================================================
# 3. CALIBRATION METRICS (SLOPE, INTERCEPT, HOSMER-LEMESHOW)
# ==============================================================================

def calculate_calibration_metrics(y_true, y_prob, n_bins=10):
    """
    Calculate comprehensive calibration metrics.

    Parameters:
    -----------
    y_true : array
        True labels (0/1)
    y_prob : array
        Predicted probabilities
    n_bins : int
        Number of bins for calibration assessment

    Returns:
    --------
    dict with calibration metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_prob = np.asarray(y_prob).flatten()

    # Basic checks
    if len(y_true) != len(y_prob):
        return {'error': 'Length mismatch between y_true and y_prob'}

    if len(np.unique(y_true)) < 2:
        return {'error': 'Need both classes in y_true'}

    results = {}

    # 1. Brier Score
    results['brier_score'] = np.mean((y_prob - y_true) ** 2)

    # 2. Calibration Slope and Intercept (logistic calibration)
    # Fit logistic regression of y_true on logit(y_prob)
    try:
        # Clip probabilities to avoid log(0)
        y_prob_clipped = np.clip(y_prob, 1e-10, 1 - 1e-10)
        logit_prob = np.log(y_prob_clipped / (1 - y_prob_clipped))

        # Simple logistic regression using Newton-Raphson
        # For large samples, use statsmodels if available
        try:
            import statsmodels.api as sm
            X = sm.add_constant(logit_prob)
            model = sm.Logit(y_true, X).fit(disp=0)
            results['calibration_intercept'] = model.params[0]
            results['calibration_slope'] = model.params[1]
            results['calibration_intercept_se'] = model.bse[0]
            results['calibration_slope_se'] = model.bse[1]
        except ImportError:
            # Fallback: simple linear regression on observed vs expected
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(solver='lbfgs', max_iter=1000)
            lr.fit(logit_prob.reshape(-1, 1), y_true)
            results['calibration_intercept'] = lr.intercept_[0]
            results['calibration_slope'] = lr.coef_[0][0]
            results['calibration_intercept_se'] = np.nan
            results['calibration_slope_se'] = np.nan
    except Exception as e:
        results['calibration_intercept'] = np.nan
        results['calibration_slope'] = np.nan
        results['calibration_intercept_se'] = np.nan
        results['calibration_slope_se'] = np.nan

    # 3. Expected Calibration Error (ECE)
    # 4. Maximum Calibration Error (MCE)
    try:
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])

        ece = 0.0
        mce = 0.0
        calibration_data = []

        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_prob = np.mean(y_prob[mask])
                bin_true = np.mean(y_true[mask])
                bin_size = np.sum(mask)

                error = abs(bin_prob - bin_true)
                ece += (bin_size / len(y_true)) * error
                mce = max(mce, error)

                calibration_data.append({
                    'bin': i + 1,
                    'mean_predicted': bin_prob,
                    'fraction_positive': bin_true,
                    'n_samples': bin_size,
                    'calibration_error': error
                })

        results['ece'] = ece
        results['mce'] = mce
        results['calibration_data'] = calibration_data

    except Exception as e:
        results['ece'] = np.nan
        results['mce'] = np.nan

    # 5. Hosmer-Lemeshow Test
    try:
        # Group into deciles
        decile_edges = np.percentile(y_prob, np.arange(0, 101, 10))
        decile_indices = np.digitize(y_prob, decile_edges[1:-1])

        observed = []
        expected = []
        n_groups = []

        for i in range(10):
            mask = decile_indices == i
            if np.sum(mask) > 0:
                observed.append(np.sum(y_true[mask]))
                expected.append(np.sum(y_prob[mask]))
                n_groups.append(np.sum(mask))

        observed = np.array(observed)
        expected = np.array(expected)
        n_groups = np.array(n_groups)

        # Chi-square statistic
        # H-L = sum((O - E)^2 / (E * (1 - E/n)))
        with np.errstate(divide='ignore', invalid='ignore'):
            expected_neg = n_groups - expected
            hl_stat = np.sum(
                (observed - expected) ** 2 / expected +
                ((n_groups - observed) - expected_neg) ** 2 / expected_neg
            )

        # Degrees of freedom = g - 2
        hl_df = len(n_groups) - 2
        hl_pvalue = 1 - chi2.cdf(hl_stat, hl_df) if hl_df > 0 else np.nan

        results['hosmer_lemeshow_chi2'] = hl_stat
        results['hosmer_lemeshow_df'] = hl_df
        results['hosmer_lemeshow_pvalue'] = hl_pvalue

        # Interpretation
        if hl_pvalue < 0.05:
            results['hosmer_lemeshow_interpretation'] = 'Poor calibration (p < 0.05)'
        else:
            results['hosmer_lemeshow_interpretation'] = 'Adequate calibration (p >= 0.05)'

    except Exception as e:
        results['hosmer_lemeshow_chi2'] = np.nan
        results['hosmer_lemeshow_df'] = np.nan
        results['hosmer_lemeshow_pvalue'] = np.nan

    # Interpretation of calibration slope
    if not np.isnan(results.get('calibration_slope', np.nan)):
        slope = results['calibration_slope']
        if slope < 0.8:
            results['slope_interpretation'] = f'Overfitting suspected (slope={slope:.3f} < 0.8)'
        elif slope > 1.2:
            results['slope_interpretation'] = f'Underfitting suspected (slope={slope:.3f} > 1.2)'
        else:
            results['slope_interpretation'] = f'Good calibration (slope={slope:.3f})'

    return results


# ==============================================================================
# 4. SAMPLE SIZE JUSTIFICATION (EVENTS PER VARIABLE)
# ==============================================================================

def calculate_epv_metrics(n_events, n_predictors, n_total=None):
    """
    Calculate Events Per Variable (EPV) and related sample size metrics.

    Guidelines:
    - EPV >= 10: Traditional rule (Peduzzi et al., 1996)
    - EPV >= 20: More conservative (Vittinghoff & McCulloch, 2007)
    - Modern: Use Riley et al. (2019) criteria

    Parameters:
    -----------
    n_events : int
        Number of events (positive cases)
    n_predictors : int
        Number of predictor variables
    n_total : int, optional
        Total sample size

    Returns:
    --------
    dict with EPV metrics and recommendations
    """
    results = {
        'n_events': n_events,
        'n_predictors': n_predictors,
        'n_total': n_total
    }

    # Events per variable
    epv = n_events / n_predictors if n_predictors > 0 else np.nan
    results['epv'] = epv

    # EPV interpretation
    if epv >= 20:
        results['epv_interpretation'] = 'Adequate (EPV >= 20, conservative criterion)'
        results['epv_adequate'] = True
    elif epv >= 10:
        results['epv_interpretation'] = 'Marginal (10 <= EPV < 20, traditional criterion)'
        results['epv_adequate'] = True
    elif epv >= 5:
        results['epv_interpretation'] = 'Low (5 <= EPV < 10, risk of overfitting)'
        results['epv_adequate'] = False
    else:
        results['epv_interpretation'] = 'Very low (EPV < 5, high risk of overfitting)'
        results['epv_adequate'] = False

    # Minimum events needed
    results['min_events_epv10'] = 10 * n_predictors
    results['min_events_epv20'] = 20 * n_predictors

    # Non-events per variable (for specificity)
    if n_total is not None:
        n_nonevents = n_total - n_events
        results['n_nonevents'] = n_nonevents
        results['nonevents_per_variable'] = n_nonevents / n_predictors if n_predictors > 0 else np.nan
        results['prevalence'] = n_events / n_total

        # Check if non-events are also sufficient
        npv = n_nonevents / n_predictors if n_predictors > 0 else np.nan
        if npv < 10:
            results['nonevents_warning'] = f'Low non-events per variable ({npv:.1f}), specificity estimates may be unstable'

    # Riley et al. (2019) criteria
    # Minimum sample size = (C / S)^2 * 1.96^2 / delta^2
    # where C is shrinkage factor target, S is expected C-statistic
    # Simplified approximation
    results['riley_recommendation'] = (
        f"Consider Riley et al. (2019) sample size calculation for "
        f"prediction models. With {n_predictors} predictors, recommend "
        f"minimum {max(200, 50 * n_predictors)} total samples for stable estimates."
    )

    return results


def generate_sample_size_report(task_data, feature_counts, output_path=None):
    """
    Generate sample size justification report for all tasks.

    Parameters:
    -----------
    task_data : dict
        Dictionary with task data
    feature_counts : dict
        {task_name: n_features}
    output_path : str, optional
        Path to save report

    Returns:
    --------
    DataFrame with sample size metrics
    """
    rows = []

    for task_name, data in task_data.items():
        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(data)

        n_features = feature_counts.get(task_name, X_train.shape[1] if hasattr(X_train, 'shape') else 0)

        # Training/test set sizes
        n_train = len(y_train)
        n_test = len(y_test)

        unique_classes = np.unique(y_train)
        is_multiclass = len(unique_classes) > 2

        # EPV event definition:
        # - Binary: positive class count (label=1)
        # - Multiclass: minority-class count (conservative one-vs-rest proxy)
        if not is_multiclass:
            n_events_train = int(np.sum(y_train == 1))
            n_events_test = int(np.sum(y_test == 1))
            event_label = "1"
            epv_method = "Binary positive class count"
        else:
            train_counts = {int(c): int(np.sum(y_train == c)) for c in unique_classes}
            test_counts = {int(c): int(np.sum(y_test == c)) for c in np.unique(y_test)}

            minority_class_train = min(train_counts, key=train_counts.get)
            n_events_train = int(train_counts[minority_class_train])
            n_events_test = int(test_counts.get(minority_class_train, 0))
            event_label = str(minority_class_train)
            epv_method = "Multiclass minority-class count (conservative)"

        epv_metrics = calculate_epv_metrics(n_events_train, n_features, n_train)

        rows.append({
            'Task': task_name,
            'N_Train': n_train,
            'N_Test': n_test,
            'N_Total': n_train + n_test,
            'N_Events_Train': n_events_train,
            'N_Events_Test': n_events_test,
            'Event_Label_Used': event_label,
            'EPV_Method': epv_method,
            'N_Features': n_features,
            'EPV': epv_metrics['epv'],
            'EPV_Adequate': epv_metrics['epv_adequate'],
            'EPV_Interpretation': epv_metrics['epv_interpretation'],
            'Prevalence_Train': n_events_train / n_train if n_train > 0 else 0,
            'Prevalence_Test': n_events_test / n_test if n_test > 0 else 0,
            'Min_Events_EPV10': epv_metrics['min_events_epv10'],
            'Min_Events_EPV20': epv_metrics['min_events_epv20']
        })

    df = pd.DataFrame(rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"  Sample size report saved to: {output_path}")

    return df


# ==============================================================================
# 5. BCa BOOTSTRAP CONFIDENCE INTERVALS
# ==============================================================================

def bca_bootstrap_ci(data, stat_func, n_bootstrap=1000, alpha=0.05, random_state=None):
    """
    Calculate Bias-Corrected and Accelerated (BCa) bootstrap confidence intervals.

    BCa intervals are preferred over percentile intervals for:
    - Asymmetric distributions
    - Small samples
    - Bounded statistics (like AUC)

    Parameters:
    -----------
    data : array or tuple of arrays
        Data to bootstrap (if tuple, bootstrap indices applied to all)
    stat_func : callable
        Function that computes the statistic from data
    n_bootstrap : int
        Number of bootstrap iterations
    alpha : float
        Significance level (default 0.05 for 95% CI)
    random_state : int, optional
        Random seed

    Returns:
    --------
    dict with:
        - point_estimate: Original statistic
        - ci_lower: Lower CI bound
        - ci_upper: Upper CI bound
        - ci_method: 'BCa'
        - bias_correction: z0 value
        - acceleration: a value
    """
    rng = np.random.RandomState(random_state)

    # Handle tuple of arrays
    if isinstance(data, tuple):
        n = len(data[0])
        original_stat = stat_func(*data)
    else:
        n = len(data)
        original_stat = stat_func(data)

    # Bootstrap distribution
    boot_stats = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, n, replace=True)

        if isinstance(data, tuple):
            boot_sample = tuple(d[indices] if hasattr(d, '__getitem__') else d for d in data)
            boot_stat = stat_func(*boot_sample)
        else:
            boot_sample = data[indices]
            boot_stat = stat_func(boot_sample)

        boot_stats.append(boot_stat)

    boot_stats = np.array(boot_stats)

    # Remove NaN values
    boot_stats = boot_stats[~np.isnan(boot_stats)]

    if len(boot_stats) < 10:
        return {
            'point_estimate': original_stat,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'ci_method': 'BCa (insufficient valid bootstraps)',
            'bias_correction': np.nan,
            'acceleration': np.nan
        }

    # Bias correction factor (z0)
    prop_less = np.mean(boot_stats < original_stat)
    prop_less = np.clip(prop_less, 0.001, 0.999)  # Avoid infinite z
    z0 = stats.norm.ppf(prop_less)

    # Acceleration factor (a) using jackknife
    jackknife_stats = []
    for i in range(n):
        jack_indices = np.concatenate([np.arange(i), np.arange(i + 1, n)])

        if isinstance(data, tuple):
            jack_sample = tuple(d[jack_indices] if hasattr(d, '__getitem__') else d for d in data)
            jack_stat = stat_func(*jack_sample)
        else:
            jack_sample = data[jack_indices]
            jack_stat = stat_func(jack_sample)

        jackknife_stats.append(jack_stat)

    jackknife_stats = np.array(jackknife_stats)
    jackknife_stats = jackknife_stats[~np.isnan(jackknife_stats)]

    if len(jackknife_stats) < 2:
        # Fall back to percentile method
        ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return {
            'point_estimate': original_stat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_method': 'Percentile (jackknife failed)',
            'bias_correction': z0,
            'acceleration': np.nan
        }

    jack_mean = np.mean(jackknife_stats)
    jack_diff = jack_mean - jackknife_stats

    # Acceleration
    a_num = np.sum(jack_diff ** 3)
    a_denom = 6 * (np.sum(jack_diff ** 2)) ** 1.5
    a = a_num / a_denom if a_denom != 0 else 0

    # BCa confidence intervals
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # Adjusted percentiles
    def adjusted_percentile(z_alpha):
        numerator = z0 + z_alpha
        denominator = 1 - a * (z0 + z_alpha)
        if denominator == 0:
            return 0.5
        return stats.norm.cdf(z0 + numerator / denominator)

    p_lower = adjusted_percentile(z_alpha_lower)
    p_upper = adjusted_percentile(z_alpha_upper)

    # Clip to valid range
    p_lower = np.clip(p_lower, 0.001, 0.999)
    p_upper = np.clip(p_upper, 0.001, 0.999)

    ci_lower = np.percentile(boot_stats, 100 * p_lower)
    ci_upper = np.percentile(boot_stats, 100 * p_upper)

    return {
        'point_estimate': original_stat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_method': 'BCa',
        'bias_correction': z0,
        'acceleration': a
    }


# ==============================================================================
# 6. SUBGROUP ANALYSIS
# ==============================================================================

def perform_subgroup_analysis(y_true, y_prob, subgroup_var, subgroup_name,
                             model_name, threshold=0.5):
    """
    Perform subgroup analysis for a single stratification variable.

    Parameters:
    -----------
    y_true : array
        True labels
    y_prob : array
        Predicted probabilities
    subgroup_var : array
        Subgroup variable (categorical)
    subgroup_name : str
        Name of the subgroup variable
    model_name : str
        Name of the model
    threshold : float
        Classification threshold

    Returns:
    --------
    DataFrame with metrics by subgroup
    """
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    subgroup_var = np.asarray(subgroup_var)

    results = []
    unique_groups = np.unique(subgroup_var[~pd.isna(subgroup_var)])

    for group in unique_groups:
        mask = subgroup_var == group

        if np.sum(mask) < 10:  # Minimum subgroup size
            continue

        y_true_sub = y_true[mask]
        y_prob_sub = y_prob[mask]
        y_pred_sub = (y_prob_sub >= threshold).astype(int)

        n_sub = len(y_true_sub)
        n_events = np.sum(y_true_sub)

        # Skip if only one class
        if len(np.unique(y_true_sub)) < 2:
            auc = np.nan
        else:
            try:
                auc = roc_auc_score(y_true_sub, y_prob_sub)
            except:
                auc = np.nan

        # Balanced accuracy
        try:
            bal_acc = balanced_accuracy_score(y_true_sub, y_pred_sub)
        except:
            bal_acc = np.nan

        # Sensitivity and specificity
        tp = np.sum((y_pred_sub == 1) & (y_true_sub == 1))
        tn = np.sum((y_pred_sub == 0) & (y_true_sub == 0))
        fp = np.sum((y_pred_sub == 1) & (y_true_sub == 0))
        fn = np.sum((y_pred_sub == 0) & (y_true_sub == 1))

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        results.append({
            'Model': model_name,
            'Subgroup_Variable': subgroup_name,
            'Subgroup_Value': group,
            'N': n_sub,
            'N_Events': n_events,
            'Prevalence': n_events / n_sub,
            'AUC': auc,
            'Balanced_Accuracy': bal_acc,
            'Sensitivity': sensitivity,
            'Specificity': specificity
        })

    return pd.DataFrame(results)


# ==============================================================================
# 7. TRIPOD COMPLIANCE CHECKLIST
# ==============================================================================

def generate_tripod_checklist(study_info=None, output_path=None):
    """
    Generate a TRIPOD (Transparent Reporting of a multivariable prediction model
    for Individual Prognosis Or Diagnosis) compliance checklist.

    Parameters:
    -----------
    study_info : dict, optional
        Dictionary with study-specific information
    output_path : str, optional
        Path to save checklist

    Returns:
    --------
    DataFrame with TRIPOD checklist items
    """

    checklist = [
        # Title and Abstract
        {'Section': 'Title', 'Item': '1', 'Description': 'Identify the study as developing and/or validating a multivariable prediction model, the target population, and the outcome to be predicted', 'Reported': '', 'Location': ''},
        {'Section': 'Abstract', 'Item': '2', 'Description': 'Provide a summary of objectives, study design, setting, participants, sample size, predictors, outcome, statistical analysis, results, and conclusions', 'Reported': '', 'Location': ''},

        # Introduction
        {'Section': 'Background', 'Item': '3a', 'Description': 'Explain the medical context and rationale for developing/validating the prediction model', 'Reported': '', 'Location': ''},
        {'Section': 'Objectives', 'Item': '3b', 'Description': 'Specify the objectives, including whether the study describes development or validation', 'Reported': '', 'Location': ''},

        # Methods
        {'Section': 'Source of Data', 'Item': '4a', 'Description': 'Describe the study design or source of data, separately for development and validation datasets', 'Reported': '', 'Location': ''},
        {'Section': 'Source of Data', 'Item': '4b', 'Description': 'Specify the key study dates, including start/end of accrual, and length of follow-up', 'Reported': '', 'Location': ''},
        {'Section': 'Participants', 'Item': '5a', 'Description': 'Specify key elements of the study setting, locations, and relevant dates', 'Reported': '', 'Location': ''},
        {'Section': 'Participants', 'Item': '5b', 'Description': 'Describe eligibility criteria for participants', 'Reported': '', 'Location': ''},
        {'Section': 'Participants', 'Item': '5c', 'Description': 'Give details of any treatments received', 'Reported': '', 'Location': ''},
        {'Section': 'Outcome', 'Item': '6a', 'Description': 'Clearly define the outcome that is predicted, including how and when assessed', 'Reported': '', 'Location': ''},
        {'Section': 'Outcome', 'Item': '6b', 'Description': 'Report any actions to blind assessment of the outcome to be predicted', 'Reported': '', 'Location': ''},
        {'Section': 'Predictors', 'Item': '7a', 'Description': 'Clearly define all predictors used in developing the model, including how and when measured', 'Reported': '', 'Location': ''},
        {'Section': 'Predictors', 'Item': '7b', 'Description': 'Report any actions to blind assessment of predictors for the outcome and other predictors', 'Reported': '', 'Location': ''},
        {'Section': 'Sample Size', 'Item': '8', 'Description': 'Explain how the study size was arrived at', 'Reported': '', 'Location': ''},
        {'Section': 'Missing Data', 'Item': '9', 'Description': 'Describe how missing data were handled with details of any imputation method', 'Reported': '', 'Location': ''},
        {'Section': 'Statistical Analysis', 'Item': '10a', 'Description': 'Describe how predictors were handled in the analyses', 'Reported': '', 'Location': ''},
        {'Section': 'Statistical Analysis', 'Item': '10b', 'Description': 'Specify type of model, all model-building procedures, and method for internal validation', 'Reported': '', 'Location': ''},
        {'Section': 'Statistical Analysis', 'Item': '10c', 'Description': 'For validation, describe how the predictions were calculated', 'Reported': '', 'Location': ''},
        {'Section': 'Statistical Analysis', 'Item': '10d', 'Description': 'Specify all measures used to assess model performance and how they were calculated', 'Reported': '', 'Location': ''},
        {'Section': 'Risk Groups', 'Item': '11', 'Description': 'Provide details on how risk groups were created, if done', 'Reported': '', 'Location': ''},

        # Results
        {'Section': 'Participants', 'Item': '13a', 'Description': 'Describe the flow of participants through the study, including number with missing data', 'Reported': '', 'Location': ''},
        {'Section': 'Participants', 'Item': '13b', 'Description': 'Describe the characteristics of the participants, including number of outcome events', 'Reported': '', 'Location': ''},
        {'Section': 'Model Development', 'Item': '14a', 'Description': 'Specify the number of participants and outcome events in each analysis', 'Reported': '', 'Location': ''},
        {'Section': 'Model Development', 'Item': '14b', 'Description': 'If done, report the unadjusted association between each predictor and outcome', 'Reported': '', 'Location': ''},
        {'Section': 'Model Specification', 'Item': '15a', 'Description': 'Present the full prediction model to allow predictions for individuals', 'Reported': '', 'Location': ''},
        {'Section': 'Model Specification', 'Item': '15b', 'Description': 'Explain how to use the prediction model', 'Reported': '', 'Location': ''},
        {'Section': 'Model Performance', 'Item': '16', 'Description': 'Report performance measures (with CIs) for the prediction model', 'Reported': '', 'Location': ''},

        # Discussion
        {'Section': 'Limitations', 'Item': '18', 'Description': 'Discuss any limitations of the study', 'Reported': '', 'Location': ''},
        {'Section': 'Interpretation', 'Item': '19a', 'Description': 'For validation, discuss the results with reference to other validation studies', 'Reported': '', 'Location': ''},
        {'Section': 'Interpretation', 'Item': '19b', 'Description': 'Give an overall interpretation of the results, considering objectives and limitations', 'Reported': '', 'Location': ''},
        {'Section': 'Implications', 'Item': '20', 'Description': 'Discuss the potential clinical use of the model and implications for future research', 'Reported': '', 'Location': ''},

        # Other
        {'Section': 'Supplementary', 'Item': '21', 'Description': 'Provide information about the availability of supplementary resources', 'Reported': '', 'Location': ''},
        {'Section': 'Funding', 'Item': '22', 'Description': 'Give the source of funding and the role of the funders', 'Reported': '', 'Location': ''},
    ]

    df = pd.DataFrame(checklist)

    # Add study-specific info if provided
    if study_info:
        for key, value in study_info.items():
            matching_items = df[df['Item'] == key]
            if not matching_items.empty:
                df.loc[df['Item'] == key, 'Reported'] = value.get('reported', '')
                df.loc[df['Item'] == key, 'Location'] = value.get('location', '')

    if output_path:
        df.to_csv(output_path, index=False)

        # Also save as formatted text
        txt_path = output_path.replace('.csv', '.txt')
        with open(txt_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TRIPOD CHECKLIST FOR PREDICTION MODEL STUDIES\n")
            f.write("=" * 80 + "\n\n")

            current_section = ''
            for _, row in df.iterrows():
                if row['Section'] != current_section:
                    f.write(f"\n{row['Section'].upper()}\n")
                    f.write("-" * 40 + "\n")
                    current_section = row['Section']

                f.write(f"  [{row['Item']}] {row['Description']}\n")
                if row['Reported']:
                    f.write(f"      Reported: {row['Reported']}\n")
                if row['Location']:
                    f.write(f"      Location: {row['Location']}\n")

        print(f"  TRIPOD checklist saved to: {output_path}")

    return df


# ==============================================================================
# MAIN RUNNER
# ==============================================================================

def run_epidemiology_analysis(data_df, results_list, task_data, feature_names_dict,
                              output_dir, include_subgroups=None):
    """
    Run comprehensive epidemiological analysis.

    Parameters:
    -----------
    data_df : DataFrame
        Original data (for missing data analysis)
    results_list : list
        List of model results
    task_data : dict
        Dictionary with task data
    feature_names_dict : dict
        Dictionary mapping task names to feature names
    output_dir : str
        Directory to save results
    include_subgroups : list, optional
        List of column names to use for subgroup analysis

    Returns:
    --------
    dict with all analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("EPIDEMIOLOGICAL ANALYSIS")
    print("=" * 70)

    all_results = {}

    # 1. Missing Data Analysis
    print("\n=== Missing Data Analysis ===")
    if data_df is not None:
        missing_results = analyze_missing_patterns(
            data_df,
            os.path.join(output_dir, "missing_data_report.txt")
        )
        all_results['missing_data'] = missing_results

    # 2. Sample Size Justification
    print("\n=== Sample Size Justification ===")
    feature_counts = {task: len(features) for task, features in feature_names_dict.items()}
    sample_size_df = generate_sample_size_report(
        task_data,
        feature_counts,
        os.path.join(output_dir, "sample_size_report.csv")
    )
    all_results['sample_size'] = sample_size_df

    # 3. Calibration Metrics
    print("\n=== Calibration Metrics (Slope, Intercept, H-L Test) ===")
    calibration_results = []

    task_results_map = {}
    for result in results_list:
        task_name = result.get("LabelDefinition", "Unknown")
        if task_name not in task_results_map:
            task_results_map[task_name] = []
        task_results_map[task_name].append(result)

    for task_name, results in task_results_map.items():
        if task_name not in task_data:
            continue

        is_multi = "Multi:" in task_name
        if is_multi:
            continue

        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(task_data[task_name])

        for result in results:
            model = result.get("FinalModel")
            model_name = result.get("ModelName", "Unknown")

            if model is not None and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    calib = calculate_calibration_metrics(y_test, y_prob)

                    calibration_results.append({
                        'Task': task_name,
                        'Model': model_name,
                        'Calibration_Slope': calib.get('calibration_slope', np.nan),
                        'Calibration_Intercept': calib.get('calibration_intercept', np.nan),
                        'Brier_Score': calib.get('brier_score', np.nan),
                        'ECE': calib.get('ece', np.nan),
                        'MCE': calib.get('mce', np.nan),
                        'HL_Chi2': calib.get('hosmer_lemeshow_chi2', np.nan),
                        'HL_PValue': calib.get('hosmer_lemeshow_pvalue', np.nan),
                        'Slope_Interpretation': calib.get('slope_interpretation', '')
                    })
                except Exception as e:
                    print(f"    Error for {model_name}: {e}")

    if calibration_results:
        calib_df = pd.DataFrame(calibration_results)
        calib_df.to_csv(os.path.join(output_dir, "calibration_metrics.csv"), index=False)
        print(f"  Saved calibration metrics for {len(calibration_results)} models")
        all_results['calibration'] = calib_df

    # 4. Decision Curve Analysis
    print("\n=== Decision Curve Analysis ===")
    for task_name, results in task_results_map.items():
        if task_name not in task_data:
            continue

        is_multi = "Multi:" in task_name
        if is_multi:
            continue

        print(f"  Processing: {task_name}")

        X_train, y_train, X_test, y_test, _, _ = _unpack_task_data(task_data[task_name])

        model_probs = {}
        model_names = []

        for result in results:
            model = result.get("FinalModel")
            model_name = result.get("ModelName", "Unknown")

            if model is not None and hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_test)[:, 1]
                    model_probs[model_name] = y_prob
                    model_names.append(model_name)
                except:
                    continue

        if model_names:
            safe_name = task_name.replace('/', '_').replace(' ', '_').replace(':', '').replace('.', '')
            dca_df, dca_summary = decision_curve_analysis(
                y_test,
                model_probs,
                model_names,
                output_path=os.path.join(output_dir, f"decision_curve_{safe_name}.csv")
            )

    # 5. TRIPOD Checklist
    print("\n=== TRIPOD Checklist ===")
    tripod_df = generate_tripod_checklist(
        output_path=os.path.join(output_dir, "tripod_checklist.csv")
    )
    all_results['tripod'] = tripod_df

    print("\n" + "=" * 70)
    print("EPIDEMIOLOGICAL ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    print("Epidemiology utilities for clinical prediction model validation.")
    print("\nAvailable functions:")
    print("  - littles_mcar_test(data)")
    print("  - analyze_missing_patterns(data)")
    print("  - decision_curve_analysis(y_true, model_predictions, model_names)")
    print("  - calculate_calibration_metrics(y_true, y_prob)")
    print("  - calculate_epv_metrics(n_events, n_predictors)")
    print("  - bca_bootstrap_ci(data, stat_func)")
    print("  - perform_subgroup_analysis(...)")
    print("  - generate_tripod_checklist()")
    print("  - run_epidemiology_analysis(...)")
# fmt: on
