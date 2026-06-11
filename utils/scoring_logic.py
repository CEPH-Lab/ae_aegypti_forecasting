import numpy as np
import pandas as pd

# --- Setting up the quantiles ---
CDC_QUANTILES_SEQ = np.round(
    np.insert(np.arange(0.01, 1.00, 0.01), (2, -2), [0.025, 0.975]), 
    6
)
NUM_QUANTILES = CDC_QUANTILES_SEQ.shape[0]

# --- 2. Setting up the Alphas (Including 1.0) ---
CDC_ALPHAS = np.round(2 * CDC_QUANTILES_SEQ[:(NUM_QUANTILES + 1) // 2], 6)

CUTOFF = 0.25

def qcol(q: float) -> str:
    return f"q{q:.3f}"

# --- WIS function ---
def weighted_interval_score_fast(
    observations, alphas, q_dict, weights=None, percent=False, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.
    
    This function implements the WIS-score (2). A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.
    
    This is a more efficient implementation using array operations instead of repeated calls of `interval_score`.
    
    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.
        
    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.
        
    (2) Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint arXiv:2005.12881.
    """
    if weights is None:
        weights = np.array(alphas)/2

    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        raise ValueError("Alpha values must be sorted in ascending order.")

    reversed_weights = list(reversed(weights))


    lower_quantiles = [q_dict.get(round(alpha / 2,6)) for alpha in alphas]
    upper_quantiles = [q_dict.get(round(1 - (alpha / 2),6)) for alpha in reversed(alphas)]
    if any(q is None for q in lower_quantiles) or any(
        q is None for q in upper_quantiles
    ):
        raise ValueError(
            f"Quantile dictionary does not include all necessary quantiles."
        )

    lower_quantiles = np.vstack(lower_quantiles)
    upper_quantiles = np.vstack(upper_quantiles)

    # Check for consistency
    if check_consistency and np.any(
        np.diff(np.vstack((lower_quantiles, upper_quantiles)), axis=0) < 0
    ):
        raise ValueError("Quantiles are not consistent.")

    lower_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    upper_q_alphas = (2 / np.array(list(reversed(alphas)))).reshape((-1, 1))

    # compute score components for all intervals
    sharpnesses = np.flip(upper_quantiles, axis=0) - lower_quantiles

    lower_calibrations = (
        np.clip(lower_quantiles - observations, a_min=0, a_max=None) * lower_q_alphas
    )
    upper_calibrations = (
        np.clip(observations - upper_quantiles, a_min=0, a_max=None) * upper_q_alphas
    )
    calibrations = lower_calibrations + np.flip(upper_calibrations, axis=0)

    # scale to percentage absolute error
    if percent:
        sharpnesses = sharpnesses / np.abs(observations)
        calibrations = calibrations / np.abs(observations)

    totals = sharpnesses + calibrations

    # weigh scores
    weights = np.array(weights).reshape((-1, 1))

    sharpnesses_weighted = sharpnesses * weights
    calibrations_weighted = calibrations * weights
    totals_weighted = totals * weights

    # normalize and aggregate all interval scores
    weights_sum = np.sum(weights)

    sharpnesses_final = np.sum(sharpnesses_weighted, axis=0) / weights_sum
    calibrations_final = np.sum(calibrations_weighted, axis=0) / weights_sum
    totals_final = np.sum(totals_weighted, axis=0) / weights_sum

    return totals_final, sharpnesses_final, calibrations_final

# --- 3. THE ROW-BY-ROW SCORING ENGINE ---
def calculate_row_metrics(df, alphas=CDC_ALPHAS):
    """
    Appends sMAPE, WIS, and all Coverage columns (cov_2 to cov_98) to the dataframe.
    """
    # Ensure observed values exist
    df = df[df['observed_abund'].notna()].copy()
    
    y_true = df['observed_abund'].values
    y_pred = df['q0.500'].values if 'q0.500' in df.columns else df['forecasted_abund'].values

    # 1. Vectorized sMAPE
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    df['sMAPE'] = np.where(denom == 0, 0, np.abs(y_true - y_pred) / denom) * 100

    # 2. Row-level WIS
    # Pre-extract quantile columns to speed up the loop
    q_cols = [qcol(q) for q in CDC_QUANTILES_SEQ]
    q_matrix = df[q_cols].values
    
    wis_scores = []
    for i in range(len(df)):
        # Map current row's quantiles
        row_q_dict = {round(q, 6): q_matrix[i, j] for j, q in enumerate(CDC_QUANTILES_SEQ)}
        obs_val = np.array([y_true[i]])
        
        score, _, _ = weighted_interval_score_fast(obs_val, alphas, row_q_dict)
        wis_scores.append(score[0])
    
    df['WIS'] = wis_scores

    # 3. Binary Coverage for all intervals
    for alpha in alphas:
        if alpha >= 1.0: continue # Coverage is not defined for a single point (median)
        
        width = int(round((1 - alpha) * 100))
        lo_col = qcol(round(alpha / 2, 6))
        hi_col = qcol(round(1 - (alpha / 2), 6))
        
        # Binary check: 1 if within range, 0 if outside
        df[f'cov_{width}'] = ((df['observed_abund'] >= df[lo_col]) & 
                              (df['observed_abund'] <= df[hi_col])).astype(int)
    
    return df