"""
Amount-manipulation adversarial attack for the IEEE-CIS fraud models.

Implements a black-box-style evasion attack under a deliberately narrow,
realistic threat model: the only feature a fraudster can manipulate is
TransactionAmt (within a multiplicative budget rho), with the coupled
feature `amt_z_for_card` recomputed to keep the transaction internally
consistent. All other features -- including the sparse V/id columns and
identity attributes -- are frozen, since a real fraudster has no control
over them.

Because the attack is one-dimensional (amount only), it works directly
against non-differentiable models (XGBoost) via a candidate-grid search,
and against differentiable models (logistic regression, the MLP) via the
same grid evaluated through their native scoring path. No gradients or
substitute-model transfer are required for this restricted threat model.

Fair comparison across models requires a shared starting point: models are
attacked from a *recall-matched* operating threshold (see
`threshold_for_recall`), not each model's default cutoff, because raw
probability thresholds are not comparable across model families.

Typical use:
    cand = make_adversarial_amounts(orig_amounts, rho=0.5, n_steps=50)
    evaded, probs = run_amount_attack_xgb(Xa, cand, xgb, threshold,
                                           amt_idx, amtz_idx, card_means, card_stds)
    sei = security_evaluation_index(rho_values, acc_values)
"""

import numpy as np
from sklearn.metrics import precision_recall_curve


def make_adversarial_amounts(orig_amounts, rho, n_steps=50):
    """Generate a grid of candidate amounts within +/- rho (fraction) of each original.

    Args:
        orig_amounts: array of original TransactionAmt values, shape (n_txn,)
        rho: perturbation budget as a fraction of the original amount (e.g. 0.5 = +/-50%)
        n_steps: number of candidate amounts per transaction

    Returns:
        candidates: array of shape (n_txn, n_steps)
    """
    lows = orig_amounts * (1 - rho)
    highs = orig_amounts * (1 + rho)
    steps = np.linspace(0, 1, n_steps)
    return lows[:, None] + steps[None, :] * (highs - lows)[:, None]


def _apply_perturbation(X_attack_raw, candidates, amt_idx, amtz_idx, card_means, card_stds):
    """Build the batch of perturbed rows, recomputing the coupled amt_z_for_card feature."""
    n_txn, n_steps = candidates.shape
    X_big = np.repeat(X_attack_raw, n_steps, axis=0)
    amt_flat = candidates.reshape(-1)
    X_big[:, amt_idx] = amt_flat

    means_rep = np.repeat(card_means, n_steps)
    stds_rep = np.repeat(card_stds, n_steps)
    X_big[:, amtz_idx] = (amt_flat - means_rep) / stds_rep

    return X_big, n_txn, n_steps


def run_amount_attack_xgb(X_attack_raw, candidates, model, threshold,
                           amt_idx, amtz_idx, card_means, card_stds):
    """Attack a model that handles missing values natively (e.g. XGBoost).

    Returns:
        evaded: bool array, shape (n_txn,) -- True if any candidate amount
                dropped the fraud probability below `threshold`
        probs: array, shape (n_txn, n_steps) -- fraud probability per candidate
    """
    X_big, n_txn, n_steps = _apply_perturbation(
        X_attack_raw, candidates, amt_idx, amtz_idx, card_means, card_stds
    )
    probs = model.predict_proba(X_big)[:, 1].reshape(n_txn, n_steps)
    evaded = (probs < threshold).any(axis=1)
    return evaded, probs


def run_amount_attack_sklearn(X_attack_raw, candidates, model, threshold,
                               amt_idx, amtz_idx, card_means, card_stds,
                               median_vals, mu, sigma):
    """Attack a model requiring imputation + scaling (e.g. scikit-learn LogisticRegression).

    Mirrors the exact preprocessing the model was trained on: impute missing
    values with train medians, then standardize with train mean/std, before
    scoring. `median_vals`, `mu`, `sigma` must all be fit on TRAIN data only.
    """
    X_big, n_txn, n_steps = _apply_perturbation(
        X_attack_raw, candidates, amt_idx, amtz_idx, card_means, card_stds
    )

    nan_mask = np.isnan(X_big)
    if nan_mask.any():
        X_big[nan_mask] = np.take(median_vals, np.where(nan_mask)[1])

    X_big_sc = (X_big - mu) / sigma

    probs = model.predict_proba(X_big_sc)[:, 1].reshape(n_txn, n_steps)
    evaded = (probs < threshold).any(axis=1)
    return evaded, probs


def run_amount_attack_torch(X_attack_raw, candidates, model, threshold,
                             amt_idx, amtz_idx, card_means, card_stds,
                             median_vals, mu, sigma, device):
    """Attack a PyTorch model (e.g. the MLP), scoring on the given device.

    `model` is expected to output raw logits; sigmoid is applied here.
    """
    import torch

    X_big, n_txn, n_steps = _apply_perturbation(
        X_attack_raw, candidates, amt_idx, amtz_idx, card_means, card_stds
    )

    nan_mask = np.isnan(X_big)
    if nan_mask.any():
        X_big[nan_mask] = np.take(median_vals, np.where(nan_mask)[1])

    X_big_sc = (X_big - mu) / sigma

    with torch.no_grad():
        xb = torch.tensor(X_big_sc, dtype=torch.float32, device=device)
        probs = torch.sigmoid(model(xb)).cpu().numpy().reshape(n_txn, n_steps)

    evaded = (probs < threshold).any(axis=1)
    return evaded, probs


def threshold_for_recall(probs, y_true, target_recall):
    """Find the probability threshold achieving (approximately) a target recall.

    Used to put models on a fair, recall-matched footing before comparing
    their security curves -- raw default thresholds are not comparable
    across model families with different probability calibrations.
    """
    prec, rec, thr = precision_recall_curve(y_true, probs)
    idx = np.argmin(np.abs(rec[:-1] - target_recall))
    return thr[idx]


def security_evaluation_index(rho_values, acc_values):
    """Compute the Security Evaluation Index: normalized area under the Acc(rho) curve.

    Follows Xiao et al. (2023), eq. 11-12. Range [0, 1]; higher = more secure
    (the model retains more of its accuracy as attack strength increases).

    Args:
        rho_values: increasing attack-strength values, rho[0] == 0
        acc_values: model accuracy on the attacked set at each rho (as fractions)
    """
    rho_arr = np.asarray(rho_values, dtype=float)
    acc_arr = np.asarray(acc_values, dtype=float)
    return np.trapezoid(acc_arr, rho_arr) / rho_arr.max()


def run_security_sweep(rho_values, attack_fn, *attack_args, **attack_kwargs):
    """Run an attack across a range of rho values and return (results, SEI).

    `attack_fn` must be one of the run_amount_attack_* functions above, called
    as attack_fn(X_attack_raw, candidates, ..., *attack_args, **attack_kwargs)
    where `candidates` is generated fresh per rho. This helper expects
    `attack_fn` already bound to everything except (X_attack_raw, candidates);
    see the accompanying notebook for the exact call pattern used per model
    type, since each model's attack signature differs slightly.
    """
    raise NotImplementedError(
        "Call the model-specific run_amount_attack_* function directly in a "
        "loop over rho_values -- signatures differ enough across model types "
        "(XGBoost vs. sklearn vs. torch) that a single generic wrapper would "
        "obscure more than it saves. See notebooks/adversarial_robustness.ipynb "
        "for the per-model sweep pattern."
    )
