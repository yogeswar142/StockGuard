"""
src/calibration.py - Empirical Probability Calibrator for StockGuard.

Calibrates raw Bayesian posteriors or candidate confidence scores using Platt scaling
(logistic regression on logit-transformed probabilities) or Isotonic regression,
strictly fitted on the calibration period (first 60% of timeline).

Ensures that:
- An alert flagged with 15% confidence actually hits the true phantom state ~15% of the time.
- No misleading "99.9%" probability claims for rare events (~5% base rate).
- Brier score is minimized on the out-of-sample test period.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.special import logit, expit
from typing import Optional, Union, Tuple


class ConfidenceCalibrator:
    """
    Fits and applies empirical calibration to raw posterior odds/probabilities.
    """
    def __init__(self, method: str = "platt", clip_bounds: Tuple[float, float] = (1e-4, 1.0 - 1e-4)):
        self.method = method.lower()
        self.clip_bounds = clip_bounds
        self.model = None
        self.is_fitted = False

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        """
        Fits calibration model on calibration-period data.
        """
        raw_probs = np.asarray(raw_probs, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        
        if len(raw_probs) == 0 or len(np.unique(y_true)) < 2:
            self.is_fitted = False
            return self

        if self.method == "platt":
            clipped_p = np.clip(raw_probs, self.clip_bounds[0], self.clip_bounds[1])
            X = logit(clipped_p).reshape(-1, 1)
            self.model = LogisticRegression(C=1.0, solver="lbfgs")
            self.model.fit(X, y_true)
            self.is_fitted = True
        elif self.method == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.90)
            self.model.fit(raw_probs, y_true)
            self.is_fitted = True
        return self

    def predict(self, raw_probs: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Transforms raw probability into empirically calibrated probability.
        """
        if not self.is_fitted or self.model is None:
            return raw_probs

        is_scalar = np.isscalar(raw_probs)
        arr = np.atleast_1d(raw_probs).astype(float)

        if self.method == "platt":
            clipped = np.clip(arr, self.clip_bounds[0], self.clip_bounds[1])
            X = logit(clipped).reshape(-1, 1)
            cal_p = self.model.predict_proba(X)[:, 1]
        elif self.method == "isotonic":
            cal_p = self.model.predict(arr)
        else:
            cal_p = arr

        cal_p = np.clip(cal_p, 0.005, 0.95)
        return float(cal_p[0]) if is_scalar else cal_p
