"""
src/bayesian_detector.py - Bayesian Posterior Engine for StockGuard.

Calculates:
    P(Phantom | Zero Streak, Stock > 0, Corroborating Evidence)
using Bayes' theorem in odds form:
    Posterior Odds = Prior Odds * Bayes Factor * Corroborating Weights
    Posterior P = Odds / (1 + Odds)
"""

import numpy as np
from typing import Dict, Tuple, Optional

from src.streak_probability import streak_probability_negbin, streak_probability_poisson


class BayesianPhantomDetector:
    """
    Computes Bayesian posterior probability of phantom inventory.
    """
    def __init__(self, prior_phantom_rate: float = 0.028, alpha_dispersion: float = 0.30):
        self.prior_phantom_rate = prior_phantom_rate
        self.prior_odds = prior_phantom_rate / max(1e-6, 1.0 - prior_phantom_rate)
        self.alpha_dispersion = alpha_dispersion

    def compute_posterior(
        self,
        expected_daily_demand: float,
        zero_streak_days: int,
        peer_multiplier: float = 1.0,
        substitute_multiplier: float = 1.0,
        flat_stock_multiplier: float = 1.0,
        use_negbin: bool = True
    ) -> float:
        """
        Computes Bayesian posterior probability of phantom stock-out.
        """
        if zero_streak_days <= 0:
            return 0.0
            
        # P(zero streak | healthy in-stock)
        if use_negbin:
            p_healthy_streak = streak_probability_negbin(
                expected_daily_demand, zero_streak_days, alpha=self.alpha_dispersion
            )
        else:
            p_healthy_streak = streak_probability_poisson(
                expected_daily_demand, zero_streak_days
            )
            
        # Bayes factor = P(zero streak | phantom) / P(zero streak | healthy)
        # Under phantom, shelf is empty, so P(streak | phantom) = 1.0
        bayes_factor = 1.0 / max(1e-12, p_healthy_streak)
        
        # Total corroborating multiplier
        total_evidence_weight = peer_multiplier * substitute_multiplier * flat_stock_multiplier
        
        # Posterior Odds = Prior Odds * Bayes Factor * Evidence Weights
        posterior_odds = self.prior_odds * bayes_factor * total_evidence_weight
        
        # Convert odds back to probability
        posterior_p = posterior_odds / (1.0 + posterior_odds)
        return float(np.clip(posterior_p, 0.0, 0.9999))
