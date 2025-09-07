# src/models/policy_estimator.py
import numpy as np

def find_optimal_discount(base_effect, max_discount=0.3, cost_per_discount=10):
    """
    Simple policy: maximize profit under cost constraint.
    """
    discounts = np.linspace(0, max_discount, 100)
    profits = [base_effect * d - cost_per_discount * d for d in discounts]
    optimal_d = discounts[np.argmax(profits)]
    return optimal_d