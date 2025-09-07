# src/evaluation/backtest.py
import numpy as np
import matplotlib.pyplot as plt

def policy_backtest(effect, discount_levels, baseline_profit=28.69):
    """
    Simulate profit under different discount policies.
    """
    impacts = effect * np.array(discount_levels)
    results = baseline_profit + impacts
    return dict(zip(discount_levels, results))

def plot_policy_backtest(backtest_results, save_path="plots/policy_backtest.png"):
    levels = list(backtest_results.keys())
    profits = list(backtest_results.values())
    plt.figure(figsize=(8,5))
    plt.plot(levels, profits, marker='o')
    plt.axhline(np.mean(profits), color='r', linestyle='--', alpha=0.7, label='Avg')
    plt.xlabel("Discount Level")
    plt.ylabel("Expected Profit")
    plt.title("Policy Backtest: Discount vs Profit")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()