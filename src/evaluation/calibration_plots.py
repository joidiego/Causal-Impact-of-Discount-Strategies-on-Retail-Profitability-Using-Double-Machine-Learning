# src/evaluation/calibration_plots.py
import matplotlib.pyplot as plt

def plot_calibration(residual_T, residual_Y, save_path="plots/calibration.png"):
    plt.figure(figsize=(8,6))
    plt.scatter(residual_T, residual_Y, alpha=0.6)
    z = np.polyfit(residual_T, residual_Y, 1)
    p = np.poly1d(z)
    plt.plot(residual_T, p(residual_T), "r--", alpha=0.8)
    plt.xlabel("T - m(X)")
    plt.ylabel("Y - g(X)")
    plt.title("Calibration Plot")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()