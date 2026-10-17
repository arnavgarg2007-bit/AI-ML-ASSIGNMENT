"""
Interactive demonstration: Data Transformation (log, square-root, Box-Cox)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from scipy.stats import skew

rng = np.random.default_rng(7)
# A reproducible, positively skewed sample similar to income/price measurements.
data = rng.lognormal(mean=2.2, sigma=0.9, size=800)

fig, (ax_hist, ax_info) = plt.subplots(1, 2, figsize=(11, 5))
plt.subplots_adjust(left=0.27, bottom=0.22, wspace=0.32)

ax_choice = plt.axes([0.03, 0.55, 0.18, 0.25])
radio = RadioButtons(ax_choice, ("Original", "Log", "Square root", "Box-Cox"))
ax_lambda = plt.axes([0.32, 0.09, 0.55, 0.04])
lambda_slider = Slider(ax_lambda, "Box-Cox lambda", -1.0, 2.0, valinit=0.5, valstep=0.1)

def transform(values, method, lam):
    if method == "Original":
        return values
    if method == "Log":
        return np.log(values)
    if method == "Square root":
        return np.sqrt(values)
    if abs(lam) < 1e-9:
        return np.log(values)
    return (values ** lam - 1) / lam

def update(_=None):
    method = radio.value_selected
    transformed = transform(data, method, lambda_slider.val)
    ax_hist.clear()
    ax_hist.hist(transformed, bins=32, edgecolor="black", linewidth=0.25, alpha=0.8)
    ax_hist.set_title(f"{method} transformation")
    ax_hist.set_xlabel("Transformed value")
    ax_hist.set_ylabel("Frequency")
    ax_hist.grid(True, axis="y", alpha=0.25)

    ax_info.clear()
    ax_info.axis("off")
    text = (
        f"Selected method: {method}\n\n"
        f"Number of values: {len(data)}\n"
        f"Original skewness: {skew(data):.3f}\n"
        f"Transformed skewness: {skew(transformed):.3f}\n"
        f"Minimum after transform: {transformed.min():.3f}\n"
        f"Maximum after transform: {transformed.max():.3f}\n\n"
        "Tip: lower skewness magnitude can indicate\n"
        "a more symmetric distribution, but it does\n"
        "not guarantee a better predictive model."
    )
    ax_info.text(0.02, 0.95, text, va="top", fontsize=10, family="monospace")
    fig.canvas.draw_idle()

radio.on_clicked(update)
lambda_slider.on_changed(update)
update()
fig.suptitle("Explore how transformations change a data distribution", fontsize=13)
plt.show()