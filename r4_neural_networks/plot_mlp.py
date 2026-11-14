import pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

panels = [("with_artist", "All features"),
          ("selected_3", "Selected 3 features"),
          ("without_artist", "Without artist features")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True, sharey=True)
for ax, (name, title) in zip(axes, panels):
    d = pd.read_csv(f"mlp_test_predictions_{name}.csv")
    r2 = r2_score(d.y_true, d.y_pred)
    rmse = mean_squared_error(d.y_true, d.y_pred) ** 0.5
    ax.scatter(d.y_true, d.y_pred, s=3, alpha=0.12)
    ax.plot([0, 100], [0, 100], "r--", label="perfect prediction")
    ax.set_title(f"MLP: {title}\nR² = {r2:.3f}, RMSE = {rmse:.2f}")
    ax.set_xlabel("actual popularity")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
axes[0].set_ylabel("predicted popularity")
axes[0].legend(loc="upper left")
plt.tight_layout()
plt.savefig("mlp_pred_vs_actual.png", dpi=150)
print("saved mlp_pred_vs_actual.png")