# scripts/analyze_tradeoff.py
# Structure–function resilience trade-off + correlations

import os
import pandas as pd
import matplotlib.pyplot as plt

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson & Spearman correlations overall and per Strategy."""
    rows = []

    def corr_block(name, sub):
        sub = sub.dropna(subset=["AUC_structural", "AUC_motif_drift"])
        if len(sub) < 3:
            return  # not enough points for a meaningful correlation
        pearson = sub[["AUC_structural", "AUC_motif_drift"]].corr(method="pearson").iloc[0, 1]
        spearman = sub[["AUC_structural", "AUC_motif_drift"]].corr(method="spearman").iloc[0, 1]
        rows.append({"Group": name, "N": len(sub), "Pearson_r": pearson, "Spearman_rho": spearman})

    # Overall
    corr_block("Overall", df)

    # Per strategy
    for strat, sub in df.groupby("Strategy"):
        corr_block(f"Strategy={strat}", sub)

    return pd.DataFrame(rows).sort_values(["Group"]).reset_index(drop=True)

def plot_tradeoff(csv_path="results/motif_resilience_summary.csv", output_dir="results/visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    # Load summary results
    df = pd.read_csv(csv_path).dropna(subset=["AUC_structural", "AUC_motif_drift"])

    # ---- Scatter (matplotlib, no extra deps) ----
    # color per strategy
    strategy_colors = {
        "random": "tab:blue",
        "degree": "tab:orange",
        "betweenness": "tab:green"
    }

    plt.figure(figsize=(8, 6))
    for strat, sub in df.groupby("Strategy"):
        plt.scatter(
            sub["AUC_structural"],
            sub["AUC_motif_drift"],
            label=strat,
            s=30,
            alpha=0.8
        )

    # annotate a few notable points (top struct / lowest drift)
    best_struct = df.sort_values("AUC_structural", ascending=False).head(3)
    best_drift  = df.sort_values("AUC_motif_drift", ascending=True).head(3)
    for _, row in pd.concat([best_struct, best_drift]).drop_duplicates().iterrows():
        plt.text(row["AUC_structural"], row["AUC_motif_drift"], row["Locality"][:16], fontsize=7, ha="left", va="bottom")

    plt.xlabel("Structural Resilience (AUC of GCC)")
    plt.ylabel("Functional Instability (AUC of Motif Drift)  ↓ better if lower")
    plt.title("Structure–Function Resilience Trade-off Across Localities", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Attack Strategy")
    out_plot = os.path.join(output_dir, "resilience_tradeoff_scatter.png")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"✅ Trade-off plot saved to {out_plot}")

    # ---- Correlations ----
    corr_df = compute_correlations(df)
    out_corr = os.path.join(output_dir, "..", "resilience_tradeoff_correlations.csv")
    corr_df.to_csv(out_corr, index=False)
    print("✅ Correlations saved to", os.path.abspath(out_corr))
    print("\nCorrelation summary:\n", corr_df.to_string(index=False))

if __name__ == "__main__":
    plot_tradeoff()
