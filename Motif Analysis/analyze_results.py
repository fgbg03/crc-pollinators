import pandas as pd

# Load your results summary
df = pd.read_csv("results/motif_resilience_summary.csv")

def rank_strategy(df, strategy):
    d = df[df["Strategy"] == strategy].copy()
    d["rank_struct"] = d["AUC_structural"].rank(ascending=False, method="min")
    d["rank_drift"] = d["AUC_motif_drift"].rank(ascending=True, method="min")
    d["rank_total"] = d["rank_struct"] + d["rank_drift"]
    return d.sort_values(["rank_total", "rank_struct", "rank_drift"])

print("\n=== RANDOM ATTACK ===")
print(rank_strategy(df, "random")[["Locality", "AUC_structural", "AUC_motif_drift", "rank_total"]].head(10))

print("\n=== DEGREE-TARGETED ATTACK ===")
print(rank_strategy(df, "degree")[["Locality", "AUC_structural", "AUC_motif_drift", "rank_total"]].head(10))

print("\n=== BETWEENNESS-TARGETED ATTACK ===")
print(rank_strategy(df, "betweenness")[["Locality", "AUC_structural", "AUC_motif_drift", "rank_total"]].head(10))
