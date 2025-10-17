# scripts/motif_analysis.py
# V-motif and network resilience analysis (locality based)

import os
import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations
from scipy.spatial.distance import cdist
from sklearn.metrics import auc


# ---------- Data loading ---------- #
def load_data(nodes_path, edges_path):
    nodes = pd.read_csv(nodes_path)
    edges = pd.read_csv(edges_path)
    return nodes, edges


# ---------- Build network by locality ---------- #
def build_locality_graph(edges_df, nodes_df, locality):
    """Build bipartite plant–pollinator network for one locality."""
    nd = nodes_df[nodes_df["Locality"] == locality]
    genus_ids = set(nd["Id"])

    G = nx.Graph()
    for _, r in nd.iterrows():
        G.add_node(r["Id"],
                   bipartite=0 if r["Type"] == "Plant" else 1,
                   type=r["Type"])

    for _, r in edges_df.iterrows():
        if r["Source"] in genus_ids and r["Target"] in genus_ids:
            G.add_edge(r["Source"], r["Target"], weight=r.get("Weight", 1.0))
    return G


# ---------- Motif counting ---------- #
def v_lambda_counts(G):
    plants = [n for n, d in G.nodes(data=True) if d.get("type") == "Plant"]
    pollinators = [n for n, d in G.nodes(data=True) if d.get("type") == "Pollinator"]
    V = L = R = 0

    # V motifs: two pollinators sharing one plant
    for p in plants:
        nbrs = [v for v in G.neighbors(p) if G.nodes[v]["type"] == "Pollinator"]
        if len(nbrs) >= 2:
            V += len(list(combinations(nbrs, 2)))

    # Λ motifs: two plants sharing one pollinator
    for a in pollinators:
        nbrs = [v for v in G.neighbors(a) if G.nodes[v]["type"] == "Plant"]
        if len(nbrs) >= 2:
            L += len(list(combinations(nbrs, 2)))

    # Rectangles: four-node motifs
    for p1, p2 in combinations(plants, 2):
        shared_pollinators = set(G.neighbors(p1)).intersection(G.neighbors(p2))
        if len(shared_pollinators) >= 2:
            R += len(list(combinations(list(shared_pollinators), 2)))

    return {"V": V, "Lambda": L, "Rect": R}


# ---------- Motif profiles ---------- #
def motif_profile(G):
    counts = v_lambda_counts(G)
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def gcc_fraction(G):
    if G.number_of_nodes() == 0:
        return 0.0
    if G.number_of_edges() == 0:
        return 1.0 if G.number_of_nodes() == 1 else 0.0
    cc = max(nx.connected_components(G), key=len)
    return len(cc) / G.number_of_nodes()


# ---------- Attack simulation ---------- #
def attack_sequence(G, strategy="random", step=0.05):
    """Simulate node removals with guaranteed progress; record actual removed fraction."""
    H = G.copy()
    n0 = H.number_of_nodes()
    if n0 == 0:
        return pd.DataFrame([{"frac_removed": 0.0, "gcc_frac": 0.0, **motif_profile(H)}])

    removed_so_far = 0
    records = []

    while H.number_of_nodes() > 0 and (removed_so_far / n0) <= 0.60 + 1e-9:
        # remove at least 1 node each step, based on initial n0
        k = max(1, int(round(step * n0)))
        k = min(k, H.number_of_nodes())

        if strategy == "random":
            rem = np.random.choice(list(H.nodes()), size=k, replace=False)
        elif strategy == "degree":
            deg = sorted(H.degree, key=lambda x: x[1], reverse=True)
            rem = [n for n, _ in deg[:k]]
        elif strategy == "betweenness":
            bw = nx.betweenness_centrality(H, normalized=True)
            rem = [n for n, _ in sorted(bw.items(), key=lambda x: x[1], reverse=True)[:k]]
        else:
            rem = np.random.choice(list(H.nodes()), size=k, replace=False)

        H.remove_nodes_from(rem)
        removed_so_far += k
        frac_removed = removed_so_far / n0

        rec = {"frac_removed": min(frac_removed, 0.60), "gcc_frac": gcc_fraction(H), **motif_profile(H)}
        records.append(rec)

        if frac_removed >= 0.60 or H.number_of_nodes() == 0:
            break

    # ensure we have a starting point at 0 for smooth AUC
    if not records or records[0]["frac_removed"] > 0:
        start = {"frac_removed": 0.0, "gcc_frac": gcc_fraction(G), **motif_profile(G)}
        records.insert(0, start)

    return pd.DataFrame(records)

# ---------- Motif drift ---------- #
def motif_drift(df, baseline, metric="braycurtis"):
    """
    Compute motif drift between each step and the baseline motif composition.
    Uses Bray–Curtis distance (robust for compositional data, handles zeros).
    """
    import numpy as np
    from scipy.spatial.distance import cdist

    # Extract motif frequencies
    X = df[["V", "Lambda", "Rect"]].values.astype(float)
    b = np.array([[baseline["V"], baseline["Lambda"], baseline["Rect"]]], dtype=float)

    # Compute Bray–Curtis distance between each row and baseline
    D = cdist(X, b, metric=metric).ravel()

    # Return updated dataframe
    df = df.copy()
    df["motif_drift"] = D
    return df


# ---------- Resilience summary ---------- #
def summarize_resilience(traj_df):
    a_struct = auc(traj_df["frac_removed"], traj_df["gcc_frac"])
    a_drift = auc(traj_df["frac_removed"], traj_df["motif_drift"])
    return a_struct, a_drift


# ---------- Full pipeline ---------- #
def run_motif_resilience(nodes_path, edges_path, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    nodes, edges = load_data(nodes_path, edges_path)
    localities = nodes["Locality"].dropna().unique()

    results = []
    for loc in localities:
        print(f"Processing locality: {loc}")
        G = build_locality_graph(edges, nodes, loc)
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            print(f"  Skipping {loc}: empty network.")
            continue

        base = motif_profile(G)
        for strat in ["random", "degree", "betweenness"]:
            traj = attack_sequence(G, strategy=strat, step=0.05)
            traj = motif_drift(traj, base, metric="braycurtis")
            a_struct, a_drift = summarize_resilience(traj)
            results.append({
                "Locality": loc,
                "Strategy": strat,
                "AUC_structural": a_struct,
                "AUC_motif_drift": a_drift
            })

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(output_dir,
                                   "motif_resilience_summary.csv"),
                      index=False)
    return df_results
