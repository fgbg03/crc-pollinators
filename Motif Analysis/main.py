# main.py — V-Motif Resilience Analysis Runner

import os
import sys
import pandas as pd

# --- Ensure the 'scripts' folder is in the Python path --- #
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# --- Import functions from your scripts --- #
from motif_analysis import (
    run_motif_resilience,
    build_locality_graph,
    attack_sequence,
    motif_drift,
    motif_profile
)
from visualise_motif import plot_locality_curves, plot_overall_curves

# --- Define file paths --- #
DATA_NODES = 'data/Nodes_data_genus_level.csv'
DATA_EDGES = 'data/Edges_data_genus_level.csv'
RESULTS_DIR = 'results'
TRAJECTORIES_DIR = os.path.join(RESULTS_DIR, 'trajectories')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'visualizations')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TRAJECTORIES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Main analysis --- #
if __name__ == "__main__":
    print("Running V-Motif Resilience Analysis...\n")

    # Step 1: Run the summary
    res_df = run_motif_resilience(DATA_NODES, DATA_EDGES, output_dir=RESULTS_DIR)
    print(f"Summary saved to {RESULTS_DIR}/motif_resilience_summary.csv")

    # Step 2: Generate trajectories per locality
    nodes = pd.read_csv(DATA_NODES)
    edges = pd.read_csv(DATA_EDGES)
    localities = nodes['Locality'].dropna().unique()

    results_dict = {}

    for loc in localities:
        G = build_locality_graph(edges, nodes, loc)
        baseline = motif_profile(G)

        for strat in ['random', 'degree', 'betweenness']:
            print(f"Computing {strat} attack trajectory for {loc}...")
            traj = attack_sequence(G, strategy=strat, step=0.05)
            traj = motif_drift(traj, baseline)
            fname = f"{loc}_{strat}.csv"
            traj.to_csv(os.path.join(TRAJECTORIES_DIR, fname), index=False)
            results_dict[f"{loc}_{strat}"] = traj
            plot_locality_curves(traj, loc, strat, PLOTS_DIR)

    # Step 3: Create global visualizations
    print("\nCreating aggregate visualizations...")
    plot_overall_curves(results_dict, PLOTS_DIR)

    print("\nAll analyses and visualizations complete!")
    print(f"Results saved in: {RESULTS_DIR}")
    print(f"Plots saved in: {PLOTS_DIR}")
