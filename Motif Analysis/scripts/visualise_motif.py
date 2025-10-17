# scripts/visualize_motifs.py

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_locality_curves(traj_df, locality, strategy, output_dir):
    """Combined figure: GCC vs Motif Drift for a given locality & strategy"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Structural Robustness (GCC fraction)
    ax[0].plot(traj_df['frac_removed'], traj_df['gcc_frac'], marker='o', color='tab:blue')
    ax[0].set_title(f"{locality} — {strategy}\nStructural Robustness")
    ax[0].set_xlabel('Fraction of Nodes Removed')
    ax[0].set_ylabel('Giant Component Fraction')
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Motif Drift curve
    ax[1].plot(traj_df['frac_removed'], traj_df['motif_drift'], marker='o', color='tab:red')
    ax[1].set_title('Motif Drift (Cosine Distance)')
    ax[1].set_xlabel('Fraction of Nodes Removed')
    ax[1].set_ylabel('Distance from Baseline')
    ax[1].grid(True, linestyle='--', alpha=0.5)

    fig.suptitle(f"V-Motif Resilience — {locality} ({strategy} Attack)", fontsize=12)

    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{locality}_{strategy}_combined.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


def plot_overall_curves(results_dict, output_dir):
    """Aggregated plots comparing localities across strategies."""
    os.makedirs(output_dir, exist_ok=True)

    # Structural Robustness Comparison
    plt.figure(figsize=(6, 4))
    for key, df in results_dict.items():
        plt.plot(df['frac_removed'], df['gcc_frac'], label=key)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Giant Component Fraction')
    plt.title('Structural Robustness Across Localities')
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_curves.png'), dpi=300)
    plt.close()

    # Motif Drift Comparison
    plt.figure(figsize=(6, 4))
    for key, df in results_dict.items():
        plt.plot(df['frac_removed'], df['motif_drift'], label=key)
    plt.xlabel('Fraction of Nodes Removed')
    plt.ylabel('Motif Drift (Cosine Distance)')
    plt.title('Motif Drift Across Localities')
    plt.legend(fontsize=8)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'motif_drift_curves.png'), dpi=300)
    plt.close()


if __name__ == '__main__':
    # Example usage (expects per-locality trajectories saved separately)
    results_path = 'results/trajectories'
    out_dir = 'results/visualizations'
    results_dict = {}

    for file in os.listdir(results_path):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(results_path, file))
            key = file.replace('.csv', '')
            results_dict[key] = df
            # Extract locality and strategy from file name
            parts = key.split('_')
            if len(parts) >= 2:
                locality = parts[0]
                strategy = parts[1]
                plot_locality_curves(df, locality, strategy, out_dir)

    plot_overall_curves(results_dict, out_dir)
    print(f"All plots saved in {out_dir}")
