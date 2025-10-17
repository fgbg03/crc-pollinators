import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import powerlaw
from scipy.stats import linregress
from os import listdir

metrics_folder = "metrics"
images_folder = "images"

def size_of_largest_component_graph(df, removal_target, removal_strategy, locality, shortform):
    removed = df[f'{removal_target}s_removed']
    size = df['largest_strongly_connected_component_size']
    n_removals = removed[len(removed)-1]

    removed = removed.apply(lambda x: 100*x / n_removals)
    size = size.apply(lambda x: 100*x / size[0])

    auc = 0
    for i in size:
        auc += i / n_removals / 100 # give these values in 0.0-1.0 range

    plt.figure()
    plt.plot(removed, size, marker='o', linestyle='-')

    # Add labels and title
    plt.xlabel(f'%{removal_target.capitalize()}s Removed')
    plt.ylabel('%Component Size')
    plt.title(f'Largest component size in {locality}')
    plt.suptitle(f'{removal_strategy} {removal_target} removal'.capitalize())
    plt.grid(True)

    plt.text(
        85, 90,                                # x, y position (in data coordinates)
        f"Area Under Curve:\n{auc:,.02f}".replace(","," "),        # text
        fontsize=10,
        ha="center",va="center",
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')  # text box style
    )

    plt.savefig(f"{images_folder}/size_of_largest_component_{shortform}.png")
    plt.close()

def number_of_components_graph(df, removal_target, removal_strategy, locality, shortform):
    n_nodes = df['largest_strongly_connected_component_size'][0]
    removed = df[f'{removal_target}s_removed']
    components = df['number_of_strongly_connected_components']
    n_removals = removed[len(removed)-1]

    removed = removed.apply(lambda x: 100*x / n_removals)
    components = components.apply(lambda x: x/n_nodes)

    plt.figure()
    plt.plot(removed, components, marker='o', linestyle='-', color="magenta")

    # Add labels and title
    plt.xlabel(f'%{removal_target.capitalize()}s Removed')
    plt.ylabel('#Connected components/Node in initial GCC')
    plt.title(f'Number of connected components in {locality}')
    plt.suptitle(f'{removal_strategy} {removal_target} removal'.capitalize())
    plt.grid(True)

    plt.savefig(f"{images_folder}/number_of_connected_components_{shortform}.png")
    plt.close()

def average_path_length_graph(df, removal_target, removal_strategy, locality, shortform):
    removed = df[f'{removal_target}s_removed']
    apl = df['largest_strongly_connected_component_average_path_length']
    n_removals = removed[len(removed)-1]

    removed = removed.apply(lambda x: 100*x / n_removals)

    plt.figure()
    plt.plot(removed, apl, marker='o', linestyle='-', color="orange")

    # Add labels and title
    plt.xlabel(f'%{removal_target.capitalize()}s Removed')
    plt.ylabel('APL in the largest connected component')
    plt.title(f'Average Path Length in {locality}')
    plt.suptitle(f'{removal_strategy} {removal_target} removal'.capitalize())
    plt.grid(True)

    plt.savefig(f"{images_folder}/average_path_length_{shortform}.png")
    plt.close()

def average_degree_graph(df, removal_target, removal_strategy, locality, shortform):
    removed = df[f'{removal_target}s_removed']
    degree = df['average_degree']
    weighted_degree = df['average_weighted_degree']
    n_removals = removed[len(removed)-1]

    removed = removed.apply(lambda x: 100*x / n_removals)

    fig,axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(removed, degree, marker='o', linestyle='-', color="lightGreen")
    axes[1].plot(removed, weighted_degree, marker='o', linestyle='-', color="green")

    # Add labels and title
    axes[0].set_xlabel(f'%{removal_target.capitalize()}s Removed')
    axes[0].set_ylabel('Average Degree')
    axes[0].set_title(f'Average Degree in {locality}')
    axes[0].grid(True)

    axes[1].set_xlabel(f'%{removal_target.capitalize()}s Removed')
    axes[1].set_ylabel('Average Weighted Degree')
    axes[1].set_title(f'Average Weighted Degree in {locality}')
    axes[1].grid(True)

    fig.suptitle(f'{removal_strategy} {removal_target} removal'.capitalize())

    fig.savefig(f"{images_folder}/average_degree_{shortform}.png")
    plt.close(fig)

def generate_graphs(df, removal_target, removal_strategy, locality):
    print(f"Drawing graphs for: {metrics_folder}/{filename}")

    # let's not make every filename too long
    targ = removal_target[0]
    strat = removal_strategy[0].upper()+"Btn" if "betweenness" in removal_strategy else removal_strategy[0].upper()+removal_strategy[1:4]
    shortform = targ+strat+"_"+locality

    # generate graphs
    size_of_largest_component_graph(df, removal_target, removal_strategy, locality, shortform)
    number_of_components_graph(df, removal_target, removal_strategy, locality, shortform)
    average_path_length_graph(df, removal_target, removal_strategy, locality, shortform)
    average_degree_graph(df, removal_target, removal_strategy, locality, shortform)

def generate_degree_distribution_graphs(df, locality):
    print(f"Drawing degree distributions for: {locality}")
    
    degree_distribution = dict(eval(df["degree_distribution"][0]))
    weighted_degree_distribution = dict(eval(df["weighted_degree_distribution"][0]))
    n_nodes = int(df["nodes_removed"][len(df["nodes_removed"])-1]) if "n_nodes" not in df else int(df["n_nodes"][0])
    
    avg_degree = df["average_degree"][0]
    avg_w_degree = df["average_weighted_degree"][0]

    # degree

    degrees = list(degree_distribution.keys())
    deg_counts = list(degree_distribution.values())
    deg_probabilities = [x/n_nodes for x in deg_counts]
    degrees_repeated = [k for k, v in degree_distribution.items() for _ in range(int(v))]

    degrees = np.array(degrees)
    deg_probabilities = np.array(deg_probabilities)

    fit = powerlaw.Fit(degrees_repeated, discrete=True)

    x_fit = np.linspace(fit.xmin, max(degrees), 100)
    y_fit = x_fit ** (-fit.alpha)
    y_fit = y_fit / y_fit[0] * deg_probabilities[degrees >= fit.xmin][0]  # scale roughly to match

    y_min = deg_probabilities.min()
    mask = y_fit >= y_min
    x_fit = x_fit[mask]
    y_fit = y_fit[mask]


    fig,axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(degrees, deg_probabilities, marker="o", linestyle="", color="goldenrod")
    axes[1].plot(degrees, deg_probabilities, marker="o", linestyle="", color="darkgoldenrod")
    axes[1].plot(x_fit, y_fit, linestyle='--', color='b', label=f'Power law fit\nα={fit.power_law.alpha:,.02f}'.replace(","," "))
    axes[1].legend()


    # Add labels and title
    axes[0].set_xlabel(f'k')
    axes[0].set_ylabel('P(k)')
    axes[1].set_xlabel(f'k')
    axes[1].set_ylabel('P(k)')
    axes[0].set_title("linear scale")
    axes[1].set_title("logarithmic scale")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    axes[0].text(
        0.7, 0.9,                # position in axes coordinates (0–1)
        f"Average Degree:\n{avg_degree:,.02f}".replace(","," "),    # placeholder text
        transform=axes[0].transAxes,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    fig.suptitle(f'Degree distribution for {locality}')

    fig.savefig(f"{images_folder}/degree_distribution:{locality}.png")
    plt.close(fig)

    # degree wheighted by number of interactions

    interactions = list(weighted_degree_distribution.keys())
    interaction_counts = list(weighted_degree_distribution.values())
    interaction_probabilities = [x/n_nodes for x in interaction_counts]
    interactions_repeated = [k for k, v in weighted_degree_distribution.items() for _ in range(int(v))]

    interactions = np.array(interactions)
    interaction_probabilities = np.array(interaction_probabilities)

    fit = powerlaw.Fit(interactions_repeated, discrete=True)

    x_fit = np.linspace(fit.xmin, max(interactions), 100)
    y_fit = x_fit ** (-fit.alpha)
    y_fit = y_fit / y_fit[0] * interaction_probabilities[interactions >= fit.xmin][0]  # scale roughly to match

    y_min = interaction_probabilities.min()
    mask = y_fit >= y_min
    x_fit = x_fit[mask]
    y_fit = y_fit[mask]



    fig,axes = plt.subplots(1,2,figsize=(12,5))
    axes[0].plot(interactions, interaction_probabilities, marker="o", linestyle="", color="olive")
    axes[1].plot(interactions, interaction_probabilities, marker="o", linestyle="", color="olivedrab")
    axes[1].plot(x_fit, y_fit, linestyle='--', color='b', label=f'Power law fit\nα={fit.power_law.alpha:,.02f}'.replace(","," "))
    axes[1].legend()
    
    # Add labels and title
    axes[0].set_xlabel(f'k')
    axes[0].set_ylabel('P(k)')
    axes[1].set_xlabel(f'k')
    axes[1].set_ylabel('P(k)')
    axes[0].set_title("linear scale")
    axes[1].set_title("logarithmic scale")

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    axes[0].text(
        0.7, 0.9,                # position in axes coordinates (0–1)
        f"Average Weight:\n{avg_w_degree:,.02f}".replace(","," "),    # placeholder text
        transform=axes[0].transAxes,
        ha='center', va='center',
        fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
    )

    fig.suptitle(f'Weighted degree distribution for {locality}')

    fig.savefig(f"{images_folder}/weighted_degree_distribution:{locality}.png")
    plt.close(fig)

degree_distributions = []
for filename in listdir(metrics_folder):
    if filename[:4] != "edge" and filename[:4] != "node" and "removal" not in filename:
        continue
    # take information from file name assumes format <target>_removal_<strategy>_<locality>.csv
    removal_target = filename[:filename.find("_")]
    removal_strategy = filename[filename.find("removal_")+len("removal_"):filename.rfind("_")]
    locality = filename[filename.rfind("_")+1:filename.rfind(".csv")]

    removal_strategy = removal_strategy.replace("_", " ")

    df = pd.read_csv(f"{metrics_folder}/{filename}")

    #generate_graphs(df, removal_target, removal_strategy, locality)

    if locality not in degree_distributions and filename[:4] == "node":
        degree_distributions.append(locality)
        generate_degree_distribution_graphs(df, locality)

# whole graph degree distributions
df = pd.read_csv(f"{metrics_folder}/whole_network_degree_distributions.csv")
generate_degree_distribution_graphs(df, "the Whole Network")