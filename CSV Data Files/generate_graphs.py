import matplotlib.pyplot as plt
import pandas as pd
from os import listdir

metrics_folder = "metrics"
images_folder = "images"

def size_of_largest_component_graph(df, removal_target, removal_strategy, locality, shortform):
    
    removed = df[f'{removal_target}s_removed']
    size = df['largest_strongly_connected_component_size']

    removed = removed.apply(lambda x: 100*x / removed[len(removed)-1])
    size = size.apply(lambda x: 100*x / size[0])

    auc = 0
    for i in size:
        auc += i

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

def generate_graphs(df, removal_target, removal_strategy, locality):
    print(f"Drawing graphs for: {metrics_folder}/{filename}")

    # let's not make every filename too long
    targ = removal_target[0]
    strat = removal_strategy[0].upper()+"Btn" if "betweenness" in removal_strategy else removal_strategy[0].upper()+removal_strategy[1:4]
    shortfrom = targ+strat+"_"+locality

    # generate graphs # TODO uncomment these functions, for now commented so identical graphs aren't regenerated
    size_of_largest_component_graph(df, removal_target, removal_strategy, locality, shortfrom)

for filename in listdir(metrics_folder):
    # take information from file name assumes format <target>_removal_<strategy>_<locality>.csv
    removal_target = filename[:filename.find("_")]
    removal_strategy = filename[filename.find("removal_")+len("removal_"):filename.rfind("_")]
    locality = filename[filename.rfind("_")+1:filename.rfind(".csv")]

    removal_strategy = removal_strategy.replace("_", " ")

    df = pd.read_csv(f"{metrics_folder}/{filename}")

    generate_graphs(df, removal_target, removal_strategy, locality)