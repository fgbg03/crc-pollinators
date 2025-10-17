import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import unicodedata

csv_dir = 'nodes_and_edges/'
edges = pd.read_csv(f'{csv_dir}/Edges_data_genus_level.csv')
nodes = pd.read_csv(f'{csv_dir}/Nodes_data_genus_level.csv')

# edges = edges[edges['Source'].str.contains(r'\(Coimbra\)', regex=True) | edges['Source'].str.contains(r'\(Anafi\)', regex=True)]
# nodes = nodes[nodes['Locality'].isin(['Coimbra', 'Anafi'])]

# print("Edges")
# print(edges)
# print("Nodes")
# print(nodes)

G = nx.Graph()

for index, row in nodes.iterrows():
    G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
for index, row in edges.iterrows():
    G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
# print(G)


pos_geo = {}

for _, row in nodes.iterrows():
    # base coordinate
    lon, lat = float(row['Longitude']), float(row['Latitude'])
    # print(row)
    # print(f"Original coordinates for {row['Id']}: ({lon}, {lat})")
    random_lon = lon + np.random.uniform(-0.05, 0.05)
    random_lat = lat + np.random.uniform(-0.05, 0.05)
    pos_geo[row['Id']] = (random_lon, random_lat)

# print(pos_geo)


# print(pos)
sccs = list(nx.connected_components(G))
print("Number of strongly connected components:"+ str(len(sccs)))

plt.figure(figsize=(100, 80))
plt.rcParams['font.family'] = 'Arial'

pos = nx.spring_layout(G,k=0.1, scale=1,pos=pos_geo, iterations=5)
print("Positions calculated")

component_positions = {}
for i, comp in enumerate(sccs):
    if comp == set() or len(comp) == 1:
        continue
    coords = np.array([pos[n] for n in comp]) 
    centroid = coords.mean(axis=0)
    label = next(iter(comp)).split(' (')[1]  
    label = label[:-1]
    label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    if label not in component_positions:
        component_positions[label] = centroid

print("Drawing graph...")

color_map = []
for node in G:
    if G.nodes[node]['type'] == 'Plant':
        color_map.append('green')
    else:
        color_map.append('pink')
print("Color map created")

nx.draw(G, pos, with_labels=False, node_size=3, node_color=color_map, edge_color='gray', width=0.2)

print("Adding labels...")
for label, (x, y) in component_positions.items():
    plt.text(x, y, label, fontsize=5, ha="center")

print("Graph drawn successfully.")
plt.title("Pollination Network at Genus Level", fontsize=20)
plt.show()
