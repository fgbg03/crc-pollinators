import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import unicodedata


edges = pd.read_csv('Edges_data_genus_level.csv')
nodes = pd.read_csv('Nodes_data_genus_level.csv')

# edges = edges[edges['Source'].str.contains(r'\(Coimbra\)', regex=True) | edges['Source'].str.contains(r'\(Anafi\)', regex=True)]
# nodes = nodes[nodes['Locality'].isin(['Coimbra', 'Anafi'])]

G = nx.Graph()

for index, row in nodes.iterrows():
    G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
for index, row in edges.iterrows():
    G.add_edge(row['Source'], row['Target'], weight=row['Weight'])

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

pos_geo = {}
for _, row in nodes.iterrows():

    lon, lat = float(row['Longitude']), float(row['Latitude'])
    random_lon = lon + np.random.uniform(-0.001, 0.001)
    random_lat = lat + np.random.uniform(-0.001, 0.001)
    pos_geo[row['Id']] = (random_lon, random_lat)

sccs = list(nx.connected_components(G))
print("Number of strongly connected components:"+ str(len(sccs)))

plt.figure(figsize=(200, 160))
plt.rcParams['font.family'] = 'Arial'
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")


pos = nx.spring_layout(G,k=0.01, scale=1,pos=pos_geo, iterations=5, fixed=pos_geo.keys())
print("Positions calculated")

component_positions = {}
for i, comp in enumerate(sccs):
    if comp == set() or len(comp) == 1:
        continue
    coords = np.array([pos_geo[n] for n in comp]) 
    centroid = coords.mean(axis=0)
    label = next(iter(comp)).split(' (')[1]  
    label = label[:-1]
    label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    if label not in component_positions:
        component_positions[label] = centroid

edge_count = 0
node_count = 0

print("Drawing nodes...")
color_map = []
for node in G:
    ax.plot(pos[node][0], pos[node][1], 'o', color='green' if G.nodes[node]['type']=="Plant" else 'pink',
            markersize=4, transform=ccrs.PlateCarree(), alpha=0.7)
    node_count += 1

ax.plot()

print("Drawing edges...")
for u, v in G.edges():
    lon1, lat1 = pos[u][0], pos[u][1]
    lon2, lat2 = pos[v][0], pos[v][1]

    ax.plot([lon1, lon2], [lat1, lat2], color="gray", linewidth=0.5,
            transform=ccrs.PlateCarree(), alpha=0.5)
    edge_count += 1

print("Adding labels...")
for label, (x, y) in component_positions.items():
    ax.text(x, y, label,
        fontsize=5, fontweight="bold",
        ha="center", va="center",
        transform=ccrs.PlateCarree())
    
plt.title("Pollination Network at Genus Level", fontsize=20)
    
print("Graph drawn successfully.")
print(f"Total nodes drawn: {node_count}")
print(f"Total edges drawn: {edge_count}")
plt.show()
