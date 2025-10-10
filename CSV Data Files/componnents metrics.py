import pandas as pd
import networkx as nx
import numpy as np
import unicodedata


edges = pd.read_csv('Edges_data_genus_level.csv')
nodes = pd.read_csv('Nodes_data_genus_level.csv')

localities = {}

max_weight = 0
max_edges = 0
max_avg_weight = 0
max_avg_degree = 0
for index, row in edges.iterrows():
    locality = row['Source']
    weight = row['Weight']
    source = row['Source']
        
        # Extract locality name from the format "Name (Locality)"
        # Example: "Apis mellifera (Berchtesgaden National Park)" -> "Berchtesgaden National Park"
    
    parsed_locality = locality.split(' (')[1][:-1]
    
    if parsed_locality not in localities:
        localities[parsed_locality] = {'weight': 0, 'edges': 0, 'average_degree': 0, 'node_degrees': {}}
    
    localities[parsed_locality]['weight'] += weight
    localities[parsed_locality]['edges'] += 1
    edges_number = localities[parsed_locality]['edges']

    if weight > max_weight:
        max_weight = weight
    if edges_number > max_edges:
        max_edges = edges_number
    
    if source not in localities[parsed_locality]['node_degrees']:
        localities[parsed_locality]['node_degrees'][source] = 0
    localities[parsed_locality]['node_degrees'][source] += weight

for locality, data in localities.items():
    weight = data['weight']
    edges_number = data['edges']
    average_degree = sum(data['node_degrees'].values()) / len(data['node_degrees'])
    localities[locality]['average_degree'] = average_degree

    if max_avg_weight < (weight / edges_number):
        max_avg_weight = weight / edges_number
    if max_avg_degree < average_degree:
        max_avg_degree = average_degree

sorted_localities = sorted(localities.items(), key=lambda x: (x[1]['edges'] / max_edges) + (x[1]['weight'] / max_weight) + (x[1]['weight']/x[1]['edges'] / max_avg_weight) + (x[1]['average_degree'] / max_avg_degree) if x[1]['edges'] > 0 else 0, reverse=True)

print (f"Max weight: {max_weight}, Max edges: {max_edges}, Max average weight: {max_avg_weight}, Max average degree: {max_avg_degree}")
for locality, data in sorted_localities[:10]:
    avg_weight = data['weight'] / data['edges'] if data['edges'] > 0 else 0
    print(f"Locality: {locality}, Total Weight: {data['weight']}, Edges: {data['edges']}, Average Weight per Edge: {avg_weight:.2f}, Average Weighted Degree: {data['average_degree']:.2f}")

centrality = {}
initial_locality_nodes = {}
for locality, data in sorted_localities[:3]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight']*(-1))
    centrality[locality] = nx.betweenness_centrality(G, weight='weight', normalized=True)
    strongly_connected_components = list(nx.connected_components(G))
    initial_locality_nodes[locality] = {'num_nodes': G.number_of_nodes(), 'largest_component_size': max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0, 'num_edges': G.number_of_edges()}
    print(f"Number of strongly connected components in {locality}: {len(strongly_connected_components)}")

for locality, cent in centrality.items():
    sorted_cent = sorted(cent.items(), key=lambda x: x[1], reverse=True)
    print(f"Top 5 central nodes in {locality}:")
    for node, cent_value in sorted_cent[:5]:
        print(f"Node: {node}, Betweenness Centrality: {cent_value:.4f}")

print("=====================\nEnd of Initial Metrics\n\n")



print("Starting Robustness Metrics based on recalculated centrality after each removal\n")
robusteness_locality_metrics = {}
for locality, data in sorted_localities[:3]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'] * (-1))
    robusteness_locality_metrics[locality] = {
        'size_of_largest_component' : [initial_locality_nodes[locality]['largest_component_size']],
        'number of strongly connected components': [len(list(nx.connected_components(G)))]
        }
    n_removed = 0
    while n_removed < initial_locality_nodes[locality]['num_nodes']:
        # Remove the node with the highest betweenness centrality
        node_to_remove = max(centrality[locality], key=centrality[locality].get)
        G.remove_node(node_to_remove)
        n_removed += 1
        centrality[locality] = nx.betweenness_centrality(G, weight='weight', normalized=True)
        strongly_connected_components = list(nx.connected_components(G))
        robusteness_locality_metrics[locality]['size_of_largest_component'].append(max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0)
        robusteness_locality_metrics[locality]['number of strongly connected components'].append(len(strongly_connected_components))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['size_of_largest_component'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number of strongly connected components'])
    print("\n")

print("End of Robustness Metrics based on recalculated centrality after each removal\n")



print("Starting Robustness Metrics based on initial centrality\n")
robusteness_locality_metrics = {}
for locality, data in sorted_localities[:3]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'] * (-1))
    robusteness_locality_metrics[locality] = {
        'size_of_largest_component' : [initial_locality_nodes[locality]['largest_component_size']],
        'number of strongly connected components': [len(list(nx.connected_components(G)))]
        }
    n_removed = 0
    centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
    centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    while n_removed < initial_locality_nodes[locality]['num_nodes']:
        # Remove the node with the highest betweenness centrality
        node_to_remove = centrality_sorted[n_removed][0]
        G.remove_node(node_to_remove)
        n_removed += 1
        strongly_connected_components = list(nx.connected_components(G))
        robusteness_locality_metrics[locality]['size_of_largest_component'].append(max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0)
        robusteness_locality_metrics[locality]['number of strongly connected components'].append(len(strongly_connected_components))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['size_of_largest_component'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number of strongly connected components'])
    print("\n")
print("End of Robustness Metrics based on initial centrality\n")



print("Starting Robusteness Metrics based on random removal\n")
robusteness_locality_metrics = {}
seed_range = 20
for locality, data in sorted_localities[:3]:
    robusteness_locality_metrics[locality] = {
        'size_of_largest_component' : [initial_locality_nodes[locality]['largest_component_size']] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'number of strongly connected components': [len(list(nx.connected_components(G)))] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        }

    for seed in range(seed_range):
        G = nx.Graph()
        edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
        nodes_locality = nodes[nodes['Locality'] == locality]
        for index, row in nodes_locality.iterrows():
            G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
        for index, row in edges_locality.iterrows():
            G.add_edge(row['Source'], row['Target'], weight=row['Weight'] * (-1))
        n_removed = 0
        node_list = list(G.nodes())
        np.random.seed(seed)
        np.random.shuffle(node_list)
        while n_removed < initial_locality_nodes[locality]['num_nodes']:
            # Remove the node with the highest betweenness centrality
            node_to_remove = node_list[n_removed]
            G.remove_node(node_to_remove)
            n_removed += 1
            strongly_connected_components = list(nx.connected_components(G))
            robusteness_locality_metrics[locality]['size_of_largest_component'][n_removed] += (max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0)
            robusteness_locality_metrics[locality]['number of strongly connected components'][n_removed] += (len(strongly_connected_components))
    robusteness_locality_metrics[locality]['size_of_largest_component'] = [robusteness_locality_metrics[locality]['size_of_largest_component'][0]] + [x / seed_range for x in robusteness_locality_metrics[locality]['size_of_largest_component'][1:]]
    robusteness_locality_metrics[locality]['number of strongly connected components'] = [robusteness_locality_metrics[locality]['number of strongly connected components'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['number of strongly connected components'][1:]]

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['size_of_largest_component'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number of strongly connected components'])
    print("\n")

print("End of Robusteness Metrics based on random removal\n")


print("Starting Robusteness Metrics based on edge removal according to betweenness centrality\n")

robusteness_locality_metrics = {}
for locality, data in sorted_localities[:3]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'] * (-1))
    robusteness_locality_metrics[locality] = {
        'size_of_largest_component' : [initial_locality_nodes[locality]['largest_component_size']],
        'number of strongly connected components': [len(list(nx.connected_components(G)))]
        }
    n_removed = 0
    while n_removed < initial_locality_nodes[locality]['num_edges']:
        centrality = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
        centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        # Remove the edge with the highest betweenness centrality
        edge_to_remove = centrality_sorted[0][0]
        G.edges[edge_to_remove]['weight'] -= 1
        if G.edges[edge_to_remove]['weight'] <= 0:
            G.remove_edge(*edge_to_remove)
            n_removed += 1
        if n_removed % 100 == 0:
            print(f"Progress in {locality}: {n_removed}/{initial_locality_nodes[locality]['num_edges']} edges removed")
        strongly_connected_components = list(nx.connected_components(G))
        robusteness_locality_metrics[locality]['size_of_largest_component'].append(max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0)
        robusteness_locality_metrics[locality]['number of strongly connected components'].append(len(strongly_connected_components))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['size_of_largest_component'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number of strongly connected components'])
    print("\n")

print("End of Robustness Metrics based on edge removal according to betweenness centrality\n")



print("Starting Robusteness Metrics based on edge removal according to initial betweenness centrality\n")

robusteness_locality_metrics = {}
for locality, data in sorted_localities[:3]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight=row['Weight'] * (-1))
    robusteness_locality_metrics[locality] = {
        'size_of_largest_component' : [initial_locality_nodes[locality]['largest_component_size']],
        'number of strongly connected components': [len(list(nx.connected_components(G)))]
        }
    centrality = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    n_removed = 0
    while n_removed < initial_locality_nodes[locality]['num_edges']:
        # Remove the edge with the highest betweenness centrality
        edge_to_remove = centrality_sorted[n_removed][0]
        G.remove_edge(*edge_to_remove)
        n_removed += 1
        strongly_connected_components = list(nx.connected_components(G))
        robusteness_locality_metrics[locality]['size_of_largest_component'].append(max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0)
        robusteness_locality_metrics[locality]['number of strongly connected components'].append(len(strongly_connected_components))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['size_of_largest_component'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number of strongly connected components'])
    print("\n")

print("End of Robustness Metrics based on edge removal according to initial betweenness centrality\n")



# print("Starting Robusteness Metrics based on edge removal based on decreasing the weight of a random edge\n")

# robusteness_locality_metrics = {}
# seed_range = 20
# for locality, data in sorted_localities[:3]:
#     total_weight = sorted_localities[locality]['weight']
#     robusteness_locality_metrics[locality] = {
#         'size_of_largest_component' : [initial_locality_nodes[locality]['largest_component_size']] +  ([0]*total_weight),
#         'number of strongly connected components': [len(list(nx.connected_components(G)))] +  ([0]*total_weight),
#         }

#     for seed in range(seed_range):
#         G = nx.Graph()
#         edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
#         nodes_locality = nodes[nodes['Locality'] == locality]
#         for index, row in nodes_locality.iterrows():
#             G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
#         for index, row in edges_locality.iterrows():
#             G.add_edge(row['Source'], row['Target'], weight=row['Weight'] * (-1))
#         n_removed = 0
#         node_list = list(G.nodes())
#         np.random.seed(seed)
#         np.random.shuffle(node_list)
#         while n_removed < total_weight:
#             # Remove the node with the highest betweenness centrality
#             node_to_remove = node_list[n_removed]
#             G.remove_node(node_to_remove)
#             n_removed += 1
#             strongly_connected_components = list(nx.connected_components(G))
#             robusteness_locality_metrics[locality]['size_of_largest_component'][n_removed] += (max(len(comp) for comp in strongly_connected_components) if strongly_connected_components else 0)
#             robusteness_locality_metrics[locality]['number of strongly connected components'][n_removed] += (len(strongly_connected_components))
#     robusteness_locality_metrics[locality]['size_of_largest_component'] = [robusteness_locality_metrics[locality]['size_of_largest_component'][0]] + [x / seed_range for x in robusteness_locality_metrics[locality]['size_of_largest_component'][1:]]
#     robusteness_locality_metrics[locality]['number of strongly connected components'] = [robusteness_locality_metrics[locality]['number of strongly connected components'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['number of strongly connected components'][1:]]

# for locality in robusteness_locality_metrics:
#     print(f"Robustness metrics for {locality}:")
#     print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['size_of_largest_component'])
#     print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number of strongly connected components'])
#     print("\n")

# print("End of Robustness Metrics based on edge removal based on decreasing the weight of a random edge\n")