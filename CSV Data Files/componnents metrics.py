import pandas as pd
import networkx as nx
import numpy as np
import unicodedata
from collections import Counter

csv_metrics_dir = 'metrics'
desired_localities = 3          # number of localities to gather data from
alpha_weight = 5                # must not be 0, fraction of the average edge weight to remove in each iteration of edge removal

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Taking general network metrics and ordering the localities by a multi-factor criterion
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


edges = pd.read_csv('nodes_and_edges/Edges_data_genus_level.csv')
nodes = pd.read_csv('nodes_and_edges/Nodes_data_genus_level.csv')

localities = {}

max_weight = 0
max_edges = 0
max_avg_weight = 0
max_avg_degree = 0
network_weight = 0
for index, row in edges.iterrows():
    locality = row['Source']
    weight = row['Weight']
    source = row['Source']
    network_weight += weight
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

average_edge_weight = (max_weight * len(edges) - network_weight )/ len(edges)
print(f"Average edge weight: {average_edge_weight}")

sorted_localities = sorted(localities.items(), key=lambda x: (x[1]['edges'] / max_edges) + (x[1]['weight'] / max_weight) + (x[1]['weight']/x[1]['edges'] / max_avg_weight) + (x[1]['average_degree'] / max_avg_degree) if x[1]['edges'] > 0 else 0, reverse=True)

print (f"Max weight: {max_weight}, Max edges: {max_edges}, Max average weight: {max_avg_weight}, Max average degree: {max_avg_degree}")
for locality, data in sorted_localities[:10]:
    avg_weight = data['weight'] / data['edges'] if data['edges'] > 0 else 0
    print(f"Locality: {locality}, Total Weight: {data['weight']}, Edges: {data['edges']}, Average Weight per Edge: {avg_weight:.2f}, Average Weighted Degree: {data['average_degree']:.2f}")

centrality = {}
initial_locality_nodes = {}
for locality, data in sorted_localities[:desired_localities]:
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

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Node removal according to recalculated betweenness centrality
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------")
print("      Starting Robustness Metrics based on recalculated centrality after each removal\n")
print("--------------------------------------------------------------------------------------------------------------")

robusteness_locality_metrics = {}
for locality, data in sorted_localities[:desired_localities]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight= 1/row['Weight'], interactions=row['Weight'] )
    strongly_connected_components = list(nx.connected_components(G))
    largest_strongly_connected_component = max(strongly_connected_components, key=len)
    G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
    robusteness_locality_metrics[locality] = {
        'number_of_strongly_connected_components': [len(list(nx.connected_components(G)))],
        'largest_strongly_connected_component_size': [len(largest_strongly_connected_component)],
        'largest_strongly_connected_component_average_path_length': [nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight')] if nx.is_connected(G_largest_strongly_connected) else [float('inf')],
        'average_degree': [sum(dict(G.degree()).values()) / G.number_of_nodes()],
        'average_weighted_degree': [sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes()],
        'degree_distribution': [dict(Counter(dict(G.degree()).values()))],
        'weighted_degree_distribution': [dict(Counter(dict(G.degree(weight="interactions")).values()))]
    }
    n_removed = 0
    while n_removed < initial_locality_nodes[locality]['num_nodes']:
        # Remove the node with the highest betweenness centrality
        node_to_remove = max(centrality[locality], key=centrality[locality].get)
        G.remove_node(node_to_remove)
        n_removed += 1
        centrality[locality] = nx.betweenness_centrality(G, weight='weight', normalized=True)
        strongly_connected_components = list(nx.connected_components(G))
        largest_strongly_connected_component = max(strongly_connected_components, key=len) if strongly_connected_components else set()
        G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
        robusteness_locality_metrics[locality]['number_of_strongly_connected_components'].append(len(strongly_connected_components))
        robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'].append(len(largest_strongly_connected_component))
        robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'].append(nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and nx.is_connected(G_largest_strongly_connected) else float('inf'))
        robusteness_locality_metrics[locality]['average_degree'].append(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
        robusteness_locality_metrics[locality]['average_weighted_degree'].append(sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
        robusteness_locality_metrics[locality]['degree_distribution'].append(dict(Counter(dict(G.degree()).values())))
        robusteness_locality_metrics[locality]['weighted_degree_distribution'].append(dict(Counter(dict(G.degree(weight="interactions")).values())))


for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number_of_strongly_connected_components'])
    print("\n")
    
for locality_key, locality_metrics in robusteness_locality_metrics.items():
    parsed_metrics = {}
    for key, values in locality_metrics.items():
        col_name = f"{key}"
        parsed_metrics[col_name] = values
    df = pd.DataFrame(parsed_metrics)
    df.to_csv(f'{csv_metrics_dir}/node_removal_recalculated_betweenness_{locality_key}.csv', index=True, index_label='nodes_removed')

print("      End of Robustness Metrics based on recalculated centrality after each removal\n")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Node removal according to initial betweenness centrality
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------")
print("      Starting Robustness Metrics based on initial centrality\n")
print("--------------------------------------------------------------------------------------------------------------")

robusteness_locality_metrics = {}
for locality, data in sorted_localities[:desired_localities]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight= 1/row['Weight'], interactions=row['Weight'] )
    strongly_connected_components = list(nx.connected_components(G))
    largest_strongly_connected_component = max(strongly_connected_components, key=len)
    G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
    robusteness_locality_metrics[locality] = {
        'number_of_strongly_connected_components': [len(list(nx.connected_components(G)))],
        'largest_strongly_connected_component_size': [len(largest_strongly_connected_component)],
        'largest_strongly_connected_component_average_path_length': [nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight')] if nx.is_connected(G_largest_strongly_connected) else [float('inf')],
        'average_degree': [sum(dict(G.degree()).values()) / G.number_of_nodes()],
        'average_weighted_degree': [sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes()],
        'degree_distribution': [dict(Counter(dict(G.degree()).values()))],
        'weighted_degree_distribution': [dict(Counter(dict(G.degree(weight="interactions")).values()))]
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
        largest_strongly_connected_component = max(strongly_connected_components, key=len) if strongly_connected_components else set()
        G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
        robusteness_locality_metrics[locality]['number_of_strongly_connected_components'].append(len(strongly_connected_components))
        robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'].append(len(largest_strongly_connected_component))
        robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'].append(nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and nx.is_connected(G_largest_strongly_connected) else float('inf'))
        robusteness_locality_metrics[locality]['average_degree'].append(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
        robusteness_locality_metrics[locality]['average_weighted_degree'].append(sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
        robusteness_locality_metrics[locality]['degree_distribution'].append(dict(Counter(dict(G.degree()).values())))
        robusteness_locality_metrics[locality]['weighted_degree_distribution'].append(dict(Counter(dict(G.degree(weight="interactions")).values())))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number_of_strongly_connected_components'])
    print("\n")

for locality_key, locality_metrics in robusteness_locality_metrics.items():
    parsed_metrics = {}
    for key, values in locality_metrics.items():
        col_name = f"{key}"
        parsed_metrics[col_name] = values
    df = pd.DataFrame(parsed_metrics)
    df.to_csv(f'{csv_metrics_dir}/node_removal_initial_betweenness_{locality_key}.csv', index=True, index_label='nodes_removed')

print("      End of Robustness Metrics based on initial centrality\n")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Random node removal
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------")
print("      Starting Robusteness Metrics based on random removal\n")
print("--------------------------------------------------------------------------------------------------------------")

robusteness_locality_metrics = {}
seed_range = 20
for locality, data in sorted_localities[:desired_localities]:
    robusteness_locality_metrics[locality] = {
        'number_of_strongly_connected_components': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'largest_strongly_connected_component_size': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'largest_strongly_connected_component_average_path_length': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'average_degree': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'average_weighted_degree': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'degree_distribution': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes']),
        'weighted_degree_distribution': [0] +  ([0]*initial_locality_nodes[locality]['num_nodes'])
    }

    for seed in range(seed_range):
        G = nx.Graph()
        edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
        nodes_locality = nodes[nodes['Locality'] == locality]
        for index, row in nodes_locality.iterrows():
            G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
        for index, row in edges_locality.iterrows():
            G.add_edge(row['Source'], row['Target'], weight= 1/row['Weight'], interactions=row['Weight'] )
        if(seed == 0):
            strongly_connected_components = list(nx.connected_components(G))
            largest_strongly_connected_component = max(strongly_connected_components, key=len)
            G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
            robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][0] = len(list(nx.connected_components(G)))
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][0] = G_largest_strongly_connected.number_of_nodes()
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][0] = nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and G_largest_strongly_connected.number_of_nodes() > 0 else 0
            robusteness_locality_metrics[locality]['average_degree'][0] = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0
            robusteness_locality_metrics[locality]['average_weighted_degree'][0] = sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0
            robusteness_locality_metrics[locality]['degree_distribution'][0] = dict(Counter(dict(G.degree()).values()))
            robusteness_locality_metrics[locality]['weighted_degree_distribution'][0] = dict(Counter(dict(G.degree(weight="interactions")).values()))

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
            largest_strongly_connected_component = max(strongly_connected_components, key=len) if strongly_connected_components else set()
            G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][n_removed] += G_largest_strongly_connected.number_of_nodes()
            robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][n_removed] += (len(strongly_connected_components))
            APL = nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and G_largest_strongly_connected.number_of_nodes() > 0 else 0
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][n_removed] += APL
            avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0
            robusteness_locality_metrics[locality]['average_degree'][n_removed] += avg_deg
            avg_w_deg = sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0
            robusteness_locality_metrics[locality]['average_weighted_degree'][n_removed] += avg_w_deg
            # creating a 'cumulative' degree distributions so at the end we can get average values for the distributions
            deg_dist = dict(Counter(dict(G.degree()).values()))
            w_deg_dist = dict(Counter(dict(G.degree(weight="interactions")).values()))
            if seed == 0: # no distribution saved yet, save as is
                robusteness_locality_metrics[locality]['degree_distribution'][n_removed] = deg_dist
                robusteness_locality_metrics[locality]['weighted_degree_distribution'][n_removed] = w_deg_dist
            else: # make calculations
                deg_dist_cum = robusteness_locality_metrics[locality]['degree_distribution'][n_removed]
                w_deg_dist_cum = robusteness_locality_metrics[locality]['weighted_degree_distribution'][n_removed]
                for k in deg_dist.keys():
                    deg_dist_cum[k] = deg_dist[k] + (deg_dist_cum.get(k) or 0)
                for k in w_deg_dist.keys():
                    w_deg_dist_cum[k] = w_deg_dist[k] + (w_deg_dist_cum.get(k) or 0)
                robusteness_locality_metrics[locality]['degree_distribution'][n_removed] = deg_dist_cum
                robusteness_locality_metrics[locality]['weighted_degree_distribution'][n_removed] = w_deg_dist_cum

    # average the results from the multiple iterations
    robusteness_locality_metrics[locality]['number_of_strongly_connected_components'] = [robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][1:]]
    robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'] = [robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][0]] + [x / seed_range for x in robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][1:]]
    robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'] = [robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][0]] + [x / seed_range for x in robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][1:]]
    robusteness_locality_metrics[locality]['average_degree'] = [robusteness_locality_metrics[locality]['average_degree'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['average_degree'][1:]]
    robusteness_locality_metrics[locality]['average_weighted_degree'] = [robusteness_locality_metrics[locality]['average_weighted_degree'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['average_weighted_degree'][1:]]

    first = True
    for dist in robusteness_locality_metrics[locality]['degree_distribution']:
        if first: # initial value is not a sampled value, no need to average
            first = False
            continue
        for k in dist.keys():
            dist[k] = dist[k] / seed_range
    
    first = True
    for dist in robusteness_locality_metrics[locality]['weighted_degree_distribution']:
        if first: # initial value is not a sampled value, no need to average
            first = False
            continue
        for k in dist.keys():
            dist[k] = dist[k] / seed_range

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number_of_strongly_connected_components'])
    print("\n")

for locality_key, locality_metrics in robusteness_locality_metrics.items():
    parsed_metrics = {}
    for key, values in locality_metrics.items():
        col_name = f"{key}"
        parsed_metrics[col_name] = values
    df = pd.DataFrame(parsed_metrics)
    df.to_csv(f'{csv_metrics_dir}/node_removal_random_{locality_key}.csv', index=True, index_label='nodes_removed')

print("      End of Robusteness Metrics based on random removal\n")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Edge removal according to recalculated betweenness centrality
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------")
print("      Starting Robusteness Metrics based on edge removal according to betweenness centrality\n")
print("--------------------------------------------------------------------------------------------------------------")

robusteness_locality_metrics = {}
for locality, data in sorted_localities[:desired_localities]:
    total_weight = data['weight'] // 2
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight= 1/row['Weight'], interactions=row['Weight'] )
    strongly_connected_components = list(nx.connected_components(G))
    largest_strongly_connected_component = max(strongly_connected_components, key=len)
    G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
    robusteness_locality_metrics[locality] = {
        'number_of_strongly_connected_components': [len(list(nx.connected_components(G)))],
        'largest_strongly_connected_component_size': [G_largest_strongly_connected.number_of_nodes()],
        'largest_strongly_connected_component_average_path_length': [nx.average_shortest_path_length(G_largest_strongly_connected)] if G_largest_strongly_connected.number_of_nodes() > 0 else [0],
        'average_degree': [sum(dict(G.degree()).values()) / G.number_of_nodes()],
        'average_weighted_degree': [sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes()],
        'degree_distribution': [dict(Counter(dict(G.degree()).values()))],
        'weighted_degree_distribution': [dict(Counter(dict(G.degree(weight="interactions")).values()))]
    }
    while G.number_of_edges() > 0:
        centrality = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
        centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        # Remove the edge with the highest betweenness centrality
        edge_to_remove = centrality_sorted[0][0]
        G.edges[edge_to_remove]['interactions'] -= average_edge_weight // alpha_weight
        G.edges[edge_to_remove]['weight'] = 1/G.edges[edge_to_remove]['interactions'] if G.edges[edge_to_remove]['interactions'] != 0 else float('inf')
        if G.edges[edge_to_remove]['interactions'] <= 0:
            G.remove_edge(*edge_to_remove)
            strongly_connected_components = list(nx.connected_components(G))
            largest_strongly_connected_component = max(strongly_connected_components, key=len) if strongly_connected_components else set()
            G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
            robusteness_locality_metrics[locality]['number_of_strongly_connected_components'].append(len(strongly_connected_components))
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'].append(len(largest_strongly_connected_component))
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'].append(nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and nx.is_connected(G_largest_strongly_connected) else float('inf'))
            robusteness_locality_metrics[locality]['average_degree'].append(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
            robusteness_locality_metrics[locality]['average_weighted_degree'].append(sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
            robusteness_locality_metrics[locality]['degree_distribution'].append(dict(Counter(dict(G.degree()).values())))
            robusteness_locality_metrics[locality]['weighted_degree_distribution'].append(dict(Counter(dict(G.degree(weight="interactions")).values())))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number_of_strongly_connected_components'])
    print("\n")

for locality_key, locality_metrics in robusteness_locality_metrics.items():
    parsed_metrics = {}
    for key, values in locality_metrics.items():
        col_name = f"{key}"
        parsed_metrics[col_name] = values
    df = pd.DataFrame(parsed_metrics)
    df.to_csv(f'{csv_metrics_dir}/edge_removal_recalculated_betweenness_{locality_key}.csv', index=True, index_label='edges_removed')

print("      End of Robustness Metrics based on edge removal according to betweenness centrality\n")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Edge removal according to initial betweenness centrality
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------")
print("      Starting Robusteness Metrics based on edge removal according to initial betweenness centrality\n")
print("--------------------------------------------------------------------------------------------------------------")

robusteness_locality_metrics = {}
for locality, data in sorted_localities[:desired_localities]:
    G = nx.Graph()
    edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
    nodes_locality = nodes[nodes['Locality'] == locality]
    for index, row in nodes_locality.iterrows():
        G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
    for index, row in edges_locality.iterrows():
        G.add_edge(row['Source'], row['Target'], weight= 1/row['Weight'], interactions=row['Weight'] )
    strongly_connected_components = list(nx.connected_components(G))
    largest_strongly_connected_component = max(strongly_connected_components, key=len)
    G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
    robusteness_locality_metrics[locality] = {
        'number_of_strongly_connected_components': [len(list(nx.connected_components(G)))],
        'largest_strongly_connected_component_size': [len(largest_strongly_connected_component)],
        'largest_strongly_connected_component_average_path_length': [nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight')] if nx.is_connected(G_largest_strongly_connected) else [float('inf')],
        'average_degree': [sum(dict(G.degree()).values()) / G.number_of_nodes()],
        'average_weighted_degree': [sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes()],
        'degree_distribution': [dict(Counter(dict(G.degree()).values()))],
        'weighted_degree_distribution': [dict(Counter(dict(G.degree(weight="interactions")).values()))]
    }
    centrality = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    centrality_sorted = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
    n_removed = 0
    while G.number_of_edges() > 0:
        # Remove the edge with the highest betweenness centrality
        edge_to_remove = centrality_sorted[n_removed][0]
        G.remove_edge(*edge_to_remove)
        n_removed += 1
        strongly_connected_components = list(nx.connected_components(G))
        largest_strongly_connected_component = max(strongly_connected_components, key=len) if strongly_connected_components else set()
        G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
        robusteness_locality_metrics[locality]['number_of_strongly_connected_components'].append(len(strongly_connected_components))
        robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'].append(len(largest_strongly_connected_component))
        robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'].append(nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and nx.is_connected(G_largest_strongly_connected) else float('inf'))
        robusteness_locality_metrics[locality]['average_degree'].append(sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
        robusteness_locality_metrics[locality]['average_weighted_degree'].append(sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0)
        robusteness_locality_metrics[locality]['degree_distribution'].append(dict(Counter(dict(G.degree()).values())))
        robusteness_locality_metrics[locality]['weighted_degree_distribution'].append(dict(Counter(dict(G.degree(weight="interactions")).values())))

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number_of_strongly_connected_components'])
    print("\n")

for locality_key, locality_metrics in robusteness_locality_metrics.items():
    parsed_metrics = {}
    for key, values in locality_metrics.items():
        col_name = f"{key}"
        parsed_metrics[col_name] = values
    df = pd.DataFrame(parsed_metrics)
    df.to_csv(f'{csv_metrics_dir}/edge_removal_initial_betweenness_{locality_key}.csv', index=True, index_label='edges_removed')

print("      End of Robustness Metrics based on edge removal according to initial betweenness centrality\n")

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                   Edge removal based on decreasing the weight of a random edge
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

print("--------------------------------------------------------------------------------------------------------------")
print("      Starting Robusteness Metrics based on edge removal based on decreasing the weight of a random edge\n")
print("--------------------------------------------------------------------------------------------------------------")

robusteness_locality_metrics = {}
seed_range = 20
for locality, data in sorted_localities[:desired_localities]:
    total_weight = data['weight'] // 2
    
    robusteness_locality_metrics[locality] = {
        'number_of_strongly_connected_components': [0] +  ([0]*(data['edges'] // 2)),
        'largest_strongly_connected_component_size': [0] +  ([0]*(data['edges'] // 2)),
        'largest_strongly_connected_component_average_path_length': [0] +  ([0]*(data['edges'] // 2)),
        'average_degree': [0] +  ([0]*(data['edges'] // 2)),
        'average_weighted_degree': [0] +  ([0]*(data['edges'] // 2)),
        'degree_distribution': [0] +  ([0]*(data['edges'] // 2)),
        'weighted_degree_distribution': [0] +  ([0]*(data['edges'] // 2))
    }

    for seed in range(seed_range):
        G = nx.Graph()
        edges_locality = edges[edges['Source'].isin(nodes[nodes['Locality'] == locality]['Id'])]
        nodes_locality = nodes[nodes['Locality'] == locality]
        for index, row in nodes_locality.iterrows():
            G.add_node(row['Id'], label=row['Id'], group=row['Locality'], lon=row['Longitude'], lat=row['Latitude'], type=row['Type'])
        for index, row in edges_locality.iterrows():
            G.add_edge(row['Source'], row['Target'], weight= 1/row['Weight'], interactions=row['Weight'] )
        if(seed == 0):
            strongly_connected_components = list(nx.connected_components(G))
            largest_strongly_connected_component = max(strongly_connected_components, key=len)
            G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
            robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][0] = len(list(nx.connected_components(G)))
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][0] = G_largest_strongly_connected.number_of_nodes()
            robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][0] = nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and G_largest_strongly_connected.number_of_nodes() > 0 else 0

        n_removed = 0
        np.random.seed(seed)
        edges_weights = 0
        while G.number_of_edges() > 0:
            edge_list = list(G.edges())
            np.random.shuffle(edge_list)
            edge_to_remove = edge_list[0]
            G.edges[edge_to_remove]['interactions'] -= average_edge_weight // alpha_weight
            G.edges[edge_to_remove]['weight'] = 1/G.edges[edge_to_remove]['interactions'] if G.edges[edge_to_remove]['interactions'] != 0 else float('inf')
            if G.edges[edge_to_remove]['interactions'] <= 0:
                G.remove_edge(*edge_to_remove)
                n_removed += 1
                strongly_connected_components = list(nx.connected_components(G))
                largest_strongly_connected_component = max(strongly_connected_components, key=len) if strongly_connected_components else set()
                G_largest_strongly_connected = G.subgraph(largest_strongly_connected_component).copy() if largest_strongly_connected_component else nx.Graph()
                robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][n_removed] += G_largest_strongly_connected.number_of_nodes()
                robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][n_removed] += (len(strongly_connected_components))
                APL = nx.average_shortest_path_length(G_largest_strongly_connected, weight='weight') if G_largest_strongly_connected.number_of_nodes() > 0 and G_largest_strongly_connected.number_of_nodes() > 0 else 0
                robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][n_removed] += APL
                avg_deg = sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0
                robusteness_locality_metrics[locality]['average_degree'][n_removed] += avg_deg
                avg_w_deg = sum(dict(G.degree(weight="interactions")).values()) / G.number_of_nodes() if G.number_of_nodes() != 0 else 0
                robusteness_locality_metrics[locality]['average_weighted_degree'][n_removed] += avg_w_deg
                # creating a 'cumulative' degree distributions so at the end we can get average values for the distributions
                deg_dist = dict(Counter(dict(G.degree()).values()))
                w_deg_dist = dict(Counter(dict(G.degree(weight="interactions")).values()))
                if seed == 0: # no distribution saved yet, save as is
                    robusteness_locality_metrics[locality]['degree_distribution'][n_removed] = deg_dist
                    robusteness_locality_metrics[locality]['weighted_degree_distribution'][n_removed] = w_deg_dist
                else: # make calculations
                    deg_dist_cum = robusteness_locality_metrics[locality]['degree_distribution'][n_removed]
                    w_deg_dist_cum = robusteness_locality_metrics[locality]['weighted_degree_distribution'][n_removed]
                    for k in deg_dist.keys():
                        deg_dist_cum[k] = deg_dist[k] + (deg_dist_cum.get(k) or 0)
                    for k in w_deg_dist.keys():
                        w_deg_dist_cum[k] = w_deg_dist[k] + (w_deg_dist_cum.get(k) or 0)
                    robusteness_locality_metrics[locality]['degree_distribution'][n_removed] = deg_dist_cum
                    robusteness_locality_metrics[locality]['weighted_degree_distribution'][n_removed] = w_deg_dist_cum

    # average the results from the multiple iterations
    robusteness_locality_metrics[locality]['number_of_strongly_connected_components'] = [robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['number_of_strongly_connected_components'][1:]]
    robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'] = [robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][0]] + [x / seed_range for x in robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'][1:]]
    robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'] = [robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][0]] + [x / seed_range for x in robusteness_locality_metrics[locality]['largest_strongly_connected_component_average_path_length'][1:]]
    robusteness_locality_metrics[locality]['average_degree'] = [robusteness_locality_metrics[locality]['average_degree'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['average_degree'][1:]]
    robusteness_locality_metrics[locality]['average_weighted_degree'] = [robusteness_locality_metrics[locality]['average_weighted_degree'][0]] +  [x / seed_range for x in robusteness_locality_metrics[locality]['average_weighted_degree'][1:]]

    first = True
    for dist in robusteness_locality_metrics[locality]['degree_distribution']:
        if first: # initial value is not a sampled value, no need to average
            first = False
            continue
        for k in dist.keys():
            dist[k] = dist[k] / seed_range
    
    first = True
    for dist in robusteness_locality_metrics[locality]['weighted_degree_distribution']:
        if first: # initial value is not a sampled value, no need to average
            first = False
            continue
        for k in dist.keys():
            dist[k] = dist[k] / seed_range

for locality in robusteness_locality_metrics:
    print(f"Robustness metrics for {locality}:")
    print("Size of largest component after each removal:", robusteness_locality_metrics[locality]['largest_strongly_connected_component_size'])
    print("Number of strongly connected components after each removal:", robusteness_locality_metrics[locality]['number_of_strongly_connected_components'])
    print("\n")

for locality_key, locality_metrics in robusteness_locality_metrics.items():
    parsed_metrics = {}
    for key, values in locality_metrics.items():
        col_name = f"{key}"
        parsed_metrics[col_name] = values
    df = pd.DataFrame(parsed_metrics)
    df.to_csv(f'{csv_metrics_dir}/edge_removal_random_{locality_key}.csv', index=True, index_label='edges_removed')

print("      End of Robustness Metrics based on edge removal based on decreasing the weight of a random edge\n")

