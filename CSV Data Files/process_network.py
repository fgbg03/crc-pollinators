import pandas as pd

df = pd.read_csv('Interaction_data.csv', encoding='ISO-8859-1')

df = df[df['Plant_rank'] == 'SPECIES']
df = df[df['Pollinator_rank'] == 'SPECIES']

df['Pollinator_accepted_name'] = df['Pollinator_accepted_name']+' ('+df['Locality']+')'
df['Plant_accepted_name'] = df['Plant_accepted_name']+' ('+df['Locality']+')'
edges = (
    df.groupby(['Pollinator_accepted_name', 'Plant_accepted_name'])
    .size()
    .reset_index(name='Weight')
)
edges1 = edges.copy()
edges2 = edges.copy()
# Rename for Gephi
edges1 = edges1.rename(columns={
    'Pollinator_accepted_name': 'Source',
    'Plant_accepted_name': 'Target'
})

#check number of unique localities
unique_localities = df['Locality'].nunique()
print(f"Number of unique localities: {unique_localities}")

edges2 = edges2.rename(columns={
    'Pollinator_accepted_name': 'Target',
    'Plant_accepted_name': 'Source'
})

edges = pd.concat([edges1, edges2], ignore_index=True)


# edges = df[['Plant_accepted_name','Pollinator_accepted_name','Country','Bioregion','Latitude','Longitude','Locality','Date']].copy()
# edges.drop_duplicates(inplace=True)
# edges.rename(columns={'Plant_accepted_name': 'Target', 'Pollinator_accepted_name': 'Source'}, inplace=True)

print("Saving edges to CSV")
edges.to_csv('Edges_data.csv', index=False)

plants = df[['Plant_accepted_name', 'Plant_order', 'Plant_family', 'Plant_genus', 'Country', 'Locality', 'Bioregion']].copy()
plants.drop_duplicates(inplace=True)

plants.rename(columns={'Plant_accepted_name': 'Id', 'Plant_order': 'Order', 'Plant_family': 'Family', 'Plant_genus': 'Genus'}, inplace=True)
plants['Type'] = 'Plant'

pollinators = df[['Pollinator_accepted_name', 'Pollinator_order', 'Pollinator_family', 'Pollinator_genus', 'Country', 'Locality', 'Bioregion']].copy()
pollinators.drop_duplicates(inplace=True)

pollinators.rename(columns={'Pollinator_accepted_name': 'Id', 'Pollinator_order': 'Order', 'Pollinator_family': 'Family', 'Pollinator_genus': 'Genus'}, inplace=True)
pollinators['Type'] = 'Pollinator'

nodes = pd.concat([plants, pollinators], ignore_index=True)
print("Saving nodes to CSV")
nodes.to_csv('Nodes_data.csv', index=False)

print("Nodes and edges dataframes created in CSV files.")