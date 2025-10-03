import pandas as pd


def do_taxonomy(plant, pollinator, level, df):
    df['Pollinator_accepted_name'] = df[pollinator]+' ('+df['Locality']+')'
    df['Plant_accepted_name'] = df[plant]+' ('+df['Locality']+')'
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
    edges.to_csv('Edges_data_'+level+'.csv', index=False)

    plants = df[['Plant_accepted_name', 'Country', 'Locality', 'Bioregion']].copy()
    plants.drop_duplicates(inplace=True)

    plants.rename(columns={'Plant_accepted_name': 'Id'}, inplace=True)
    plants['Type'] = 'Plant'

    pollinators = df[['Pollinator_accepted_name', 'Country', 'Locality', 'Bioregion']].copy()
    pollinators.drop_duplicates(inplace=True)

    pollinators.rename(columns={'Pollinator_accepted_name': 'Id'}, inplace=True)
    pollinators['Type'] = 'Pollinator'

    nodes = pd.concat([plants, pollinators], ignore_index=True)
    print("Saving nodes to CSV"+level)
    nodes.to_csv('Nodes_data_'+level+'.csv', index=False)

    print("Nodes and edges dataframes created in CSV files "+level+".")


df = pd.read_csv('Interaction_data.csv', encoding='ISO-8859-1')

df = df[df['Plant_rank'] == 'SPECIES']
df = df[df['Pollinator_rank'] == 'SPECIES']

do_taxonomy('Plant_genus', 'Pollinator_genus', 'genus_level', df)
do_taxonomy('Plant_family', 'Pollinator_family', 'family_level', df)
do_taxonomy('Plant_order', 'Pollinator_order', 'order_level', df)
do_taxonomy('Plant_accepted_name', 'Pollinator_accepted_name', 'species_level', df)