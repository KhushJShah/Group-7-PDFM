#%% 1. Data Loading and Graph Construction
import pandas as pd
import networkx as nx
import math
import os

# Load data with error handling
try:
    df = pd.read_csv(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv')
except FileNotFoundError:
    print("Error: File not found. Check the path:")
    print(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv')
    exit()

# Create unique county identifiers
df['county_state'] = df['county'] + ', ' + df['state']

# Initialize graph
G = nx.Graph()

# Haversine distance calculation
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# Add nodes with metadata
for _, row in df.iterrows():
    G.add_node(
        row['county_state'],
        pos=(row['longitude'], row['latitude']),
        state=row['state'],
        county=row['county'],
        latitude=row['latitude'],
        longitude=row['longitude']
    )

# Add edges with distance weights
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if i < j:
            distance = haversine_distance(
                row1['latitude'], row1['longitude'],
                row2['latitude'], row2['longitude']
            )
            G.add_edge(row1['county_state'], row2['county_state'], weight=distance)



#%% 3. Save Adjacency Matrix
adj_matrix = nx.to_numpy_array(G, weight='weight')
node_names = list(G.nodes())
adj_df = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
adj_df.to_csv(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\adjacency_matrix_with_weights_r9.csv')

print("Process completed successfully!")
print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")

# %%
