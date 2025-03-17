'''
In this file, we will connect the pincodes as a graph/mesh structure so that we have a network on which we can run the GNNs.
'''

#%%
'''Importing the libraries'''
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox
import math

#%%
'''Loading the dataset'''
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv')
print(df.head())

#%%
G = nx.Graph()

# Download the road network for the area covering all counties
# Use a bounding box that covers all latitude/longitude points in your dataset
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    return distance

# Add counties as nodes to the graph
for index, row in df.iterrows():
    G.add_node(row['county'], pos=(row['longitude'], row['latitude']))

# Add edges based on distance between counties
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if i < j:  # Avoid duplicate edges and self-loops
            distance = haversine_distance(row1['latitude'], row1['longitude'], row2['latitude'], row2['longitude'])
            G.add_edge(row1['county'], row2['county'], weight=distance)

# Get node positions for visualization
pos = nx.get_node_attributes(G, 'pos')

#%%
pos = nx.get_node_attributes(G, 'pos')  # Get positions (latitude, longitude) for visualization
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('County Connectivity Graph with Distances')
plt.show()
# %%
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=False, node_size=50, node_color='blue', font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos)
plt.title('County Connectivity Graph with Distances')
plt.show()
# %%
adj_matrix = nx.to_numpy_array(G, weight='weight')

# Get the list of node names (counties) in the order they appear in the adjacency matrix
node_names = list(G.nodes())

# Define the file path
file_path = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\adjacency_matrix_with_weights_r9.txt'

# Save the adjacency matrix with weights and node names
with open(file_path, 'w') as f:
    # Write node names as header
    f.write('\t'.join(node_names) + '\n')
    
    # Write the adjacency matrix with weights
    for i, row in enumerate(adj_matrix):
        f.write(f"{node_names[i]}\t")
        f.write('\t'.join(map(str, row)) + '\n')
# %%
number_of_links = G.number_of_edges()
print(f"The number of links in the graph is: {number_of_links}")
# %%
adj_matrix = nx.to_numpy_array(G, weight='weight')
node_names = list(G.nodes())

adj_df = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)

csv_file_path = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\adjacency_matrix_with_weights_r9.csv'
adj_df.to_csv(csv_file_path)
# %%
