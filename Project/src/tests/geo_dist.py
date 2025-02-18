'''
In this file, we will connect the pincodes as a graph/mesh structure so that we have a network on which we can run the GNNs.
'''

#%%
'''Importing the libraries'''
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import osmnx as ox

#%%
'''Loading the dataset'''
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment.csv')
print(df.head())

#%%
G = nx.Graph()

# Download the road network for the area covering all counties
# Use a bounding box that covers all latitude/longitude points in your dataset
north = df['latitude'].max()
south = df['latitude'].min()
east = df['longitude'].max()
west = df['longitude'].min()
bbox = (north, south, east, west)

# Download the road network for the area covering all counties
road_network = ox.graph_from_bbox(bbox, network_type='drive')

# Add counties as nodes to the graph
for index, row in df.iterrows():
    # Find the nearest node in the road network for each county's latitude/longitude
    nearest_node = ox.distance.nearest_nodes(road_network, row['longitude'], row['latitude'])
    G.add_node(row['county'], osmnx_node=nearest_node, pos=(row['longitude'], row['latitude']))
#%%
# Add edges based on route availability between counties
for i, row1 in df.iterrows():
    for j, row2 in df.iterrows():
        if i < j:  # Avoid duplicate edges and self-loops
            node1 = G.nodes[row1['county']]['osmnx_node']
            node2 = G.nodes[row2['county']]['osmnx_node']
            
            # Check if there's a route between the two nodes in the road network
            try:
                route = nx.shortest_path(road_network, node1, node2, weight='length')
                distance = nx.shortest_path_length(road_network, node1, node2, weight='length')
                G.add_edge(row1['county'], row2['county'], weight=distance)
            except nx.NetworkXNoPath:
                # No path exists between these two counties
                pass

#%%
pos = nx.get_node_attributes(G, 'pos')  # Get positions (latitude, longitude) for visualization
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=50, node_color='blue', font_size=8)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('County Connectivity Graph with Distances')
plt.show()
# %%
