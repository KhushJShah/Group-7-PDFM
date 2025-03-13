'''In this file, we will explore the GT-GAT model with LSTM'''

#%%
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data, DataLoader

#%%
# Load your dataset
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment.csv')

#%%
# Extract time series data
time_series_data = df.filter(regex='^(19|20)').values
time_series_data = (time_series_data - np.mean(time_series_data, axis=0)) / np.std(time_series_data, axis=0)
#%%
print(time_series_data)
#%%
# Extract node features (population density)
n_counties = 58  # Number of counties
n_time_steps = time_series_data.shape[1]  # Total time steps in the dataset

# Reshape to (counties, time_steps, features)
time_series_data = time_series_data.reshape(n_counties, n_time_steps, 1)

print(f"Reshaped time series data shape: {time_series_data.shape}")

# Extract node features (e.g., population density)
node_features = df['population_density_log10'].values.reshape(-1, 1)

print(f"Node features shape: {node_features.shape}")
# Load adjacency matrix
adj_matrix = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/adjacency_matrix_with_weights.csv', index_col=0).values

# Prepare graph data
edge_index = []
edge_attr = []

for i in range(len(adj_matrix)):
    for j in range(i + 1, len(adj_matrix)):
        if adj_matrix[i, j] != 0:  # Exclude self-loops and zero-weight edges
            edge_index.append([i, j])
            edge_attr.append([adj_matrix[i, j]])

# Convert to numpy arrays

edge_attr = torch.tensor(edge_attr)
edge_attr = edge_attr.float()
edge_index = torch.tensor(edge_index).T.contiguous()
print("edge_attr type:", edge_attr.dtype)
print("edge_index type:", edge_index.dtype)
# Count the number of edges
num_edges = edge_index.shape[1]
print("Number of edges:", num_edges)

edge_attr = (edge_attr - edge_attr.mean()) / edge_attr.std()


#%%
# Prepare data for LSTM
def create_sequences(data, n_past=12, n_future=3):
    X, y = [], []
    for i in range(data.shape[1] - n_past - n_future + 1):
        X.append(data[:, i:i+n_past, :])
        y.append(data[:, i+n_past:i+n_past+n_future, :])
    return np.array(X), np.array(y)

X, y = create_sequences(time_series_data)

# Create PyTorch Geometric Data objects
data_list = [Data(x=torch.tensor(x, dtype=torch.float),
                  edge_index=edge_index,
                  edge_attr=edge_attr,
                  y=torch.tensor(y[i], dtype=torch.float))
             for i, x in enumerate(X)]

# Split data into train and test sets
train_data = data_list[:-12]  # Use last year for testing
test_data = data_list[-12:]

print(f"Number of data points: {len(data_list)}")
print(f"Shape of x in first data point: {data_list[0].x.shape}")
print(f"Shape of y in first data point: {data_list[0].y.shape}")

#%%
print(type(train_data[0].edge_index))
print(type(train_data[0].x))
print(type(train_data[0].edge_attr))

print(train_data[0].edge_index.shape)
print(train_data[0].x.shape)
print(train_data[0].edge_attr.shape)


#%%
print(f"Edges: {train_data[0].edge_index.shape[1]}, Nodes: {train_data[0].x.size(0)}")

#%%

edge_index = torch.tensor(train_data[0].edge_index).T.contiguous()
print(train_data[0].edge_index.shape)

#%%

#%%
# Define the ST-GAT model
class ST_GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, heads=8, dropout=0.0):
        super(ST_GAT, self).__init__()
        self.n_pred = out_channels
        self.n_nodes = n_nodes
        self.gat_out_channels = 64

        self.gat = GATConv(in_channels, self.gat_out_channels, heads=heads, dropout=dropout, concat=False, edge_dim=1)
        self.lstm = torch.nn.LSTM(input_size=self.gat_out_channels, hidden_size=64, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(64, self.n_pred)

    def forward(self, data):
        
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Process each time step through GAT
        out = []
        for t in range(x.size(1)):
            h = self.gat(x[:, t, :], edge_index, edge_attr)
            out.append(h)
        
        x = torch.stack(out, dim=1)  # [n_nodes, time_steps, gat_out_channels]
        
        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last output
        
        # Linear layer
        x = self.linear(x)
        
        return x.unsqueeze(-1) 






#%%
# Initialize model, loss function, and optimizer
model = ST_GAT(in_channels=1, out_channels=3, n_nodes=58)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 20
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for data in train_data:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data)}')

#%%
# Evaluation
model.eval()
predictions = []
true_values = []

with torch.no_grad():
    for data in test_data:
        out = model(data)
        predictions.append(out.numpy())
        true_values.append(data.y.numpy())

predictions = np.concatenate(predictions)
true_values = np.concatenate(true_values)

# Reshape to 2D arrays
predictions = predictions.reshape(-1, predictions.shape[-1])
true_values = true_values.reshape(-1, true_values.shape[-1])

# Calculate MAE and RMSE
mae = mean_absolute_error(true_values, predictions)
rmse = np.sqrt(mean_squared_error(true_values, predictions))

print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# %%
