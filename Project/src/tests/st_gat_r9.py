#%% Final GNN Code with County-wise Metrics
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.data import Data
import os

#%% Data Preparation
def create_temporal_split(data, n_past=12, n_future=6, test_months=6):
    """Create temporal split with last 'test_months' as test set"""
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    split_idx = data.shape[1] - test_months - n_future
    
    # Training sequences
    for i in range(split_idx - n_past + 1):
        X_train.append(data[:, i:i+n_past, :])
        y_train.append(data[:, i+n_past:i+n_past+n_future, :])
    
    # Test sequences
    test_start = data.shape[1] - test_months - n_past
    X_test.append(data[:, test_start:test_start+n_past, :])
    y_test.append(data[:, -test_months:, :])
    
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# Load data
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv')
time_series_data = df.filter(regex='^(19|20)').values
n_counties = len(df)
n_time_steps = time_series_data.shape[1]

# Reshape without normalization
time_series_data = time_series_data.reshape(n_counties, n_time_steps, 1)

# Create temporal split
X_train, y_train, X_test, y_test = create_temporal_split(time_series_data)

# Prepare graph data
adj_matrix = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/adjacency_matrix_with_weights_r9.csv', index_col=0).values
edge_index = []
edge_attr = []

for i in range(len(adj_matrix)):
    for j in range(i + 1, len(adj_matrix)):
        if adj_matrix[i, j] != 0:
            edge_index.append([i, j])
            edge_attr.append([adj_matrix[i, j]])

edge_index = torch.tensor(edge_index).T.contiguous()
edge_attr = torch.tensor(edge_attr).float()

# Create Data objects
train_data = [Data(x=torch.tensor(x, dtype=torch.float), 
                   edge_index=edge_index, 
                   edge_attr=edge_attr,
                   y=torch.tensor(y, dtype=torch.float)) 
              for x, y in zip(X_train, y_train)]

test_data = [Data(x=torch.tensor(x, dtype=torch.float),
                  edge_index=edge_index,
                  edge_attr=edge_attr,
                  y=torch.tensor(y, dtype=torch.float))
             for x, y in zip(X_test, y_test)]

#%% Model Architecture
class ST_GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, n_nodes, heads=12, dropout=0.0):
        super().__init__()
        self.n_pred = out_channels
        self.n_nodes = n_nodes
        self.gat_out_channels = 64

        self.gat = GATConv(in_channels, self.gat_out_channels, heads=heads, 
                          dropout=dropout, concat=False, edge_dim=1)
        self.lstm = torch.nn.LSTM(input_size=self.gat_out_channels, 
                                 hidden_size=64, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(64, self.n_pred)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # GAT processing
        out = []
        for t in range(x.size(1)):
            h = self.gat(x[:, t, :], edge_index, edge_attr)
            out.append(h)
        x = torch.stack(out, dim=1)
        
        # LSTM processing
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        # Final prediction
        x = self.linear(x)
        return x.unsqueeze(-1)

#%% Training and Evaluation
def plot_predictions(zipcode, actual, pred, county, state):
    """Plot predictions vs actual values"""
    os.makedirs("gnn_plots", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(pred, label='Predicted', linestyle='--', marker='x')
    plt.title(f'GNN Predictions - {county}, {state} ({zipcode})')
    plt.xlabel('Month')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.savefig(f"gnn_plots/{zipcode}_predictions.png", dpi=300)
    plt.close()

def main():
    # Initialize model
    model = ST_GAT(in_channels=1, out_channels=6, n_nodes=n_counties)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(30):
        total_loss = 0
        for data in train_data:
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_data):.4f}')
    
    # Evaluation
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in test_data:
            pred = model(data)
            predictions.append(pred.numpy())
    
    # Process predictions
    predictions = np.concatenate(predictions, axis=0).squeeze(-1)
    actuals = np.concatenate([d.y.numpy() for d in test_data], axis=0).squeeze(-1)
    
    # Calculate and save county-wise metrics
    results = []
    zipcodes = df['zipcode'].values
    for idx in range(n_counties):
        zipcode = zipcodes[idx]
        county = df.iloc[idx]['county']
        state = df.iloc[idx]['state']
        
        actual_ts = time_series_data[idx, -6:, 0]
        pred_ts = predictions[0, idx, :]
        
        # Calculate metrics
        mae = mean_absolute_error(actual_ts, pred_ts)
        rmse = np.sqrt(mean_squared_error(actual_ts, pred_ts))
        
        # Store results
        results.append({
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4)
        })
        
        # Generate plot
        plot_predictions(zipcode, actual_ts, pred_ts, county, state)
    
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('gnn_county_metrics.csv', index=False)
    print("\nMetrics saved to gnn_county_metrics.csv")

if __name__ == "__main__":
    main()

# %%
