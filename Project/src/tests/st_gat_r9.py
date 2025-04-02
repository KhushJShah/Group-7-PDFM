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
    def __init__(self, in_channels, out_channels, n_nodes, heads=14, dropout=0.0):
        super().__init__()
        self.n_pred = out_channels
        self.n_nodes = n_nodes
        self.gat_out_channels = 64

        self.gat = GATConv(in_channels, self.gat_out_channels, heads=heads, 
                          dropout=dropout, concat=False, edge_dim=1)
        self.lstm = torch.nn.LSTM(input_size=self.gat_out_channels, 
                                 hidden_size=64, num_layers=6, batch_first=True)
        self.linear = torch.nn.Linear(64, self.n_pred)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        #print(f"\nInput shape: {x.shape}")  # [nodes, time_steps, features]
        
        # GAT processing
        out = []
        for t in range(x.size(1)):
            h = self.gat(x[:, t, :], edge_index, edge_attr)
            out.append(h)
            
            
        x = torch.stack(out, dim=1)
        #print(f"After GAT stacking: {x.shape}")  # [nodes, time_steps, features]
        
        # LSTM processing
        x, _ = self.lstm(x)
        #print(f"After LSTM: {x.shape}")  # [nodes, time_steps, hidden_size]
        x = x[:, -1, :]  # Take last output
        #print(f"After LSTM selection: {x.shape}")  # [nodes, hidden_size]
        
        # Final prediction
        x = self.linear(x)
        #print(f"After Linear: {x.shape}")  # [nodes, pred_steps]
        return x.unsqueeze(-1)  # [nodes, pred_steps, 1]

#%% Training and Evaluation
def plot_predictions(zipcode, train_actual, train_pred, test_actual, test_pred, county, state):
    """Plot training and test predictions in subplots"""
    os.makedirs("gnn_plots", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Training plot
    ax1.plot(train_actual, label='Actual', marker='o')
    ax1.plot(train_pred, label='Predicted', linestyle='--', marker='x')
    ax1.set_title(f'Training Predictions - {county}, {state} ({zipcode})')
    ax1.set_xlabel('Training Time Steps')
    ax1.set_ylabel('Unemployment Rate')
    ax1.legend()
    
    # Test plot
    ax2.plot(test_actual, label='Actual', marker='o')
    ax2.plot(test_pred, label='Predicted', linestyle='--', marker='x')
    ax2.set_title(f'Test Forecast - {county}, {state} ({zipcode})')
    ax2.set_xlabel('Test Time Steps')
    ax2.set_ylabel('Unemployment Rate')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f"gnn_plots/{zipcode}_combined.png", dpi=300)
    plt.close()

def main():
    # Initialize model
    model = ST_GAT(in_channels=1, out_channels=6, n_nodes=n_counties)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(50):
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
    train_preds, test_preds = [], []
    
    with torch.no_grad():
        # Training predictions
        for data in train_data:
            train_preds.append(model(data).numpy())
        
        # Test predictions
        for data in test_data:
            test_preds.append(model(data).numpy())
    
    # Process predictions
    train_preds = np.concatenate(train_preds, axis=0).squeeze(-1)
    test_preds = np.concatenate(test_preds, axis=0).squeeze(-1)
    
    # Get actual values
    train_actuals = np.concatenate([d.y.numpy() for d in train_data], axis=0).squeeze(-1)
    test_actuals = np.concatenate([d.y.numpy() for d in test_data], axis=0).squeeze(-1)

    print("\nShapes for Verification:")
    print(f"Train preds: {train_preds.shape}, Train actuals: {train_actuals.shape}")
    print(f"Test preds: {test_preds.shape}, Test actuals: {test_actuals.shape}")

    # Calculate and save county-wise metrics
    results = []
    zipcodes = df['zipcode'].values
    for idx in range(n_counties):
        zipcode = zipcodes[idx]
        county = df.iloc[idx]['county']
        state = df.iloc[idx]['state']
        
        # Corrected indexing for 2D arrays
        train_actual = train_actuals[idx, :].flatten()  # Changed from [:, idx, :]
        train_pred = train_preds[idx, :].flatten()      # Changed from [:, idx, :]
        test_actual = test_actuals[idx, :].flatten()    # Changed from [:, idx, :]
        test_pred = test_preds[idx, :].flatten()        # Changed from [:, idx, :]
        
        # Calculate metrics
        train_mae = mean_absolute_error(train_actual, train_pred)
        train_rmse = np.sqrt(mean_squared_error(train_actual, train_pred))
        test_mae = mean_absolute_error(test_actual, test_pred)
        test_rmse = np.sqrt(mean_squared_error(test_actual, test_pred))
        
        # Store results
        results.append({
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'train_MAE': round(train_mae, 4),
            'train_RMSE': round(train_rmse, 4),
            'test_MAE': round(test_mae, 4),
            'test_RMSE': round(test_rmse, 4)
        })
        
        # Generate combined plot
        plot_predictions(zipcode, train_actual, train_pred, 
                        test_actual, test_pred, county, state)
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('gnn_county_metrics_full.csv', index=False)
    print("\nMetrics saved to gnn_county_metrics_full.csv")

if __name__ == "__main__":
    main()

# %%
