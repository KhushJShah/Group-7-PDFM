#%% Fixed Implementation
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os

#%% 1. File Path Fixes (Update these paths)
DATA_PATH = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/'
UNEMPLOYMENT_FILE = 'merged_data_unemployment_r9.csv'  # Verify exact filename
ADJACENCY_FILE = 'adjacency_matrix_with_weights_r9.csv'  # Verify exact filename

#%% 2. Data Loading with Error Handling
try:
    df = pd.read_csv(DATA_PATH + UNEMPLOYMENT_FILE)
    adj_matrix = pd.read_csv(DATA_PATH + ADJACENCY_FILE, index_col=0).values
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please verify:")
    print(f"1. File exists: {DATA_PATH + UNEMPLOYMENT_FILE}")
    print(f"2. File exists: {DATA_PATH + ADJACENCY_FILE}")
    raise

#%% 3. Data Preparation
# Extract time series (90 counties × 412 months)
time_series = df.filter(regex='^(19|20)').values.astype(np.float32)
n_counties, n_months = time_series.shape

# Split into train (406 months) and test (6 months)
train_data = time_series[:, :-6]  # (90, 406)
test_data = time_series[:, -6:]   # (90, 6)

#%% 4. Graph Construction
edge_index = []
edge_attr = []
for i in range(len(adj_matrix)):
    for j in range(i + 1, len(adj_matrix)):
        if adj_matrix[i, j] > 0:
            edge_index.append([i, j])
            edge_attr.append([adj_matrix[i, j]])

edge_index = torch.tensor(edge_index).T.contiguous().long()
edge_attr = torch.tensor(edge_attr).float()


#%% 5. Model Architecture
class FullHistoryGAT(nn.Module):
    def __init__(self):
        super().__init__()
        # Remove concat=False (now defaults to True)
        self.gat = GATConv(1, 64, heads=8, edge_dim=1, add_self_loops=False)  # Changed
        
        # Update LSTM input_size to match GAT's output dimension (64 * heads)
        self.lstm = nn.LSTM(64 * 8, 64, num_layers=4, batch_first=True)  # Changed input_size
        
        self.linear = nn.Linear(64, 6)

    def forward(self, data):
        x = data.x
        seq_len = x.size(1)
        
        gat_outputs = []
        for t in range(seq_len):
            out = self.gat(x[:, t, :], data.edge_index, data.edge_attr)
            #print(f"GAT output shape: {out.shape}")  # Now [90, 640] (64*10 heads)
            gat_outputs.append(out.unsqueeze(1))  # [90, 1, 640]
        
        x = torch.cat(gat_outputs, dim=1)  # [90, seq_len, 640]
        #print(f"LSTM input shape: {x.shape}")  # Verify new dimension
        
        x, _ = self.lstm(x)  # Now accepts [90, 406, 640] -> [90, 406, 128]
        return self.linear(x[:, -1, :]).unsqueeze(-1)  # [90, 6, 1]




#%%
# Modified Data creation with print statements
full_dataset = Data(
    x=torch.tensor(time_series.T, dtype=torch.float).unsqueeze(-1),  # [412, 90, 1]
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=torch.tensor(time_series.T, dtype=torch.float).unsqueeze(-1)   # [412, 90, 1]
)

# Split into train/test datasets
train_dataset = Data(
    x=full_dataset.x[:406],  # First 406 months
    edge_index=full_dataset.edge_index,
    edge_attr=full_dataset.edge_attr,
    y=full_dataset.y[:406]
)

test_dataset = Data(
    x=full_dataset.x[-6:],   # Last 6 months
    edge_index=full_dataset.edge_index,
    edge_attr=full_dataset.edge_attr,
    y=full_dataset.y[-6:]
)
from torch_geometric.loader import DataLoader
train_loader = DataLoader([train_dataset], batch_size=1)  # Single graph batch
test_loader = DataLoader([test_dataset], batch_size=1)
#%% 7. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullHistoryGAT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

#%% 8. Training Loop
for epoch in range(50):
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        pred = model(batch)        # [406, 90]
        true = batch.y.squeeze(-1) # [406, 90]
        
        loss = criterion(pred, true)
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
#%% 9. Evaluation
def evaluate_and_save(model, full_dataset, df):
    model.eval()
    with torch.no_grad():
        # Get all predictions at once
        all_pred = model(full_dataset).cpu().numpy().squeeze()  # [412, 90]
        all_true = full_dataset.y.cpu().numpy().squeeze()       # [412, 90]

    # Calculate metrics
    metrics = []
    for i in range(n_counties):
        county_name = df.iloc[i]['county']
        
        # Training period metrics (first 406 months)
        train_rmse = np.sqrt(mean_squared_error(
            all_true[full_dataset.train_mask, i], 
            all_pred[full_dataset.train_mask, i]
        ))
        
        # Test period metrics (last 6 months)
        test_rmse = np.sqrt(mean_squared_error(
            all_true[full_dataset.test_mask, i],
            all_pred[full_dataset.test_mask, i]
        ))

        metrics.append({
            'county': county_name,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse
        })

    # Save results (keep your original code)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('dense_gnn/results.csv', index=False)

    # Updated plotting with train/test separation
    os.makedirs('dense_gnn/plots', exist_ok=True)
    for i in range(n_counties):
        plt.figure(figsize=(12, 6))
        
        # Plot full timeline
        plt.plot(all_true[:, i], label='Actual', color='blue')
        plt.plot(all_pred[:, i], label='Predicted', color='orange', linestyle='--')
        
        # Add vertical split line
        plt.axvline(x=406, color='red', linestyle=':', label='Train/Test Split')
        
        plt.title(f"{df.iloc[i]['county']} Unemployment Forecast")
        plt.xlabel("Months")
        plt.ylabel("Normalized Rate")
        plt.legend()
        plt.savefig(f'dense_gnn/plots/{df.iloc[i]["county"].replace(" ", "_")}_full.png')
        plt.close()

# Update the evaluation call
print("\nEvaluating model...")
evaluate_and_save(model, full_dataset, df)  # Pass single dataset


# %%
