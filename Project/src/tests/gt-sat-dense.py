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
        self.gat = GATConv(1, 64, heads=10, edge_dim=1, add_self_loops=False)  # Changed
        
        # Update LSTM input_size to match GAT's output dimension (64 * heads)
        self.lstm = nn.LSTM(64 * 10, 64, num_layers=4, batch_first=True)  # Changed input_size
        
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
print("\nCreating training dataset...")
train_dataset = Data(
    x=torch.tensor(train_data.T, dtype=torch.float).unsqueeze(-1).permute(1, 0, 2),  # [90, 406, 1]
    edge_index=edge_index,  # Built for 90 nodes
    edge_attr=edge_attr,
    y=torch.tensor(test_data.T, dtype=torch.float).unsqueeze(-1).permute(1, 0, 2)     # [90, 6, 1]
)

print("\nTraining dataset shapes:")
print(f"Input x shape: {train_dataset.x.shape}")
print(f"Target y shape: {train_dataset.y.shape}")
print(f"Edge index shape: {train_dataset.edge_index.shape}")
print(f"Edge attr shape: {train_dataset.edge_attr.shape}")

test_dataset = Data(
    x=torch.tensor(test_data.T, dtype=torch.float).unsqueeze(-1).permute(1, 0, 2),    # [90, 6, 1]
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=torch.tensor(test_data.T, dtype=torch.float).unsqueeze(-1).permute(1, 0, 2)
)

print("\nTesting dataset shapes:")
print(f"Test input x shape: {test_dataset.x.shape}")
print(f"Test Target y shape: {test_dataset.y.shape}")
print(f"Test Edge index shape: {test_dataset.edge_index.shape}")
print(f"Test Edge attr shape: {test_dataset.edge_attr.shape}")
#%% 7. Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullHistoryGAT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

#%% 8. Training Loop
model.train()
train_dataset = train_dataset.to(device)

for epoch in range(50):
    optimizer.zero_grad()
    pred = model(train_dataset)
    loss = criterion(pred, train_dataset.y)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

#%% 9. Evaluation
def evaluate_and_save_results(model, train_dataset, test_dataset, df):
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Move datasets to device
    train_dataset = train_dataset.to(device)
    test_dataset = test_dataset.to(device)
    
    # Evaluate on both train and test sets
    model.eval()
    
    # Get predictions for datasets
    with torch.no_grad():
        train_pred = model(train_dataset).cpu().numpy().squeeze()  # [90, 6]
        train_true = train_dataset.y.cpu().numpy().squeeze()
        
        test_pred = model(test_dataset).cpu().numpy().squeeze()    # [90, 6]
        test_true = test_dataset.y.cpu().numpy().squeeze()

    # Calculate metrics per county
    metrics = []
    for i in range(90):
        county_name = df.iloc[i]['county']
        
        # Train metrics
        train_mae = mean_absolute_error(train_true[i], train_pred[i])
        train_rmse = np.sqrt(mean_squared_error(train_true[i], train_pred[i]))
        
        # Test metrics
        test_mae = mean_absolute_error(test_true[i], test_pred[i])
        test_rmse = np.sqrt(mean_squared_error(test_true[i], test_pred[i]))
        
        metrics.append({
            'county': county_name,
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'test_mae': test_mae,
            'test_rmse': test_rmse
        })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv("dense_results/county_metrics.csv", index=False)
    print("Metrics saved to results/county_metrics.csv")

    
    # Generate plots for each county
    os.makedirs("dense_results/plots", exist_ok=True)
    for i in range(90):
        county_name = df.iloc[i]['county']
        
        plt.figure(figsize=(15, 6))
        
        # Training plot
        plt.subplot(1, 2, 1)
        plt.plot(train_true[i], label='Actual', color='blue', alpha=0.7)
        plt.plot(train_pred[i], label='Predicted', color='orange', alpha=0.7)
        plt.title(f"{county_name}\nTraining Predictions")
        plt.xlabel("Months (1-406)")
        plt.ylabel("Unemployment Rate")
        
        # Testing plot
        plt.subplot(1, 2, 2)
        plt.plot(test_true[i], label='Actual', color='blue', alpha=0.7)
        plt.plot(test_pred[i], label='Predicted', color='orange', alpha=0.7)
        plt.title(f"{county_name}\nTest Predictions")
        plt.xlabel("Months (407-412)")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"dense_results/plots/{county_name.replace(' ', '_')}.png")
        plt.close()
    
    print("Plots saved to results/plots directory")
    
    # Calculate global averages
    global_metrics = {
        'train_mae': metrics_df['train_mae'].mean(),
        'train_rmse': metrics_df['train_rmse'].mean(),
        'test_mae': metrics_df['test_mae'].mean(),
        'test_rmse': metrics_df['test_rmse'].mean()
    }
    
    print("\nGlobal Average Metrics:")
    print(f"Train MAE: {global_metrics['train_mae']:.4f}")
    print(f"Train RMSE: {global_metrics['train_rmse']:.4f}")
    print(f"Test MAE: {global_metrics['test_mae']:.4f}")
    print(f"Test RMSE: {global_metrics['test_rmse']:.4f}")

# Usage after training:
evaluate_and_save_results(model, train_dataset, test_dataset, df)


# %%
