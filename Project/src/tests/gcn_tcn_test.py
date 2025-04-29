'''
This script loads a trained GCN-TCN model and evaluates its forecasting performance on the test set (last 6 months) for each county-state pair.
'''

#%% Loading libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

#%% Configuration
RESULTS_DIR = r'Project\graphs\gcn_tcn_plots'
MODEL_PATH = r'Project\src\component\deep learning models\gcn_tcn_model.pth'
PLOT_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Hyperparameters
WINDOW_SIZE = 39
GCN_HIDDEN = 32
TCN_CHANNELS = [64, 64]
PRED_HORIZON = 6

#%% Data Paths
DATA_PATH = r'Project\data\merged_data_unemployment_r9.csv'
ADJ_PATH = r'Project\data\adjacency_matrix_with_weights_r9.csv'

#%% Load data
df = pd.read_csv(DATA_PATH)
county_names = df['county'].values
state_names = df['state'].values if 'state' in df.columns else [""] * len(county_names)
time_columns = [col for col in df.columns if col[:4].isdigit()]
full_data = df[time_columns].values.astype(np.float32)
full_data = full_data[:, :, np.newaxis]  # Add feature dimension

#%% Load adjacency matrix
adj_df = pd.read_csv(ADJ_PATH, index_col=0)
adj_matrix = adj_df.values
edge_index = torch.tensor(np.stack(np.where(adj_matrix > 0)), dtype=torch.long).to(DEVICE)

#%% Model Architecture
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.net = nn.Sequential(
            self.conv1, nn.ReLU(),
            self.conv2, nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        out = out[:, :, :x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class GCN_TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(1, GCN_HIDDEN)
        self.tcn = nn.Sequential(
            TemporalBlock(GCN_HIDDEN, TCN_CHANNELS[0], kernel_size=3, dilation=1),
            TemporalBlock(TCN_CHANNELS[0], TCN_CHANNELS[1], kernel_size=3, dilation=2)
        )
        self.linear = nn.Linear(TCN_CHANNELS[-1], 1)
        
    def forward(self, x, edge_index):
        batch_size, num_counties, seq_len, _ = x.size()
        
        # GCN processing
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, :, t, :].reshape(-1, 1)
            out = F.relu(self.gcn(x_t, edge_index))
            gcn_outputs.append(out.view(batch_size, num_counties, -1))
        
        # TCN processing
        x = torch.stack(gcn_outputs, dim=2)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(batch_size * num_counties, GCN_HIDDEN, seq_len)
        x = self.tcn(x)
        x = x[..., -1]
        x = x.view(batch_size, num_counties, -1)
        return self.linear(x)

#%% Recursive Forecasting Function
def recursive_forecast(model, initial_data, edge_index, steps=6):
    """initial_data: [counties, window, 1]"""
    current_window = initial_data.unsqueeze(0).to(DEVICE)  # [1, counties, window, 1]
    predictions = torch.zeros(initial_data.size(0), steps).to(DEVICE)
    
    for step in range(steps):
        with torch.no_grad():
            pred = model(current_window, edge_index)
            pred_reshaped = pred.unsqueeze(-1)  # [1, counties, 1, 1]
            new_window = torch.cat([
                current_window[:, :, 1:, :],  # [1, counties, window-1, 1]
                pred_reshaped                 # [1, counties, 1, 1]
            ], dim=2)
        current_window = new_window
        predictions[:, step] = pred.squeeze(0).squeeze(-1)
    return predictions.cpu().numpy()  # [counties, steps]

#%% Prepare test window and ground truth
test_window = full_data[:, -PRED_HORIZON-WINDOW_SIZE:-PRED_HORIZON, :]
test_truth = full_data[:, -PRED_HORIZON:, 0]

#%% Load model
model = GCN_TCN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

#%% Generate predictions
predictions = recursive_forecast(model, torch.tensor(test_window, dtype=torch.float32), edge_index, steps=PRED_HORIZON)

#%% Plotting
for i, (county, state) in enumerate(zip(county_names, state_names)):
    plt.figure(figsize=(10, 5))
    plt.plot(test_truth[i], label='True Values', marker='o', color='blue')
    plt.plot(predictions[i], label='Predictions', marker='x', color='orange', linestyle='--')
    plt.title(f'GCN-TCN Forecast vs Actual - {county}, {state}')
    plt.xlabel('Months')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid(True)
    fname = f"{state.replace(' ', '_')}_{county.replace(' ', '_')}_gcn_tcn_forecast.png"
    plt.savefig(os.path.join(PLOT_DIR, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for {county}, {state}")

print("All plots saved in:", PLOT_DIR)
