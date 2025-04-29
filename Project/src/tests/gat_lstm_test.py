"""
This script evaluates a trained GAT-LSTM model on the test set (last 6 months) for unemployment rate forecasting.
It generates county-state specific plots comparing model predictions against actual values and saves evaluation metrics.
Results are organized in dedicated directories for reproducibility and analysis.
"""
#%% Loading libraries
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from torch_geometric.nn import GATConv

#%%
class Config:
    data_path = r'Project\data\merged_data_unemployment_r9.csv'
    adj_path = r'Project\data\adjacency_matrix_with_weights_r9.csv'
    results_dir = r'Project\src\component\deep learning models\lstm_models'
    model_file = os.path.join(results_dir, 'best_model.pth')
    plot_dir = r'Project\graphs\gat_lstm_plots'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_size = 406
    test_size = 6
    gat_hidden = 64
    gat_heads = 8
    lstm_hidden = 128
    lstm_layers = 3
    pred_horizon = 6

os.makedirs(Config.plot_dir, exist_ok=True)

# --- Model Definition (must match training) ---
class SpatioTemporalForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATConv(1, Config.gat_hidden, heads=Config.gat_heads, edge_dim=1)
        self.lstm = nn.LSTM(
            Config.gat_hidden * Config.gat_heads, 
            Config.lstm_hidden, 
            num_layers=Config.lstm_layers, 
            batch_first=True
        )
        self.regressor = nn.Linear(Config.lstm_hidden, Config.pred_horizon)

    def forward(self, x, edge_index, edge_attr):
        batch_size, seq_len, num_counties, _ = x.size()
        gat_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :].reshape(-1, 1)  # [batch_size*num_counties, 1]
            h = self.gat(x_t, edge_index, edge_attr)  # [batch_size*num_counties, gat_hidden*heads]
            h = h.view(batch_size, num_counties, -1)
            gat_outputs.append(h)
        lstm_input = torch.stack(gat_outputs, dim=1)  # [batch_size, seq_len, num_counties, features]
        # Reshape for LSTM: [batch_size*num_counties, seq_len, features]
        lstm_input = lstm_input.permute(0, 2, 1, 3).reshape(batch_size*num_counties, seq_len, -1)
        lstm_out, _ = self.lstm(lstm_input)
        out = self.regressor(lstm_out[:, -1, :])  # [batch_size*num_counties, pred_horizon]
        out = out.view(batch_size, num_counties, -1)  # [batch_size, num_counties, pred_horizon]
        return out

# --- Data Preparation ---
df = pd.read_csv(Config.data_path)
adj_matrix = pd.read_csv(Config.adj_path, index_col=0).values
county_names = df['county'].values
state_names = df['state'].values
time_series = df.filter(regex='^(19|20)').values.astype(np.float32)
n_counties, n_months = time_series.shape

# Build edge_index and edge_attr
edge_src, edge_dst = np.where(adj_matrix > 0)
edge_attr = adj_matrix[edge_src, edge_dst]
edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long).to(Config.device)
edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(Config.device)

# Prepare test input: last 56 months before test
WINDOW_SIZE = Config.train_size - (n_months - Config.train_size)
test_input = time_series[:, -(WINDOW_SIZE + Config.pred_horizon):-Config.pred_horizon]  # [n_counties, window]
test_input = torch.tensor(test_input.T, dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(Config.device)  # [1, window, n_counties, 1]

# Prepare ground truth: last 6 months
test_truth = time_series[:, -Config.pred_horizon:]  # [n_counties, 6]

# --- Load Model ---
model = SpatioTemporalForecaster().to(Config.device)
model.load_state_dict(torch.load(Config.model_file, map_location=Config.device))
model.eval()

# --- Make Predictions ---
with torch.no_grad():
    preds = model(test_input, edge_index, edge_attr)  # [1, n_counties, 6]
    preds = preds.squeeze(0).cpu().numpy()  # [n_counties, 6]

# --- Plotting ---
for i in range(n_counties):
    county = county_names[i]
    state = state_names[i]
    plt.figure(figsize=(10, 6))
    plt.plot(test_truth[i], label='Actual', marker='o', color='blue')
    plt.plot(preds[i], label='GAT-LSTM Forecast', linestyle='--', marker='x', color='orange')
    plt.title(f'GAT-LSTM Forecast vs Actual - {county}, {state}')
    plt.xlabel('Months')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    fname = f"{state.replace(' ', '_')}_{county.replace(' ', '_')}_gat_lstm_forecast.png"
    plt.savefig(os.path.join(Config.plot_dir, fname), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for {county}, {state}")

print("All plots saved in:", Config.plot_dir)
