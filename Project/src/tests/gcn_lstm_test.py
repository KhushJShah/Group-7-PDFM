'''
This script loads a trained GCN-LSTM model and evaluates its performance on the test set (last 6 months) for each county-state pair.
'''
#%%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

#%% Configuration
MODEL_PATH = r'Project\src\component\deep learning models\gcn_lstm.pth'
PLOT_DIR = r'Project\graphs\results'
os.makedirs(PLOT_DIR, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% Hyperparameters
WINDOW_SIZE = 45
GCN_HIDDEN = 64
LSTM_HIDDEN = 256
PRED_HORIZON = 6

#%% Load Data
DATA_PATH = r'Project\data\merged_data_unemployment_r9.csv'
ADJ_PATH = r'Project\data\adjacency_matrix_with_weights_r9.csv'

#%% Model Architecture
class CountyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn = GCNConv(1, GCN_HIDDEN)
        self.lstm = nn.LSTM(GCN_HIDDEN, LSTM_HIDDEN, num_layers=5, batch_first=True)
        self.linear = nn.Linear(LSTM_HIDDEN, 1)
        
    def forward(self, x, edge_index):
        batch_size, num_counties, seq_len, _ = x.size()
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x[:, :, t, :].reshape(-1, 1)
            out = F.relu(self.gcn(x_t, edge_index))
            gcn_outputs.append(out.view(batch_size, num_counties, -1))
        x = torch.stack(gcn_outputs, dim=2)
        x = x.view(batch_size * num_counties, seq_len, -1)
        x, _ = self.lstm(x)
        x = x.view(batch_size, num_counties, seq_len, -1)[:, :, -1, :]
        return self.linear(x)

def recursive_forecast(model, initial_data, edge_index, steps=6):
    current_window = initial_data.unsqueeze(0).to(DEVICE)
    predictions = torch.zeros(initial_data.size(0), steps).to(DEVICE)
    
    for step in range(steps):
        with torch.no_grad():
            pred = model(current_window, edge_index)
            pred_reshaped = pred.unsqueeze(-1)
            new_window = torch.cat([
                current_window[:, :, 1:, :],
                pred_reshaped
            ], dim=2)
        current_window = new_window
        predictions[:, step] = pred.squeeze(0).squeeze(-1)
    
    return predictions.cpu().numpy()

#%%
if __name__ == "__main__":
    # Load data and model
    df = pd.read_csv(DATA_PATH)
    county_names = df['county'].values
    state_names = df['state'].values
    time_columns = [col for col in df.columns if col[:4].isdigit()]
    full_data = df[time_columns].values.astype(np.float32)[..., np.newaxis]
    
    # Load adjacency matrix
    adj_df = pd.read_csv(ADJ_PATH, index_col=0)
    edge_index = torch.tensor(np.array(np.where(adj_df.values > 0)), dtype=torch.long).to(DEVICE)

    # Prepare test data
    test_window_start = -PRED_HORIZON - WINDOW_SIZE
    test_input = full_data[:, test_window_start:-PRED_HORIZON, :]
    test_truth = full_data[:, -PRED_HORIZON:, 0]

    # Load trained model
    model = CountyGCN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Generate predictions
    with torch.no_grad():
        test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(DEVICE)
        predictions = recursive_forecast(model, test_input_tensor, edge_index, PRED_HORIZON)


    # Generate plots
    for i, (county, state) in enumerate(zip(county_names, state_names)):
        plt.figure(figsize=(10, 6))
        plt.plot(test_truth[i], label='Actual', marker='o', color='blue')
        plt.plot(predictions[i], label='Predicted', linestyle='--', marker='x', color='orange')
        plt.title(f'GCN-LSTM Forecast - {county}, {state}')
        plt.xlabel('Months')
        plt.ylabel('Unemployment Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sanitize filename
        safe_county = county.replace(" ", "_").replace("/", "-")
        safe_state = state.replace(" ", "_").replace("/", "-")
        plt.savefig(os.path.join(PLOT_DIR, f"{safe_state}_{safe_county}_forecast.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

