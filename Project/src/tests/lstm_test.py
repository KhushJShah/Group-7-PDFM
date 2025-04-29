"""
This script loads each county-state LSTM model, runs it on the last 6 months of the time series (using the preceding 56 months as input for recursive forecasting),
and saves a plot comparing predicted and actual unemployment rates for each county-state pair in the 'lstm_plots' directory.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- Parameters ---
WINDOW_SIZE = 56
PRED_HORIZON = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_DIR = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\lstm_results\models'
DATA_PATH = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv'
PLOT_DIR = r'Project\graphs\lstm_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Model Definition (must match training) ---
class UnemploymentLSTM(nn.Module):
    def __init__(self, input_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=128, num_layers=2, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, PRED_HORIZON)
        )
    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.regressor(x)

# --- Plotting Function ---
def plot_test_forecast(test_data, test_pred, county, state, save_dir=PLOT_DIR):
    plt.figure(figsize=(10, 6))
    plt.plot(test_data, label='Actual', marker='o', color='blue')
    plt.plot(test_pred, label='LSTM Forecast', linestyle='--', marker='x', color='orange')
    plt.title(f'LSTM Forecast vs Actual - {county}, {state}')
    plt.xlabel('Months')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    filename = f"{state.replace(' ', '_')}_{county.replace(' ', '_')}_lstm_forecast.png"
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# --- Load Data ---
merged_data = pd.read_csv(DATA_PATH)
unique_pairs = merged_data[['county', 'state']].drop_duplicates().values
time_cols = [col for col in merged_data.columns if col[:4].isdigit()]

# --- Main Loop ---
for county, state in unique_pairs:
    try:
        # Extract time series for county-state
        county_data = merged_data[(merged_data['county'] == county) & (merged_data['state'] == state)]
        if county_data.empty:
            print(f"No data for {county}, {state}")
            continue
        series = county_data[time_cols].values[0].astype(float)
        
        # Prepare test window and ground truth
        test_window = series[-(WINDOW_SIZE + PRED_HORIZON):-PRED_HORIZON]
        test_truth = series[-PRED_HORIZON:]
        
        # Load model
        model_path = os.path.join(MODEL_DIR, f"{state.replace(' ', '_')}_{county.replace(' ', '_')}_lstm.pth")
        if not os.path.exists(model_path):
            print(f"Model file not found for {county}, {state}")
            continue
        model = UnemploymentLSTM().to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        # Recursive forecasting for PRED_HORIZON steps
        input_seq = torch.FloatTensor(test_window).unsqueeze(0).unsqueeze(-1).to(DEVICE)  # [1, window, 1]
        with torch.no_grad():
            pred = model(input_seq).cpu().numpy().flatten()
        
        # Plot results
        plot_test_forecast(test_truth, pred, county, state, save_dir=PLOT_DIR)
        print(f"Processed {county}, {state}")
        
    except Exception as e:
        print(f"Error processing {county}, {state}: {str(e)}")
