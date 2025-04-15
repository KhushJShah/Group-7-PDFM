#%% 1. Imports and Configuration
#%% 1. Imports and Configuration
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import os
from ast import literal_eval

# Configuration
DATA_PATH = r'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_COUNTIES = 90  # Verify with your data
WINDOW_SIZE = 100
PREDICTION_HORIZON = 6

#%% 2. Data Loading and Validation
print("Loading data...")
try:
    # Load main dataset
    df = pd.read_csv(DATA_PATH + 'merged_data_unemployment_r9.csv')
    
    # Load adjacency matrix with county-state labels
    adj_df = pd.read_csv(DATA_PATH + 'adjacency_matrix_with_weights_r9.csv', index_col=0)
    county_states = adj_df.index.tolist()  # Get ordered list from adjacency matrix
    
    # Load model configurations
    config_df = pd.read_excel(DATA_PATH + 'results_forward_prediction.xlsx')
except FileNotFoundError as e:
    print(f"Error loading file: {e}")
    raise

# Create county-state identifiers in main DataFrame
df['county_state'] = df['county'] + ', ' + df['state']

config_df['county_state'] = config_df['county'] + ', ' + config_df['state']

# 2. Merge configurations into main DataFrame
df = pd.merge(
    df,
    config_df[['county_state', 'best_order_ar', 'best_order_sarima', 'best_seasonal_order_sarima']],
    on='county_state',
    how='left'
)
# Validate alignment between DataFrame and adjacency matrix
missing_in_data = set(county_states) - set(df['county_state'])
missing_in_adj = set(df['county_state']) - set(county_states)
assert not missing_in_data, f"Counties in adjacency matrix missing in data: {missing_in_data}"
assert not missing_in_adj, f"Counties in data missing in adjacency matrix: {missing_in_adj}"

# Reorder DataFrame to match adjacency matrix order
df = df.set_index('county_state').loc[county_states].reset_index()

# Extract time series data (Counties × Months)
time_series = df.filter(regex='^(19|20)').values.astype(np.float32)  # Ensure ordering matches adjacency matrix
test_dates = df.filter(regex='^(19|20)').columns[-6:].tolist()

#%% 3. Graph Construction with Validation (Updated)
def create_graph_components(adj_df):
    """Create valid PyG graph components from labeled adjacency matrix"""
    edge_index = []
    edge_attr = []
    
    for i, county_i in enumerate(adj_df.index):
        for j, county_j in enumerate(adj_df.columns):
            if i < j and adj_df.iloc[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append([adj_df.iloc[i, j]])
    
    edge_index = torch.tensor(edge_index).T.contiguous().long().to(DEVICE)
    edge_attr = torch.tensor(edge_attr).float().to(DEVICE)
    return edge_index, edge_attr

# Create and validate graph
edge_index, edge_attr = create_graph_components(adj_df)
print(f"Graph contains {edge_index.shape[1]} edges between {len(adj_df)} nodes")
#%% 4. Model Definition with Dimension Safety
class CountyGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GATConv(1, 64, heads=8, edge_dim=1, add_self_loops=False)
        self.lstm = nn.LSTM(64*8, 128, num_layers=2, batch_first=True)
        self.linear = nn.Linear(128, PREDICTION_HORIZON)

    def forward(self, x, edge_index, edge_attr):
        # x shape: [num_counties, window_size, 1]
        batch_size, seq_len, _ = x.size()
        
        # Validate input dimensions
        if edge_index.max() >= batch_size:
            raise ValueError(f"Edge index contains invalid node indices. Max index: {edge_index.max().item()}, Num nodes: {batch_size}")
        
        # Process each timestep through GAT
        gat_outputs = []
        for t in range(seq_len):
            out = self.gat(x[:, t, :], edge_index, edge_attr)
            gat_outputs.append(out.unsqueeze(1))
        
        # Temporal processing
        x = torch.cat(gat_outputs, dim=1)
        x, _ = self.lstm(x)
        return self.linear(x[:, -1, :])

#%%
def train_county_models():
    """Train AR and SARIMA models for each county-state"""
    ar_models = {}
    sarima_models = {}
    
    for idx, row in df.iterrows():
        county_state = row['county_state']
        train_data = time_series[idx, :-6]
        
        # AR Model
        ar_order = row['best_order_ar']
        try:
            if ar_order:
                ar_model = AutoReg(train_data, lags=ar_order).fit()
                ar_models[county_state] = ar_model
        except:
            ar_models[county_state] = None
        
        # SARIMA Model
        sarima_order = row['best_order_sarima']
        seasonal_order = row['best_seasonal_order_sarima']
        try:
            if sarima_order and seasonal_order:
                sarima_model = SARIMAX(train_data, order=sarima_order, 
                                     seasonal_order=seasonal_order).fit(disp=False)
                sarima_models[county_state] = sarima_model
        except:
            sarima_models[county_state] = None
    
    return ar_models, sarima_models

def calculate_weights(train_metrics):
    """Calculate dynamic ensemble weights based on training performance"""
    weights = pd.DataFrame(index=train_metrics.index)
    for model in ['AR', 'SARIMA', 'GAT']:
        weights[f'{model}_Weight'] = 1 / train_metrics[f'{model}_RMSE']
    
    # Normalize and handle NaNs
    total = weights.sum(axis=1)
    for model in ['AR', 'SARIMA', 'GAT']:
        weights[f'{model}_Weight'] = weights[f'{model}_Weight'].div(total)
    
    # Handle failed models
    for idx in weights.index:
        valid_models = [m for m in ['AR', 'SARIMA', 'GAT'] 
                       if not np.isnan(train_metrics.loc[idx, f'{m}_RMSE'])]
        
        if not valid_models:
            weights.loc[idx] = 1/3
        else:
            total_valid = weights.loc[idx, [f'{m}_Weight' for m in valid_models]].sum()
            for model in valid_models:
                weights.loc[idx, f'{model}_Weight'] /= total_valid
                
    return weights.fillna(0)
#%% 5. Training Pipeline with Dimension Checks
def train_gat_model(time_series):
    """Train GAT model with full graph validation"""
    model = CountyGAT().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Prepare training data with dimension checks
    train_data = time_series[:, :-6]  # Shape: [num_counties, 406]
    
    # Create input sequences (sliding window)
    X = []
    y = []
    for i in range(train_data.shape[1] - WINDOW_SIZE - PREDICTION_HORIZON + 1):
        X.append(train_data[:, i:i+WINDOW_SIZE])
        y.append(train_data[:, i+WINDOW_SIZE:i+WINDOW_SIZE+PREDICTION_HORIZON])
    
    X = np.array(X).transpose(1, 0, 2)  # [num_counties, num_windows, window_size]
    y = np.array(y).transpose(1, 0, 2)  # [num_counties, num_windows, horizon]

    # Convert to tensors
    X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(DEVICE)  # [num_counties, num_windows, window_size, 1]
    y_tensor = torch.FloatTensor(y).to(DEVICE)  # [num_counties, num_windows, horizon]

    print("\nTraining GAT Model...")
    for epoch in range(1):
        model.train()
        optimizer.zero_grad()
        
        # Process each window
        total_loss = 0
        for window_idx in range(X_tensor.size(1)):
            x_window = X_tensor[:, window_idx, :, :]  # [num_counties, window_size, 1]
            y_window = y_tensor[:, window_idx, :]     # [num_counties, horizon]
            
            pred = model(x_window, edge_index, edge_attr)
            loss = criterion(pred, y_window)
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Avg Loss: {total_loss/X_tensor.size(1):.4f}")
    
    return model
#%% 6. Main Execution
if __name__ == "__main__":
    # Train models
    ar_models, sarima_models = train_county_models()
    gat_model = train_gat_model(time_series)
    county_states = df['county_state'].tolist()
    
    # Calculate training metrics
    print("\nCalculating training performance...")
    train_metrics = []
    for county_state in county_states:
        idx = county_states.index(county_state)  # Get DataFrame index
        train = time_series[idx, :-6]
        last_train = time_series[idx, -6-PREDICTION_HORIZON:-6]
        
        # AR metrics
        # AR metrics
        ar_model = ar_models.get(county_state)
        ar_pred = ar_model.predict(start=len(train)-PREDICTION_HORIZON, end=len(train)-1) \
                  if ar_model else np.full(PREDICTION_HORIZON, np.nan)
        ar_rmse = np.sqrt(mean_squared_error(last_train, ar_pred)) if not np.isnan(ar_pred).any() else np.nan
        
        # SARIMA metrics
        sarima_model = sarima_models.get(county_state)
        sarima_pred = sarima_model.forecast(PREDICTION_HORIZON) \
                      if sarima_model else np.full(PREDICTION_HORIZON, np.nan)
        sarima_rmse = np.sqrt(mean_squared_error(last_train, sarima_pred)) if not np.isnan(sarima_pred).any() else np.nan
        # GAT metrics
        with torch.no_grad():
            all_gat_input = torch.FloatTensor(time_series[:, -WINDOW_SIZE-6:-6]).unsqueeze(-1).to(DEVICE)  # [N_COUNTIES, WINDOW_SIZE, 1]
            all_gat_pred = gat_model(all_gat_input, edge_index, edge_attr).cpu().numpy()  # [N_COUNTIES, PREDICTION_HORIZON]
            gat_pred = all_gat_pred[idx]
        gat_rmse = np.sqrt(mean_squared_error(last_train, gat_pred))
        
        train_metrics.append({
            'County_State': county_state,
            'AR_RMSE': ar_rmse,
            'SARIMA_RMSE': sarima_rmse,
            'GAT_RMSE': gat_rmse
        })
    
    train_metrics_df = pd.DataFrame(train_metrics)
    weights_df = calculate_weights(train_metrics_df)
    
    # Generate predictions
    print("\nGenerating final predictions...")
    results = []
    for county_idx in range(N_COUNTIES):
        county_state = df.iloc[county_idx]['county_state']  # Get full identifier
        county_name = df.iloc[county_idx]['county']
        test_data = time_series[county_idx, -6:]
        
        # Model predictions using county_state keys
        ar_model = ar_models.get(county_state)
        ar_pred = ar_model.predict(start=len(time_series[county_idx, :-6]), 
                                end=len(time_series[county_idx, :-6])+5) \
                if ar_model else np.full(6, np.nan)
        
        sarima_model = sarima_models.get(county_state)
        sarima_pred = sarima_model.forecast(6) \
                    if sarima_model else np.full(6, np.nan)
        
        with torch.no_grad():
        # Process all counties at once
            all_gat_input = torch.FloatTensor(time_series[:, -WINDOW_SIZE-6:-6]).unsqueeze(-1).to(DEVICE)
            all_gat_pred = gat_model(all_gat_input, edge_index, edge_attr).cpu().numpy()
            gat_pred = all_gat_pred[county_idx]
        
        # Get weights
        weights = weights_df.iloc[county_idx]
        
        # Create weighted ensemble
        valid_preds = []
        valid_weights = []
        for model in ['AR', 'SARIMA', 'GAT']:
            pred = locals()[f"{model.lower()}_pred"]
            weight = weights[f'{model}_Weight']
            if not np.isnan(pred).any() and weight > 0:
                valid_preds.append(pred)
                valid_weights.append(weight)
        
        if valid_preds:
            valid_weights = np.array(valid_weights) / sum(valid_weights)
            ensemble_pred = np.sum([p*w for p,w in zip(valid_preds, valid_weights)], axis=0)
        else:
            ensemble_pred = np.full(6, np.nan)

        
        # Store results
        results.append({
            'County': county_name,
            'Ensemble_RMSE': np.sqrt(mean_squared_error(test_data, ensemble_pred)),
            'Ensemble_MAE': mean_absolute_error(test_data, ensemble_pred),
            'AR_Weight': weights['AR_Weight'],
            'SARIMA_Weight': weights['SARIMA_Weight'],
            'GAT_Weight': weights['GAT_Weight'],
            'AR_Order': df.iloc[county_idx]['best_order_ar'],
            'SARIMA_Order': df.iloc[county_idx]['best_order_sarima'],
            'SARIMA_Seasonal_Order': df.iloc[county_idx]['best_seasonal_order_sarima'],
            'AR_RMSE': np.sqrt(mean_squared_error(test_data, ar_pred)) if not np.isnan(ar_pred).any() else np.nan,
            'SARIMA_RMSE': np.sqrt(mean_squared_error(test_data, sarima_pred)) if not np.isnan(sarima_pred).any() else np.nan,
            'GAT_RMSE': np.sqrt(mean_squared_error(test_data, gat_pred))
        })
        
        # Plot
        plt.figure(figsize=(12,6))
        plt.plot(test_dates, test_data, 'o-', label='Actual')
        plt.plot(test_dates, ensemble_pred, 'x--', label='Ensemble')
        plt.title(f"{county_name} Unemployment Forecast\nRMSE: {results[-1]['Ensemble_RMSE']:.4f}")
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs(DATA_PATH + 'results', exist_ok=True)
    results_df.to_csv(DATA_PATH + 'results/final_ensemble_results.csv', index=False)
    print("\nResults saved to:", DATA_PATH + 'results/final_ensemble_results.csv')


# %%
