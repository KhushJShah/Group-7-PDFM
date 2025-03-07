#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment.csv')

# Extract time series data
time_series_columns = [col for col in df.columns if col.startswith('19') or col.startswith('20')]
time_series_data = df[time_series_columns].values

# Normalize the data
scaler = MinMaxScaler()
time_series_data_normalized = scaler.fit_transform(time_series_data)

# Prepare sequences
def create_sequences(data, n_past=12, n_future=3):
    X, y = [], []
    for i in range(data.shape[1] - n_past - n_future + 1):
        X.append(data[:, i:i+n_past])
        y.append(data[:, i+n_past:i+n_past+n_future])
    return np.array(X), np.array(y)

X, y = create_sequences(time_series_data_normalized)

# Convert to PyTorch tensors
X = torch.FloatTensor(X).permute(1, 0, 2)  # [counties, sequences, time_steps]
y = torch.FloatTensor(y).permute(1, 0, 2) 

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model
input_size = X.size(2)  # Number of time steps in input
hidden_size = 64
num_layers = 2
output_size = y.size(2)  # Number of time steps to predict

model = LSTMModel(input_size, hidden_size, num_layers, output_size)

#%%
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    outputs = model(X)
    optimizer.zero_grad()
    loss = criterion(outputs, y[:, -1, :])  # Compare with the last sequence of y
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X)
    
    # Reshape predictions and true values to match the original data shape
    predictions = np.zeros((58, 412))
    true_values = np.zeros((58, 412))
    
    predictions[:, -3:] = test_outputs.cpu().numpy()
    true_values[:, -3:] = y[:, -1, :].cpu().numpy()
    
    # Inverse transform
    predictions = scaler.inverse_transform(predictions)[:, -3:]
    true_values = scaler.inverse_transform(true_values)[:, -3:]

# Calculate MAE and RMSE
mae = mean_absolute_error(true_values.flatten(), predictions.flatten())
rmse = np.sqrt(mean_squared_error(true_values.flatten(), predictions.flatten()))
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')

# %%
