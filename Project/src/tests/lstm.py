#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import os

#%%
def create_sequences(data, n_steps=12, n_output=6):
    X, y = [], []
    for i in range(len(data) - n_steps - n_output + 1):
        X.append(data[i:(i + n_steps)])
        y.append(data[(i + n_steps):(i + n_steps + n_output)])
    return np.array(X), np.array(y)

def build_lstm_model(n_steps=12, n_features=1):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6))
    model.compile(optimizer='adam', loss='mse')
    return model

def process_zipcode(zipcode_df, n_steps=12, n_output=6):
    ts_cols = [col for col in zipcode_df.columns if col.startswith(('19', '20'))]
    ts_data = zipcode_df[ts_cols].values.flatten()
    
    if zipcode_df['Type'].iloc[0] == 'Non-Stationary':
        ts_data = pd.Series(ts_data).diff(12).dropna().values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(ts_data.reshape(-1, 1))
    
    X, y = create_sequences(scaled_data, n_steps, n_output)
    
    test_size = 6
    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, y_train, X_test, y_test, scaler, ts_data

def train_and_predict(zipcode_df):
    county = zipcode_df['county'].iloc[0]
    state = zipcode_df['state'].iloc[0]
    zipcode = zipcode_df['zipcode'].iloc[0]
    ts_type = zipcode_df['Type'].iloc[0]
    
    X_train, y_train, X_test, y_test, scaler, original_data = process_zipcode(zipcode_df)
    
    model = build_lstm_model()
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    # Training predictions
    train_predict = model.predict(X_train)
    train_predict = scaler.inverse_transform(train_predict)
    y_train_original = scaler.inverse_transform(y_train)
    
    # Test predictions
    test_predict = model.predict(X_test)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_original = scaler.inverse_transform(y_test)
    
    if ts_type == 'Non-Stationary':
        last_train_value = original_data[-18]  # 12 (diff) + 6 (test)
        train_predict = np.cumsum(train_predict, axis=1) + last_train_value
        y_train_original = original_data[12:-6]  # Adjust for differencing and test set
        
        last_test_value = original_data[-18]
        test_predict = np.cumsum(test_predict, axis=1) + last_test_value
        y_test_original = original_data[-6:]
    
    mae = mean_absolute_error(y_test_original, test_predict)
    rmse = np.sqrt(mean_squared_error(y_test_original, test_predict))
    
    # Plot training results
    plt.figure(figsize=(15, 6))
    plt.plot(y_train_original, label='Actual Training')
    plt.plot(train_predict.flatten(), label='Predicted Training')
    plt.title(f'LSTM Training - {county}, {state} ({zipcode})')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'lstm_results/lstm_training_{zipcode}.png', dpi=300)
    plt.close()
    
    # Plot test results
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_original, label='Actual Test')
    plt.plot(test_predict.flatten(), label='Predicted Test')
    plt.title(f'LSTM Forecast - {county}, {state} ({zipcode})')
    plt.xlabel('Months')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'lstm_results/lstm_forecast_{zipcode}.png', dpi=300)
    plt.close()
    
    return {
        'zipcode': zipcode,
        'county': county,
        'state': state,
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'Type': ts_type
    }

#%%
def main():
    df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv')
    results = []
    
    os.makedirs('lstm_results', exist_ok=True)
    
    for zipcode in df['zipcode'].unique():
        try:
            zipcode_df = df[df['zipcode'] == zipcode]
            result = train_and_predict(zipcode_df)
            results.append(result)
        except Exception as e:
            print(f"Error processing {zipcode}: {str(e)}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_results/lstm_predictions.csv', index=False)

if __name__ == "__main__":
    main()
