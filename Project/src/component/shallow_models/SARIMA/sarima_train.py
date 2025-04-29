#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle
import os
from ast import literal_eval

# Load datasets
merged_data = pd.read_csv(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv')
hyperparams = pd.read_excel(r'C:\Users\nupur\computer\Desktop\results_forward_prediction.xlsx')

# Convert string tuples to actual tuples
hyperparams['best_order_sarima'] = hyperparams['best_order_sarima'].apply(literal_eval)
hyperparams['best_seasonal_order_sarima'] = hyperparams['best_seasonal_order_sarima'].apply(literal_eval)

# Create output directory if not exists
output_dir = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\src\component\shallow_models\SARIMA'
os.makedirs(output_dir, exist_ok=True)

# Process each county
for county in merged_data['county'].unique():
    # Filter data for current county
    county_data = merged_data[merged_data['county'] == county]
    
    # Get time series data (adjust column name as needed)
    ts_data = county_data.filter(regex='^(19|20)').mean(axis=0)
    ts_data.index = pd.to_datetime(ts_data.index)
        
    # Split data
    train = ts_data[:406]
    test = ts_data[406:]
    
    # Get hyperparameters from Excel
    county_params = hyperparams[hyperparams['county'] == county]
    if county_params.empty:
        print(f"Hyperparameters not found for county: {county}")
        continue
        
    order = county_params['best_order_sarima'].values[0]
    seasonal_order = county_params['best_seasonal_order_sarima'].values[0]
    
    # Get state name
    state = county_data['state'].iloc[0]
    
    try:
        # Create and fit model
        model = SARIMAX(train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        
        model_fit = model.fit(disp=False)
        
        # Save model
        filename = f"{state}_{county}_sarima.pkl"
        save_path = os.path.join(output_dir, filename)
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_fit, f)
            
        print(f"Successfully saved model for {county}, {state}")
        
    except Exception as e:
        print(f"Error processing {county}, {state}: {str(e)}")

# %%
