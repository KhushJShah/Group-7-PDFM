#%%
import pandas as pd
import os
import pickle
from statsmodels.tsa.ar_model import AutoReg
from ast import literal_eval

# Load datasets
merged_data = pd.read_csv(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv')
hyperparams = pd.read_excel(r'C:\Users\nupur\computer\Desktop\results_forward_prediction.xlsx')

# Convert best_order_ar column to integer
hyperparams['best_order_ar'] = hyperparams['best_order_ar'].astype(int)
# Create output directory
output_dir = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\src\component\shallow_models\AR'
os.makedirs(output_dir, exist_ok=True)

# Process each county
for county in merged_data['county'].unique():
    try:
        # Filter county data
        county_data = merged_data[merged_data['county'] == county]
        state = county_data['state'].iloc[0]
        
        # Extract time series
        ts_series = county_data.filter(regex='^(19|20)').mean(axis=0)
        ts_series.index = pd.to_datetime(ts_series.index)
        ts_data = ts_series.sort_index().values
        
        # Validate length
        if len(ts_data) != 412:
            print(f"Skipping {county} - incorrect length {len(ts_data)}")
            continue
            
        # Split data
        train = ts_data[:406]
        
        # Get AR order
        county_params = hyperparams[hyperparams['county'] == county]
        if county_params.empty:
            print(f"No hyperparams for {county}")
            continue
            
        order = county_params['best_order_ar'].values[0]
        
        # Train and save model
        model = AutoReg(train, lags=order)
        model_fit = model.fit()
        
        filename = f"{state}_{county}_ar.pkl".replace(" ", "_")
        with open(os.path.join(output_dir, filename), 'wb') as f:
            pickle.dump(model_fit, f)
            
        print(f"Saved {county}, {state} (order {order})")
        
    except Exception as e:
        print(f"Failed {county}: {str(e)}")

# %%
