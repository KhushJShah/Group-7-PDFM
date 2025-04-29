'''
This file plots the time series graph for the counties for all the unemployment data from 412 months.
'''
#%%
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
def plot_all_zipcodes(df, output_dir='timeseries_plots'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get time series columns (assuming columns are formatted as 'YYYY-MM')
    ts_cols = [col for col in df.columns if col[:4].isdigit() and '-' in col]
    
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        
        # Extract time series data
        ts_data = zipcode_df[ts_cols].squeeze()
        
        # Convert column names to datetime
        dates = pd.to_datetime(ts_cols)
        
        # Create plot
        plt.figure(figsize=(15, 6))
        plt.plot(dates, ts_data.values, label=f'Zipcode: {zipcode}')
        
        # Format plot
        plt.title(f'Time Series for {county}, {state} ({zipcode})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{output_dir}/timeseries_{zipcode}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

#%%
df = pd.read_csv('Project/data/merged_data_unemployment_r9.csv')
plot_all_zipcodes(df)

# %%
