#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

#%%
def plot_rolling_statistics(df, window=12):
    """
    Plot original time series, rolling mean, and rolling variance for each zipcode.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with time series data
    window (int): Window size for rolling statistics (default is 12 for monthly data)
    """
    # Create output directory for plots
    import os
    os.makedirs("rolling_stats_plots", exist_ok=True)
    
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        
        # Extract time series data
        ts_data = zipcode_df.filter(regex='^(19|20)').mean(axis=0)
        ts_data.index = pd.to_datetime(ts_data.index)
        
        # Calculate rolling statistics
        rolling_mean = ts_data.rolling(window=window).mean()
        rolling_var = ts_data.rolling(window=window).var()
        
        # Create plot
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1])
        
        # Original series and rolling mean
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(ts_data.index, ts_data, label='Original', color='blue')
        ax1.plot(rolling_mean.index, rolling_mean, label=f'{window}-month Rolling Mean', color='red')
        ax1.set_title(f'Time Series and Rolling Mean - {county}, {state} ({zipcode})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Unemployment Rate')
        ax1.legend()
        
        # Rolling variance
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(rolling_var.index, rolling_var, label=f'{window}-month Rolling Variance', color='green')
        ax2.set_title(f'Rolling Variance - {county}, {state} ({zipcode})')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Variance')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'rolling_stats_plots/{zipcode}_rolling_stats.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot created for zipcode {zipcode}")

#%%
# Main execution
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'
    df = pd.read_csv(file_path)
    plot_rolling_statistics(df)
    print("All plots created successfully.")

if __name__ == "__main__":
    main()

# %%
