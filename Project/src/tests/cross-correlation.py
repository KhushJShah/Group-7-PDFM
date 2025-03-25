#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
import os

#%%
def load_data(file_path):
    df = pd.read_csv(file_path)
    ts_cols = [col for col in df.columns if col.startswith(('19', '20'))]
    df['ts_data'] = df[ts_cols].values.tolist()
    return df[['zipcode', 'county', 'state', 'ts_data']]

def compute_cross_correlations(df, max_lag=2):
    # Create a matrix of time series
    zipcodes = df.zipcode.unique()
    ts_matrix = np.array(df.ts_data.tolist())
    
    # Initialize correlation storage
    correlations = []
    
    # Compute pairwise correlations
    for i, target_zc in enumerate(zipcodes):
        for j, source_zc in enumerate(zipcodes):
            if i != j:
                target_ts = ts_matrix[i]
                source_ts = ts_matrix[j]
                
                # Normalize the data
                target_norm = (target_ts - np.mean(target_ts)) / np.std(target_ts)
                source_norm = (source_ts - np.mean(source_ts)) / np.std(source_ts)
                
                # Compute cross-correlation
                corr = correlate(target_norm, source_norm, mode='full')
                max_lag_idx = len(target_ts) - 1
                
                # Extract lags 1 and 2
                lag1 = corr[max_lag_idx + 1]
                lag2 = corr[max_lag_idx + 2]
                
                correlations.append({
                    'source_zipcode': source_zc,
                    'target_zipcode': target_zc,
                    'lag1_corr': lag1,
                    'lag2_corr': lag2
                })
    
    return pd.DataFrame(correlations)

def plot_top_correlations(corr_df, target_zipcode, n_top=5):
    # Filter and sort correlations
    target_corr = corr_df[corr_df.target_zipcode == target_zipcode]
    top_lag1 = target_corr.nlargest(n_top, 'lag1_corr')
    top_lag2 = target_corr.nlargest(n_top, 'lag2_corr')
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Lag 1 Plot
    ax1.barh(top_lag1.source_zipcode.astype(str), top_lag1.lag1_corr, color='darkblue')
    ax1.set_title(f'Top {n_top} Lag-1 Correlations for {target_zipcode}')
    ax1.set_xlabel('Correlation Coefficient')
    
    # Lag 2 Plot
    ax2.barh(top_lag2.source_zipcode.astype(str), top_lag2.lag2_corr, color='darkgreen')
    ax2.set_title(f'Top {n_top} Lag-2 Correlations for {target_zipcode}')
    ax2.set_xlabel('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(f'correlations_{target_zipcode}.png', dpi=300, bbox_inches='tight')
    plt.close()

#%%
def main():
    # Load and prepare data
    df = load_data('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv')
    
    # Compute correlations
    corr_df = compute_cross_correlations(df)
    
    # Analyze specific zipcode (example: 90210)
    target_zipcode = 4001  # Replace with your zipcode of interest
    plot_top_correlations(corr_df, target_zipcode)
    
    # Save all correlations
    corr_df.to_csv('cross_correlations.csv', index=False)

if __name__ == "__main__":
    main()
# %%
