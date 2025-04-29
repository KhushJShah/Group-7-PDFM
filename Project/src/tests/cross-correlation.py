#%%
import pandas as pd
import numpy as np
from scipy.signal import correlate

#%%
# Function to load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    ts_cols = [col for col in df.columns if col.startswith(('19', '20'))]
    df['ts_data'] = df[ts_cols].values.tolist()
    return df[['zipcode', 'county', 'state', 'ts_data']]

#%%
# Function to compute cross-correlations for multiple lags
def compute_cross_correlations(df, max_lag=6):
    # Create a matrix of time series
    zipcodes = df.zipcode.unique()
    ts_matrix = np.array(df.ts_data.tolist())
    
    # Initialize correlation storage
    correlations = []

    # Compute pairwise correlations for all zipcodes
    for i, target_zc in enumerate(zipcodes):
        target_ts = ts_matrix[i]
        target_mean = np.nanmean(target_ts)
        target_std = np.nanstd(target_ts)
        
        for j, source_zc in enumerate(zipcodes):
            if i != j:
                source_ts = ts_matrix[j]
                source_mean = np.nanmean(source_ts)
                source_std = np.nanstd(source_ts)

                # Compute correlations for all lags
                lag_corrs = {}
                for lag in range(1, max_lag + 1):
                    # Align the time series with lag
                    min_length = min(len(target_ts), len(source_ts) - lag)
                    if min_length < 10:  # Ensure sufficient overlap
                        continue
                    
                    target_slice = target_ts[:min_length]
                    source_slice = source_ts[lag:lag + min_length]

                    # Remove pairs with missing values
                    valid_mask = ~(np.isnan(target_slice) | np.isnan(source_slice))
                    if np.sum(valid_mask) < 10:
                        continue
                    
                    target_valid = target_slice[valid_mask]
                    source_valid = source_slice[valid_mask]

                    # Compute Pearson correlation
                    corr_coef = np.corrcoef(target_valid, source_valid)[0, 1]
                    lag_corrs[f'lag_{lag}_corr'] = corr_coef

                correlations.append({
                    'source_zipcode': source_zc,
                    'target_zipcode': target_zc,
                    **lag_corrs  # Add lag correlation values as columns
                })

    return pd.DataFrame(correlations)

#%%
# Main function to execute the workflow
def main():
    # Load and prepare data
    file_path = r'Project\data\merged_data_unemployment_r9.csv'
    df = load_data(file_path)
    
    # Compute correlations for multiple lags (up to max_lag=6)
    max_lag = 6
    corr_df = compute_cross_correlations(df, max_lag=max_lag)
    
    # Save all correlations to CSV file
    output_file = 'cross_correlations.csv'
    corr_df.to_csv(output_file, index=False)
    
    print(f"Cross-correlation results saved to {output_file}")

if __name__ == "__main__":
    main()

# %%
