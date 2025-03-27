#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

#%%
def perform_stationarity_tests(series):
    """Perform ADF (lag=1) and KPSS tests on original data"""
    if len(series) < 2:
        return None

    # ADF Test with fixed lag order of 1
    adf_result = adfuller(series, maxlag=1, autolag=None)
    
    # KPSS Test
    try:
        kpss_result = kpss(series, regression='c', nlags='auto')
    except:
        kpss_result = [np.nan, np.nan, np.nan, {'5%': np.nan}]

    return {
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'kpss_statistic': kpss_result[0],
        'kpss_5pct_cv': kpss_result[3]['5%'],
        'adf_stationary': adf_result[1] < 0.05,
        'kpss_stationary': kpss_result[0] < kpss_result[3]['5%'] if not np.isnan(kpss_result[0]) else False
    }

def analyze_stationarity(df):
    results = []
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        
        # Extract original time series data
        ts_data = zipcode_df.filter(regex='^(19|20)').mean(axis=0)
        
        # Skip short series
        if len(ts_data) < 2:
            continue
            
        test_result = perform_stationarity_tests(ts_data)
        if test_result:
            results.append({
                'zipcode': zipcode,
                'county': county,
                'state': state,
                **test_result
            })
    
    return pd.DataFrame(results)

#%%
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'
    output_path = 'stationarity_results.csv'
    
    df = pd.read_csv(file_path)
    results_df = analyze_stationarity(df)
    
    if not results_df.empty:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    else:
        print("No valid results generated")
    
    print("Analysis completed")

if __name__ == "__main__":
    main()

# %%
