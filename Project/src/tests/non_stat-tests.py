#%%
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

#%%
# Load your dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Prepare data for modeling
def prepare_data(df):
    heteroskedasticity_results = []
    kpss_results = []
    
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        type = zipcode_df['Type'].iloc[0]
        
        # Select time series columns
        time_series_columns = [col for col in zipcode_df.columns if col.startswith('19') or col.startswith('20') or col.startswith('199')]
        ts_data = zipcode_df[time_series_columns].mean(axis=0)
        
        # Split data into training and testing sets
        train_data = ts_data[:-6]
        test_data = ts_data[-6:]
        
        if type == 'Non-Stationary':
            # Perform seasonal differencing with a period of 12
            ts_data_diff = pd.Series(ts_data).diff(12).dropna()
            train_data_diff = ts_data_diff[:-6]
            test_data_diff = ts_data_diff[-6:]
            
            # Fit AR model on differenced data
            model = AutoReg(train_data_diff, lags=1)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=6)
            
            # Calculate errors
            mae = np.mean(np.abs(test_data_diff - forecast))
            rmse = np.sqrt(np.mean((test_data_diff - forecast)**2))
        else:
            # Fit AR model on original data
            model = AutoReg(train_data, lags=1)
            model_fit = model.fit()
            
            # Forecast
            forecast = model_fit.forecast(steps=6)
            
            # Calculate errors
            mae = np.mean(np.abs(test_data - forecast))
            rmse = np.sqrt(np.mean((test_data - forecast)**2))
        
        # Check for heteroscedasticity using Breusch-Pagan test
        if type == 'Non-Stationary':
            model = ols('value ~ time', data=pd.DataFrame({'value': ts_data_diff, 'time': np.arange(len(ts_data_diff))})).fit()
            residuals = model.resid
            test_result = het_breuschpagan(residuals, model.model.exog)
            print(f"Heteroscedasticity Test for {county}, {state}, Zipcode: {zipcode}:")
            print(f"p-value = {test_result[1]}")
            if test_result[1] < 0.05:
                print("Reject the null hypothesis. There is evidence of heteroscedasticity.")
            else:
                print("Fail to reject the null hypothesis. No evidence of heteroscedasticity.")
            
            heteroskedasticity_results.append({
                'zipcode': zipcode,
                'county': county,
                'state': state,
                'p-value': test_result[1],
                'Conclusion': "Heteroscedasticity Present" if test_result[1] < 0.05 else "No Heteroscedasticity"
            })
        else:
            model = ols('value ~ time', data=pd.DataFrame({'value': ts_data, 'time': np.arange(len(ts_data))})).fit()
            residuals = model.resid
            test_result = het_breuschpagan(residuals, model.model.exog)
            print(f"Heteroscedasticity Test for {county}, {state}, Zipcode: {zipcode}:")
            print(f"p-value = {test_result[1]}")
            if test_result[1] < 0.05:
                print("Reject the null hypothesis. There is evidence of heteroscedasticity.")
            else:
                print("Fail to reject the null hypothesis. No evidence of heteroscedasticity.")
            
            heteroskedasticity_results.append({
                'zipcode': zipcode,
                'county': county,
                'state': state,
                'p-value': test_result[1],
                'Conclusion': "Heteroscedasticity Present" if test_result[1] < 0.05 else "No Heteroscedasticity"
            })
        
        # Perform KPSS test
        if type == 'Non-Stationary':
            result = kpss(ts_data_diff)
            print(f"KPSS Test for {county}, {state}, Zipcode: {zipcode}:")
            print(f"p-value = {result[1]}")
            if result[1] < 0.05:
                print("Reject the null hypothesis. The series is not trend-stationary.")
            else:
                print("Fail to reject the null hypothesis. The series is trend-stationary.")
            
            kpss_results.append({
                'zipcode': zipcode,
                'county': county,
                'state': state,
                'p-value': result[1],
                'Conclusion': "Not Trend-Stationary" if result[1] < 0.05 else "Trend-Stationary"
            })
        else:
            result = kpss(ts_data)
            print(f"KPSS Test for {county}, {state}, Zipcode: {zipcode}:")
            print(f"p-value = {result[1]}")
            if result[1] < 0.05:
                print("Reject the null hypothesis. The series is not trend-stationary.")
            else:
                print("Fail to reject the null hypothesis. The series is trend-stationary.")
            
            kpss_results.append({
                'zipcode': zipcode,
                'county': county,
                'state': state,
                'p-value': result[1],
                'Conclusion': "Not Trend-Stationary" if result[1] < 0.05 else "Trend-Stationary"
            })
    
    # Save results to CSV
    heteroskedasticity_df = pd.DataFrame(heteroskedasticity_results)
    kpss_df = pd.DataFrame(kpss_results)
    
    heteroskedasticity_df.to_csv('heteroskedasticity_results.csv', index=False)
    kpss_df.to_csv('kpss_results.csv', index=False)

#%%
# Main function
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'  # Replace with your dataset path
    
    df = pd.read_csv(file_path)
    prepare_data(df)

if __name__ == "__main__":
    main()

# %%
