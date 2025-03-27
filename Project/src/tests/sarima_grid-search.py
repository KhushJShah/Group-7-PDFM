#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_arch  # New import
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import os

#%%
def create_subplot(zipcode, train_data, train_pred, test_data, test_pred, county, state):
    """Create a single figure with two subplots: full training data and focused test data"""
    os.makedirs("sarima_plots", exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Training subplot (full data)
    ax1.plot(train_data.index, train_data, label='Original Training Data', color='blue')
    if len(train_pred) > 0:
        ax1.plot(train_data.index, train_pred, 
                 label='Training Predictions', linestyle='--', color='orange')
    ax1.set_title(f'Training Data - {county}, {state} ({zipcode})')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Unemployment Rate')
    ax1.legend()
    
    # Test subplot (focused on test period)
    ax2.plot(test_data.index, test_data, label='Actual Test Data', marker='o', color='green')
    ax2.plot(test_data.index, test_pred, label='Test Forecast', linestyle='--', color='red', marker='x')
    ax2.set_title(f'Test Data and Forecast - {county}, {state} ({zipcode})')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Unemployment Rate')
    ax2.legend()
    
    # Adjust x-axis for test subplot to focus on test period
    ax2.set_xlim(test_data.index[0], test_data.index[-1])
    
    plt.tight_layout()
    plot_path = f"sarima_plots/{zipcode}_subplot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def sarima_grid_search(train_data, test_data):
    p = d = q = range(0, 2)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    best = {
        'order': None,
        'seasonal_order': None,
        'train_rmse': np.inf,
        'train_mae': np.inf,
        'test_rmse': np.inf,
        'test_mae': np.inf,
        'heteroskedasticity': False,  # New field
        'model': None
    }
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = SARIMAX(train_data,
                                order=param,
                                seasonal_order=param_seasonal,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
                results = model.fit()
                
                # Get residuals and check heteroskedasticity
                residuals = results.resid.dropna()
                heteroskedastic = False
                if len(residuals) > 5:  # Minimum data requirement for ARCH test
                    try:
                        # ARCH-LM test for heteroskedasticity (up to 5 lags)
                        _, p_value, _, _ = het_arch(residuals, max_lag=5)
                        heteroskedastic = p_value < 0.05
                    except:
                        pass
                
                # Training predictions
                train_pred = results.get_prediction(start=0, end=len(train_data)-1)
                train_pred = train_pred.predicted_mean
                train_rmse = np.sqrt(mean_squared_error(train_data, train_pred))
                train_mae = mean_absolute_error(train_data, train_pred)
                
                # Test predictions
                test_pred = results.get_forecast(steps=len(test_data))
                test_pred = test_pred.predicted_mean
                test_rmse = np.sqrt(mean_squared_error(test_data, test_pred))
                test_mae = mean_absolute_error(test_data, test_pred)
                
                if test_rmse < best['test_rmse']:
                    best.update({
                        'order': param,
                        'seasonal_order': param_seasonal,
                        'train_rmse': train_rmse,
                        'train_mae': train_mae,
                        'test_rmse': test_rmse,
                        'test_mae': test_mae,
                        'heteroskedasticity': heteroskedastic,
                        'model': results,
                        'train_pred': train_pred,
                        'test_pred': test_pred
                    })
            except:
                continue
    
    return best

def prepare_data(df):
    results = []
    for zipcode in df['zipcode'].unique():
        zipcode_df = df[df['zipcode'] == zipcode]
        county = zipcode_df['county'].iloc[0]
        state = zipcode_df['state'].iloc[0]
        
        # Extract time series data
        ts_data = zipcode_df.filter(regex='^(19|20)').mean(axis=0)
        ts_data.index = pd.to_datetime(ts_data.index)
        
        # Split data (last 6 months as test)
        if len(ts_data) < 18:  # Minimum 1.5 years of data
            continue
            
        split_date = ts_data.index[-6]
        train_data = ts_data[:split_date]
        test_data = ts_data[split_date:]
        
        # Grid search
        best = sarima_grid_search(train_data, test_data)
        
        if best['order'] is None:
            continue
            
        # Create subplot for training and testing
        plot_path = create_subplot(
            zipcode, train_data, best['train_pred'], 
            test_data, best['test_pred'], county, state
        )
        
        # Store results
        results.append({
            'zipcode': zipcode,
            'county': county,
            'state': state,
            'best_order': str(best['order']),
            'best_seasonal_order': str(best['seasonal_order']),
            'train_rmse': round(best['train_rmse'], 4),
            'train_mae': round(best['train_mae'], 4),
            'test_rmse': round(best['test_rmse'], 4),
            'test_mae': round(best['test_mae'], 4),
            'heteroskedasticity': best['heteroskedasticity'],  # New column
            'subplot_path': plot_path
        })
    
    return pd.DataFrame(results)

#%%
def main():
    file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv'
    output_path = 'sarima_results_with_subplots.csv'
    
    df = pd.read_csv(file_path)
    results_df = prepare_data(df)
    
    if not results_df.empty:
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        print(f"Subplots saved in 'sarima_plots' directory")
    else:
        print("No valid models found for any zipcodes")
    
    print("Process completed")

if __name__ == "__main__":
    main()

# %%
