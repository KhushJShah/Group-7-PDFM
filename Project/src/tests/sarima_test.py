"""
This script loads pre-trained SARIMA models for each county, extracts the corresponding test set (last 6 months) of unemployment rate data, generates out-of-sample forecasts, and plots the predicted versus actual values for each county and state. 
All plots are saved in the 'sarima_test_plots' directory. The script automatically finds and processes all SARIMA model files, handling data extraction, forecasting, and visualization in a streamlined loop with error handling for robustness.
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import glob

def plot_test_forecast(test_data, test_pred, county, state, zipcode=None):
    """Create focused plot for test period with confidence interval"""
    os.makedirs("sarima_test_plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted
    plt.plot(test_data.index, test_data, label='Actual', marker='o', color='#1f77b4')
    plt.plot(test_data.index, test_pred, label='SARIMA Forecast', 
             linestyle='--', marker='x', color='#ff7f0e')
    
    # Formatting
    title = f'SARIMA Forecast vs Actual - {county}, {state}'
    if zipcode:
        title += f' ({zipcode})'
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    filename = f"{state}_{county}_forecast.png".replace(" ", "_")
    plot_path = os.path.join("sarima_plots", filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

# Load test data
merged_data = pd.read_csv(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv')

# Get all saved models
model_dir = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\src\component\shallow_models\SARIMA'
model_files = glob.glob(os.path.join(model_dir, "*.pkl"))

for model_path in model_files:
    try:
        # Extract state/county from filename
        filename = os.path.basename(model_path)
        state, county = filename.split("_")[:2]
        county = county.replace("_", " ")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get test data
        county_data = merged_data[merged_data['county'] == county]
        ts_series = county_data.filter(regex='^(19|20)').mean(axis=0)
        ts_series.index = pd.to_datetime(ts_series.index)
        test_data = ts_series[-6:]  # Last 6 months
        
        # Generate forecast
        forecast = model.get_forecast(steps=6)
        test_pred = forecast.predicted_mean
        conf_int = forecast.conf_int()
        
        # Create plot
        plot_test_forecast(test_data, test_pred, county, state)
        
        print(f"Successfully processed {county}, {state}")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# %%
