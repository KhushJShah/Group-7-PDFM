"""
This script loads pre-trained AutoRegressive (AR) models for each county,
generates 6-month unemployment rate forecasts, and creates comparison plots
of actual vs predicted values for the test period. All visualizations are
saved in the 'ar_plots' directory, with automated model loading and error handling.
"""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
import glob

def plot_test_forecast(test_data, test_pred, county, state):
    """Create focused plot comparing AR forecast with actual test data"""
    os.makedirs("ar_plots", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    # Plot actual vs predicted
    plt.plot(test_data.index, test_data, 
             label='Actual', marker='o', color='#2ca02c')
    plt.plot(test_data.index, test_pred,
             label='AR Forecast', linestyle='--', marker='x', color='#d62728')
    
    # Formatting
    plt.title(f'AR Model Forecast vs Actual - {county}, {state}')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    filename = f"{state}_{county}_ar_forecast.png".replace(" ", "_")
    plot_path = os.path.join("ar_plots", filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

# Load merged data
merged_data = pd.read_csv(r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\data\merged_data_unemployment_r9.csv')

# Get all AR models
model_dir = r'C:\Users\nupur\computer\Desktop\Group-7-PDFM\Project\src\component\shallow_models\AR'
model_files = glob.glob(os.path.join(model_dir, "*.pkl"))

for model_path in model_files:
    try:
        # Extract location info from filename
        filename = os.path.basename(model_path)
        state, county = filename.split("_")[:2]
        county = county.replace("_", " ").replace("_ar.pkl", "")
        
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Get test data
        county_data = merged_data[merged_data['county'] == county]
        ts_series = county_data.filter(regex='^(19|20)').mean(axis=0)
        ts_series.index = pd.to_datetime(ts_series.index)
        test_data = ts_series[-6:]  # Last 6 months
        
        # Generate forecast
        test_pred = model.forecast(steps=6)
        
        # Create plot
        plot_test_forecast(test_data, test_pred, county, state)
        
        print(f"Processed {county}, {state} successfully")
        
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")

# %%
