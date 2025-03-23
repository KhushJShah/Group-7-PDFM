#%%
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/merged_data_unemployment_r9.csv')

#%%
necessary_columns = ['county', 'zipcode','state','Count_Person'] + [col for col in df.columns if col.startswith('1990') or col.startswith('1991') or col.startswith('1992') or col.startswith('1993') or col.startswith('1994') or col.startswith('1995') or col.startswith('1996') or col.startswith('1997') or col.startswith('1998') or col.startswith('1999') or col.startswith('2000') or col.startswith('2001') or col.startswith('2002') or col.startswith('2003') or col.startswith('2004') or col.startswith('2005') or col.startswith('2006') or col.startswith('2007') or col.startswith('2008') or col.startswith('2009') or col.startswith('2010') or col.startswith('2011') or col.startswith('2012') or col.startswith('2013') or col.startswith('2014') or col.startswith('2015') or col.startswith('2016') or col.startswith('2017') or col.startswith('2018') or col.startswith('2019') or col.startswith('2020') or col.startswith('2021') or col.startswith('2022') or col.startswith('2023') or col.startswith('2024')]

# Filter the dataset to keep only necessary columns
filtered_df = df[necessary_columns]
#%%
filtered_df.head()
#%%
# Initialize lists to categorize counties
stationary_zipcodes = []
non_stationary_zipcodes = []

# Initialize dictionaries for statewise counts
statewise_stationary = {}
statewise_non_stationary = {}

#%%

# Iterate through each county in the dataset
for zipcode in filtered_df['zipcode'].unique():
    county_name = filtered_df[filtered_df['zipcode'] == zipcode]['county'].iloc[0]  # Store the county name
    state_name = filtered_df[filtered_df['zipcode'] == zipcode]['state'].iloc[0]  # Store the state name
    
    # Extract time-series data for the zipcode
    zipcode_data = filtered_df[filtered_df['zipcode'] == zipcode].iloc[:, 4:]  # Assuming the first three columns are 'county', 'zipcode', and 'state'
    
    # Transpose the data for easier handling
    zipcode_data = zipcode_data.T
    
    # Initialize counters for stationary and non-stationary columns
    stationary_columns = 0
    non_stationary_columns = 0
    
    # Perform ADF test for each column
    for column_name in zipcode_data.columns:
        result = adfuller(zipcode_data[column_name])
        
        print(f"Zipcode: {zipcode}, County: {county_name}, State: {state_name}, Column: {column_name}")
        print(f'ADF Statistic: {result[0]}')
        print(f'p-value: {result[1]}')
        
        if result[1] > 0.05:
            print("Time series is not stationary. Differencing is required.")
            non_stationary_columns += 1
        else:
            print("Time series is stationary.")
            stationary_columns += 1
        
        print("\n")
    
    # Categorize the zipcode based on the majority of its columns
    if stationary_columns > non_stationary_columns:
        stationary_zipcodes.append((zipcode, county_name, state_name))
        if state_name in statewise_stationary:
            statewise_stationary[state_name] += 1
        else:
            statewise_stationary[state_name] = 1
    else:
        non_stationary_zipcodes.append((zipcode, county_name, state_name))
        if state_name in statewise_non_stationary:
            statewise_non_stationary[state_name] += 1
        else:
            statewise_non_stationary[state_name] = 1


#%%
# Print results
print("Stationary Counties:")
for zipcode, county_name, state_name in stationary_zipcodes:
    print(f"Zipcode: {zipcode}, County: {county_name}, State: {state_name}")

# Print statewise counts
print("\nStatewise Counts:")
for state, count in statewise_stationary.items():
    print(f"State: {state}, Stationary Zipcodes: {count}")
for state, count in statewise_non_stationary.items():
    print(f"State: {state}, Non-Stationary Zipcodes: {count}")

# Print total counts
print(f"\nTotal Stationary Zipcodes: {len(stationary_zipcodes)}")
print(f"Total Non-Stationary Zipcodes: {len(non_stationary_zipcodes)}")

#%%
for zipcode, county_name, state_name in non_stationary_zipcodes:
    population_density = filtered_df[filtered_df['zipcode'] == zipcode]['Count_Person'].iloc[0]
    
    # Extract time-series data for the zipcode
    zipcode_data = filtered_df[filtered_df['zipcode'] == zipcode].iloc[:, 4:]  # Assuming the first three columns are 'county', 'zipcode', and 'state'
    
    # Transpose the data for easier handling
    zipcode_data = zipcode_data.T
    
    # Perform differencing
    differenced_data = zipcode_data.diff(12).dropna()
    
    # Plot ACF and PACF for each column
    for column_name in differenced_data.columns:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plot_acf(differenced_data[column_name], ax=plt.gca(), title=f'ACF for {county_name}, {state_name}, Zipcode: {zipcode}, Column: {column_name}\nPopulation Density (log10): {population_density}')
        
        plt.subplot(2, 1, 2)
        plot_pacf(differenced_data[column_name], ax=plt.gca(), title=f'PACF for {county_name}, {state_name}, Zipcode: {zipcode}, Column: {column_name}\nPopulation Density (log10): {population_density}')
        
        plt.tight_layout()
        plt.show()
# %%
for zipcode, county_name, state_name in stationary_zipcodes:
    population_density = filtered_df[filtered_df['zipcode'] == zipcode]['Count_Person'].iloc[0]
    
    # Extract time-series data for the zipcode
    zipcode_data = filtered_df[filtered_df['zipcode'] == zipcode].iloc[:, 4:]  # Assuming the first three columns are 'county', 'zipcode', and 'state'
    
    # Transpose the data for easier handling
    zipcode_data = zipcode_data.T
    
    # Perform differencing
    #differenced_data = zipcode_data.diff(12).dropna()
    
    # Plot ACF and PACF for each column
    for column_name in differenced_data.columns:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plot_acf(differenced_data[column_name], ax=plt.gca(), title=f'ACF for {county_name}, {state_name}, Zipcode: {zipcode}, Column: {column_name}\nPopulation Density (log10): {population_density}')
        
        plt.subplot(2, 1, 2)
        plot_pacf(differenced_data[column_name], ax=plt.gca(), title=f'PACF for {county_name}, {state_name}, Zipcode: {zipcode}, Column: {column_name}\nPopulation Density (log10): {population_density}')
        
        plt.tight_layout()
        plt.show()

#%%
for zipcode, county_name, state_name in stationary_zipcodes:
    population_density = filtered_df[filtered_df['zipcode'] == zipcode]['Count_Person'].iloc[0]
    
    # Extract time-series data for the zipcode
    zipcode_data = filtered_df[filtered_df['zipcode'] == zipcode].iloc[:, 4:]  # Assuming the first three columns are 'county', 'zipcode', and 'state'
    
    # Transpose the data for easier handling
    zipcode_data = zipcode_data.T
    
    # Plot ACF and PACF for each column
    for column_name in zipcode_data.columns:
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plot_acf(zipcode_data[column_name], ax=plt.gca(), lags=50,title=f'ACF for {county_name}, {state_name}, Zipcode: {zipcode}, Column: {column_name}\nPopulation Density (log10): {population_density}')
        
        plt.subplot(2, 1, 2)
        plot_pacf(zipcode_data[column_name], ax=plt.gca(), lags=50,title=f'PACF for {county_name}, {state_name}, Zipcode: {zipcode}, Column: {column_name}\nPopulation Density (log10): {population_density}')
        
        plt.tight_layout()
        plt.show()


# %%
for zipcode, county_name, state_name in stationary_zipcodes:
    # Extract time-series data for the zipcode
    zipcode_data = filtered_df[filtered_df['zipcode'] == zipcode].iloc[:, 4:]  # Assuming the first three columns are 'county', 'zipcode', and 'state'

    # Transpose the data for easier handling
    zipcode_data = zipcode_data.T

    # Create a date index
    date_index = pd.to_datetime([col for col in zipcode_data.index], format='%Y-%m')

    # Plot the time series
    plt.figure(figsize=(10, 6))
    for column_name in zipcode_data.columns:
        plt.plot(date_index, zipcode_data[column_name], label=column_name)

    plt.title(f'Time Series Plot of Unemployment Data for {county_name}, {state_name} (Zipcode: {zipcode})')
    plt.xlabel('Year')
    plt.ylabel('Unemployment Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
# %%
stationary_data = [(county_name, 'Stationary') for _, county_name, _ in stationary_zipcodes]
non_stationary_data = [(county_name, 'Non-Stationary') for _, county_name, _ in non_stationary_zipcodes]

# Combine both lists
combined_data = stationary_data + non_stationary_data

# Create a DataFrame
county_type_df = pd.DataFrame(combined_data, columns=['County Name', 'Type'])

# Save the DataFrame to a CSV file
output_file_path = 'county_type_data.csv'
county_type_df.to_csv(output_file_path, index=False)

# Confirm the file path
print(output_file_path)
# %%
