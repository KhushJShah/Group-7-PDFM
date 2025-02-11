'''
This file has the code to produce the poc-dataset, i.e. a merged file with common information from the datasets:
1. conus27.csv
2. county_unemployment.csv
3. zcta_poverty.csv

This dataset will work as a poc for the GNN models, and we can add all the relevant information from the other columns moving forward.
From the region 9, we will be currently working just on the counties from California state. 

'''

#%%
'''Importing the libraries'''
import pandas as pd
import os


#%%
'''Loading the files'''
data1 = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/conus27.csv')
data2 = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/county_unemployment.csv')
data3 = pd.read_csv('C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/zcta_poverty.csv')


#%%
'''Glimpse of the data'''
print('Conus27: \n',data1)
print('County unemployment: \n',data2)
print('Poverty: \n',data3)

#%%
'''Splitting the columns so that the zsipcodes match'''
# Step: Filter data1 to select rows where state == 'CA'
data1_filtered = data1[data1['state'] == 'CA']

# Step: Splitting and renaming columns in each dataset
for df in [data1_filtered, data2, data3]:
    df['zipcode'] = df['place'].str.split('/').str[0]
    df.drop(columns=['place'], inplace=True)

#%%
'''Merging the datasets with common information for the CA counties'''
# Step: Merging datasets on 'zipcode'
merged_data = pd.merge(data1_filtered, data2, on='zipcode', how='outer')
merged_data = pd.merge(merged_data, data3, on='zipcode', how='outer')

# Displaying the final merged dataset
print(merged_data)