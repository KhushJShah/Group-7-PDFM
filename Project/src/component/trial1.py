# -*- coding: utf-8 -*-
"""
Author: Khush Shah
Date: 02/10/2025
Version: 1.0
"""

#%%
import pandas as pd
embeddings_file_path = 'C:/Users/nupur/computer/Desktop/Group-7-PDFM/Project/data/county_embeddings.csv' #@param {type:"string"}

county_embeddings = pd.read_csv(embeddings_file_path).set_index('place')


#%%
embedding_features = [f'feature{x}' for x in range(330)]
county_embeddings.head(2)

#%%
import datacommons_pandas as dc



#%%
label = 'UnemploymentRate_Person'
# Due to response size limits, we'll query in batches.
batch_size = 200

all_labels = []
for start in range(0, county_embeddings.index.shape[0], batch_size):
    batch_indices = county_embeddings.index[start : start + batch_size]
    batch_data = dc.build_time_series_dataframe(batch_indices, label)
    all_labels.append(batch_data)

df_labels = pd.concat(all_labels)

print(df_labels.shape)
df_labels.head(2)

#%%
import matplotlib.dates as mdates
_ax = df_labels.loc['geoId/01001'].plot(figsize=(10, 3))

#%%
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PMAP_USE_TENSORSTORE'] = 'false'
import timesfm

