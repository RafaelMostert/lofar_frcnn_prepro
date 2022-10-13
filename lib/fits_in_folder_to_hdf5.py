"""
Running this script and passing paths in its wake will convert all
the csv
"""
import time
import numpy as np
import pandas as pd
from astropy.table import Table
import sys
import os


# unwrap given paths
paths = sys.argv[1:]
for path in paths:
    files = os.listdir(path)
    cat_paths = [os.path.join(path,f) for f in files if f.endswith('.csv') or f.endswith('.fits')]
    print("Attempting to convert the following files to HDF5:", cat_paths)

for cat_path in cat_paths:
    start = time.time()
    if cat_path.endswith('.fits'):

        cat_path_hdf = cat_path.replace('.fits', '.h5')
        if os.path.exists(cat_path_hdf):
            print("Skipping as HDF5 version already exists:", cat_path_hdf)
            continue
        # Load Fits cat
        cat = Table.read(cat_path).to_pandas()
        str_df = cat.select_dtypes([np.object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            cat[col] = str_df[col]
        #print(cat.info())
        #print(cat.head())
        #print(time.time() - start)
    elif cat_path.endswith('.csv'):
        cat_path_hdf = cat_path.replace('.csv', '.h5')
        if os.path.exists(cat_path_hdf):
            print("Skipping as HDF5 version already exists:", cat_path_hdf)
            continue
        # Load Fits cat
        cat = pd.read_csv(cat_path)
        #print(cat.info())
        #print(cat.head())
        #print(time.time() - start)

    # Write to hdf5
    cat.to_hdf(cat_path_hdf, 'df')

    # Test result
    start = time.time()
    cat2 = pd.read_hdf(cat_path_hdf, 'df')
    #print(cat2.info())
    #print(cat2.head())
    #print(time.time() - start)
