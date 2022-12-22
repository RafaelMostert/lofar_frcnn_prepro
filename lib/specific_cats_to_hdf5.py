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

field_dir = os.getenv('FIELD_DATA')
overwrite = bool(int(os.getenv('PIPE_OVERWRITE')))
# unwrap given paths
cat_paths = [os.path.join(field_dir,f) for f in [os.getenv('SRL_NAME'), os.getenv('GAUS_NAME'),
    os.getenv('SRL_LR_NAME'),os.getenv('GAUS_LR_NAME')]]
if all([os.path.exists(c.replace('.fits','.h5')) for c in cat_paths]) and not overwrite:
    print("DONE: Converted catalogues from fits to HDF5.")
print("Attempting to convert the following files to HDF5:", cat_paths)

for cat_path in cat_paths:
    start = time.time()
    if cat_path.endswith('.fits'):
        cat_path_hdf = cat_path.replace('.fits', '.h5')
        if os.path.exists(cat_path_hdf) and not overwrite:
            print("Skipping as HDF5 version already exists:", cat_path_hdf)
            continue
        if os.path.getsize(cat_path)/1024**3 > 10:
            print("Skipping as fits file is too big:", os.path.getsize(cat_path)/1024**3,'GB')
            continue
        # Load Fits cat
        cat = Table.read(cat_path).to_pandas()
        str_df = cat.select_dtypes([object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            cat[col] = str_df[col]

    # Write to hdf5
    cat.to_hdf(cat_path_hdf, 'df')

