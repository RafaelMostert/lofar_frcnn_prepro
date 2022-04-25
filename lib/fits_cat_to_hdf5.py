import time

import numpy as np
import pandas as pd
from astropy.table import Table

# Define cat path
cat_path = '/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits'
cat_path = '/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2.fits'
cat_path = '/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v0.9.srl.fixed.fits'
cat_path = '/home/rafael/data/mostertrij/data/catalogues/LoTSS_DR2_v100.srl.fits'
cat_path = '/home/rafael/data/mostertrij/data/catalogues/GradientBoostingClassifier_lotss_31504_16features_train_test_balanced_new_feaures_DC_pred_correct_fast_final.fits'
cat_path = '/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits'
cat_path = '/home/rafael/data/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v0.99.gaus.fits'
cat_path = '/data2/mostertrij/data/catalogues/LoTSS_DR2_v100.gaus.fits'
cat_path_hdf = cat_path.replace('.fits', '.h5')

# Load Fits cat
start = time.time()
cat = Table.read(cat_path).to_pandas()
str_df = cat.select_dtypes([np.object])
str_df = str_df.stack().str.decode('utf-8').unstack()
for col in str_df:
    cat[col] = str_df[col]
print(cat.info())
print(cat.head())
print(time.time() - start)

# Write to hdf5
cat.to_hdf(cat_path_hdf, 'df')

# Test result
start = time.time()
cat2 = pd.read_hdf(cat_path_hdf, 'df')
print(cat2.info())
print(cat2.head())
print(time.time() - start)
