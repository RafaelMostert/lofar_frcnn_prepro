"""Prints the split used for training, validation and testing"""
import numpy as np
import pandas as pd

# Load sourcelist
sourcelist_path = "/data2/mostertrij/data/catalogues/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.h5"
sourcelist = pd.read_hdf(sourcelist_path)
# Get fieldnames
field_names = sorted(list(set(sourcelist.Mosaic_ID)))
np.random.seed(42)
np.random.shuffle(field_names)
# split fields
# Train, val, test split
# (expressed in terms of their boundaries)
field_split = [0, 38, 48, 58]
train_fields = field_names[field_split[0]:field_split[1]]
val_fields = field_names[field_split[1]:field_split[2]]
test_fields = field_names[field_split[2]:field_split[3]]
print(f"{len(train_fields)} pointings in training data set:")
print(train_fields)
print(f"{len(val_fields)} pointings in validation data set:")
print(val_fields)
print(f"{len(test_fields)} pointings in test data set:")
print(test_fields)
