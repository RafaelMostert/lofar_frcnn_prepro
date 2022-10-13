import os
import pickle
import time
from collections import Counter
from sys import argv, exit, path

import numpy as np
import pandas as pd

path.insert(0, os.environ['PROJECTPATH'])
import lib.decision_tree_lib as lib
import warnings
from lib.cutout_object import cutout

# from lib.download_utils import get_wise
warnings.filterwarnings("ignore", category=UserWarning)

"""
Script used to make a list of sources, that comply to given criteria, for which cut-outs will be made. In this version the following options are available:

argv[1] (bool(int))  - Training mode (1) or inference mode (0)
argv[2] (int)  - Max number of sources to include in dataset, set to large number to include all
argv[3] (str)  - The name of the file in which the list will be saved
argv[4] (bool(int)) - Whether or not to overwrite an already existing file
argv[5] (str) - Dataset name
argv[6] (int) - Number of fields to include, set to large number to include all 
argv[7] (bool(int))  - Get data from remote immutable data folders (1=True, 0=False)
"""
start = time.time()
assert len(argv) == 9, 'Script expects 9 input arguments.'
print("Start prepro script 1/4.")
# MODE Training mode or Predict mode 
training_mode = bool(int(argv[1]))

# Assumptions

# Set number of cutouts to be generated
n_cutouts = int(argv[2])
# Set output list name
list_name = argv[3]
# Set overwrite flag (1=True,0=False)
overwrite = bool(int(argv[4]))
# get dataset name
dataset_name = argv[5]
# Number of fields to be included, set to infinity to include all
n_fields = int(argv[6])
# Set remote data flag (1=True,0=False)
remote = bool(int(argv[7]))
# GBC visual inspection required / unresolved threshold
UNRESOLVED_THRESHOLD = argv[8]

if not (overwrite or not os.path.exists(list_name + '.pkl')):
    print('Source list already exists. Skipping since overwrite flag = False.')
    exit()

# Define paths and filenames
assert not os.environ['IMAGEDIR'] in ['train', 'val', 'test'], \
    "root dataset directory name should not be \'train\', \'val\' or \'test\'."
local_dr2_path = os.environ['LOCAL_MOSAICS_PATH_DR2']
decision_tree_cat_path = os.environ['LIKELY_UNRESOLVED_CATALOGUE']
rms_filename = 'mosaic-blanked.rms.fits'
cat_filename = 'mosaic-blanked.cat.fits'
field_filename = 'mosaic-blanked.fits'
CACHE_PATH = os.environ['CACHE_PATH']
MASX_store_dir = os.path.join(CACHE_PATH, '2MASX_queries')
cache_dir = os.path.join(CACHE_PATH, 'cache')
cutout_dir = os.path.join(CACHE_PATH, 'cutout_images')
dataset_dir = os.path.join(os.environ['IMAGEDIR'], dataset_name)
[os.makedirs(d, exist_ok=True) for d in [cache_dir, dataset_dir, cutout_dir, MASX_store_dir]]


# For visualization purposes, store names, total versus peak flux and size
names, total_fluxes, peak_fluxes, source_sizes = [], [], [], []
field_range_dict = {}

def check_if_in_value_added_catalogue(bright):
    # get indices of compcat
    comp_sub = compcat[compcat.Component_Name.isin(bright.Source_Name)]
    # Component cat indices 
    indices = [comp_index_name_dict[n] for n in comp_sub.Component_Name.values]

    # Value added cat indices
    vac_indices = [vac_index_name_dict[n] for n in comp_sub[skey].values]
    return compcat.loc[indices], vac.loc[vac_indices]


# Get all field directories
try:
    exclude_DR1_area = bool(int(os.environ['EXCLUDE_DR1_AREA']))
    print("Excluding DR1 area from pointing selection.")
except:
    exclude_DR1_area = False

if remote:
    immutable_dr2_path = os.environ['MOSAICS_PATH_DR2']
    field_names = [f.name for f in os.scandir(os.path.join(immutable_dr2_path, "RA0h_field")) if f.is_dir() and
                   f.name.startswith('P')] + [f.name for f in
                                              os.scandir(os.path.join(immutable_dr2_path, "RA13h_field")) if
                                              f.is_dir() and f.name.startswith('P')]
    field_folders = [f.path for f in os.scandir(os.path.join(immutable_dr2_path, "RA0h_field")) if f.is_dir() and
                     f.name.startswith('P')] + [f.path for f in
                                                os.scandir(os.path.join(immutable_dr2_path, "RA13h_field")) if
                                                f.is_dir() and f.name.startswith('P')]
else:
    print("Looking for pointing folders in:",local_dr2_path)
    field_names = [f.name for f in os.scandir(local_dr2_path) 
            if f.is_dir() and f.name.startswith('P')]
    field_folders = [f.path for f in os.scandir(local_dr2_path) 
            if f.is_dir() and f.name.startswith('P')]
    print("Found the following fields:",field_names)
local_field_folders = [os.path.join(local_dr2_path, f) for f in field_names]

# If in training_mode, we want labels for our cutouts, thus
# discard field directories that are not present in the value added catalogue
raw_cat = pd.read_hdf(os.environ['LOTSS_RAW_CATALOGUE_DR2'], 'df')
print("Reading source-cat from:",os.environ['LOTSS_RAW_CATALOGUE_DR2'])
decision_cat = pd.read_hdf(decision_tree_cat_path)
decision_dict = {sn:gbc_output for sn, gbc_output in zip(
    decision_cat.Source_Name.values,decision_cat[UNRESOLVED_THRESHOLD].values)}
decision_cat = decision_cat.set_index('Source_Name')
raw_field_names = list(set(raw_cat['Mosaic_ID']))
if exclude_DR1_area:
    # Exclude DR1 area
    compcat = pd.read_hdf(os.environ['LOTSS_COMP_CATALOGUE'], 'df')
    dr1_field_names = list(set(compcat['Mosaic_ID']))
    raw_field_names = [f for f in raw_field_names if not f in dr1_field_names]
    field_folders = [f for n, f in zip(field_names, field_folders) if n in raw_field_names]
    local_field_folders = [f for n, f in zip(field_names, local_field_folders) if n in raw_field_names]
    field_names = [n for n in field_names if n in raw_field_names]

field_names = field_names  # [:n_fields]
field_folders = field_folders  # [:n_fields]
local_field_folders = local_field_folders  # [:n_fields]
[os.makedirs(f, exist_ok=True) for f in local_field_folders]
print('Iterating over the following fields:', field_names)

tally_total,tally_in_vac=0,0
field_paths = [os.path.join(p, field_filename) for p in field_folders]
field_cat_paths = [os.path.join(p, cat_filename) for p in field_folders]
# save fields and paths for subsequent scripts
field_indices = []
for field_idx, (field_name, field_folder, local_field_folder, field_path, field_cat_path) in enumerate(zip(field_names,
                                                                                                           field_folders,
                                                                                                           local_field_folders,
                                                                                                           field_paths,
                                                                                                           field_cat_paths)):
    # Load image and raw PyBDSF source catalogue (NOT gaussians, NOT component, NOT value-added)
    source_cat = raw_cat[raw_cat.Mosaic_ID == field_name]
    if n_cutouts < len(source_cat):
        source_cat = source_cat.iloc[:n_cutouts]


    print(f'\nField: {field_name}\nWe start out with {len(source_cat)} components. Immutable field folder '
          f'{field_folder} and local one {local_field_folder}')

    # Filter out sources that do require visual inspection according to GBC
    keep_indices = [sn_i for sn_i, sn in enumerate(source_cat.Source_Name.values)
            if decision_dict[sn]==0]
    source_cat = source_cat.iloc[keep_indices]
    print(f'After discarding the sources that do not require visual inspection according to the'
            f' GBC we are left with {len(source_cat)} sources.')


    # When in training_mode, check if sources appear in value-added catalog
    if training_mode:
        if new_DR2:
            source_cat, _ = check_if_in_value_added_catalogue(source_cat)
        else:
            source_cat, vac_final = check_if_in_value_added_catalogue(source_cat)
        print(f'or {len(final)} sources in the value added catalogue.')
        tally_in_vac += len(source_cat)
    tally_total += len(source_cat)

    if len(source_cat) == 0:
        print(f"Field {field_name} skipped as none of its sources appear in the merged catalogue.")
        continue
    ras, decs = source_cat.RA, source_cat.DEC
    offset = 220 / 3600  # approx. half the diagonal axis of a 300 arcsec cutout
    field_range_dict[field_name] = (
    np.min(ras) - offset, np.max(ras) + offset, np.min(decs) - offset, np.max(decs) + offset)
    total_fluxes.append(source_cat.Total_flux.values)
    peak_fluxes.append(source_cat.Peak_flux.values)
    source_sizes.append(source_cat.Maj.values)
    if training_mode:
        names.append(source_cat.Component_Name.values)
    else:
        names.append(source_cat.Source_Name.values)

    # Making cut-out objects for each source
    if training_mode:
        new_name_list = [n for n in source_cat['Component_Name'].values]  # .str.decode('utf-8')]
        multis = [cn for cn in source_cat[skey].values if cn in names_multiples]
        print(f"Er zijn {len(multis)} multi comp sources, bijv.", multis[:10])
        cutout_list = [cutout(row, i, ra, dec, va_sname=vac_row.Source_Name, mosaic_id=field_name) for
                       i, ((irow, row), (vac_irow, vac_row), ra, dec) in
                       enumerate(zip(source_cat.iterrows(), vac_final.iterrows(), ras, decs))]

    else:
        new_name_list = [f'{field_name}_{n}' for n in source_cat['Source_Name'].values]  # .str.decode('utf-8')]
        cutout_list = [cutout(row, i, ra, dec, mosaic_id=field_name) for i, ((irow, row), ra, dec) in
                       enumerate(zip(source_cat.iterrows(), ras, decs))]

    # Assign paths to cutout_list
    for c in cutout_list:
        c.set_lofarname_DR2(field_path)

    # Saving cut-out list to pickle
    with open(os.path.join(local_field_folder, list_name + '.pkl'), 'wb') as output:
        pickle.dump(cutout_list, output, pickle.HIGHEST_PROTOCOL)
    print(f'Saved list to {field_name}/{list_name}.pkl')
    # Saving name list 
    with open(os.path.join(local_field_folder, list_name + '_name_list.pkl'), 'wb') as output:
        pickle.dump(new_name_list, output, pickle.HIGHEST_PROTOCOL)
    print(f'Name list saved to {field_name}/{list_name}_name_list.pkl')
    field_indices.append(field_idx)
    if len(field_indices) >= n_fields:
        break

# save stats
np.savez(os.path.join(dataset_dir, 'selected_sources_stats.npz'), names=names, total_fluxes=total_fluxes,
         peak_fluxes=peak_fluxes, source_sizes=source_sizes)
# Save field ranges
field_names = np.array(field_names)[field_indices]
local_field_folders = np.array(local_field_folders)[field_indices]
field_paths = np.array(field_paths)[field_indices]
field_cat_paths = np.array(field_cat_paths)[field_indices]
np.savez(os.path.join(dataset_dir, 'fields.npz'), fname=field_names,
         ffolder=local_field_folders, fpath=field_paths, fcat=field_cat_paths)
np.save(os.path.join(dataset_dir, 'field_ranges.npy'), field_range_dict)
print(f"\nWe created source class objects for {tally_total} sources ranging {len(field_indices)} pointings.")
if training_mode:
    print(f"{tally_in_vac} of the source_cat subset appear in the LoTSS-DR1 value added catalogue.")
print(f'Scipt 1 Creating list of objects done. Time taken: {time.time() - start:.1f} sec\n\n')
