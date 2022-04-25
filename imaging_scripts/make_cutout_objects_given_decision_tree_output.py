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
Implement decision tree as in Fig. 5, The LOFAR Two-metre Sky Survey
III. First Data Release: optical/IR identifications and value-added catalogue
[Wendy Williams et al. 2018]

Script used to make a list of sources, that comply to given criteria, for which cut-outs will be made. In this version the following options are available:

argv[1] (int)  - Giving the length of the list to be made
argv[2] (str)  - The name of the file in which the list will be saved
argv[3] (bool) - Whether or not to overwrite an already existing file

Example: 

python3 $PROJECTPATH/imaging_scripts/general_sample_list_create_class.py 100 'list_name' 1 
"""
start = time.time()
assert len(argv) == 9, 'Script expects 9 input arguments.'
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

if not (overwrite or not os.path.exists(list_name + '.pkl')):
    print('Source list already exists. Skipping since overwrite flag = False.')
    exit()

# Define paths and filenames
immutable_dr2_path = os.environ['MOSAICS_PATH_DR2']
local_dr2_path = os.environ['LOCAL_MOSAICS_PATH_DR2']
decision_tree_cat_path = os.environ['LIKELY_UNRESOLVED_CATALOGUE']
# wise_dir = os.path.join(IMAGEDIR,'downloads')
rms_filename = 'mosaic-blanked.rms.fits'
cat_filename = 'mosaic-blanked.cat.fits'
field_filename = 'mosaic-blanked.fits'
CACHE_PATH = os.environ['CACHE_PATH']
MASX_store_dir = os.path.join(CACHE_PATH, '2MASX_queries')
cache_dir = os.path.join(CACHE_PATH, 'cache')
cutout_dir = os.path.join(CACHE_PATH, 'cutout_images')
dataset_dir = os.path.join(os.environ['IMAGEDIR'], dataset_name)
[os.makedirs(d, exist_ok=True) for d in [cache_dir, dataset_dir, cutout_dir, MASX_store_dir]]
tally_total, tally_artefacts, tally_nearby, tally_large, tally_large_and_bright, tally_in_vac = 0, 0, 0, 0, 0, 0

try:
    artefact_and_nearby_filtering = bool(int(os.environ['ARTEFACT_AND_NEARBY_FILTERING']))
except:
    artefact_and_nearby_filtering = False

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
    remote = bool(int(os.environ['REMOTE_IMAGES']))
except:
    remote = True
try:
    exclude_DR1_area = bool(int(os.environ['EXCLUDE_DR1_AREA']))
    print("Excluding DR1 area from pointing selection.")
except:
    exclude_DR1_area = False
if remote:
    field_names = [f.name for f in os.scandir(os.path.join(immutable_dr2_path, "RA0h_field")) if f.is_dir() and
                   f.name.startswith('P')] + [f.name for f in
                                              os.scandir(os.path.join(immutable_dr2_path, "RA13h_field")) if
                                              f.is_dir() and f.name.startswith('P')]
    field_folders = [f.path for f in os.scandir(os.path.join(immutable_dr2_path, "RA0h_field")) if f.is_dir() and
                     f.name.startswith('P')] + [f.path for f in
                                                os.scandir(os.path.join(immutable_dr2_path, "RA13h_field")) if
                                                f.is_dir() and f.name.startswith('P')]
else:
    field_names = [f.name for f in os.scandir(os.path.join(local_dr2_path, "RA0h_field")) if f.is_dir() and
                   f.name.startswith('P')] + [f.name for f in
                                              os.scandir(os.path.join(local_dr2_path, "RA13h_field")) if
                                              f.is_dir() and f.name.startswith('P')]
    field_folders = [f.path for f in os.scandir(os.path.join(local_dr2_path, "RA0h_field")) if f.is_dir() and
                     f.name.startswith('P')] + [f.path for f in
                                                os.scandir(os.path.join(local_dr2_path, "RA13h_field")) if
                                                f.is_dir() and f.name.startswith('P')]
    # field_names = [f.name for f in os.scandir(local_dr2_path) if f.is_dir() and f.name.startswith('P')]
    # field_folders = [f.path for f in os.scandir(local_dr2_path) if f.is_dir() and f.name.startswith('P')]
local_field_folders = [os.path.join(local_dr2_path, f) for f in field_names]

# If in training_mode, we want labels for our cutouts, thus
# discard field directories that are not present in the value added catalogue


raw_cat = pd.read_hdf(os.environ['LOTSS_RAW_CATALOGUE_DR2'], 'df')
decision_cat = pd.read_hdf(decision_tree_cat_path)
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

    # Only use sources where decision tree thinks they require LGZ
    source_cat = source_cat[decision_cat.loc[source_cat.Source_Name]["0.20"] == 0]

    print(f'\nField: {field_name}\nWe start out with {len(source_cat)} sources. Immutable field folder '
          f'{field_folder} and local one {local_field_folder}')
    tally_total += len(source_cat)

    if artefact_and_nearby_filtering:
        # Check for artefacts
        artefacts, not_artefacts = lib.check_for_artefacts(source_cat, overwrite=overwrite, store_dir=cache_dir)
        lib.end_of_tree(source_cat, artefacts, "Artefacts (need visual inspection)")
        if not artefacts is None:
            tally_artefacts += len(artefacts)

        # Check for large optical galaxies
        large_optical_galaxies, not_large_optical_galaxies = lib.check_for_large_optical_galaxies(not_artefacts,
                                                                                                  MASX_size_arcsec,
                                                                                                  store_dir=MASX_store_dir,
                                                                                                  overwrite=overwrite)
        lib.end_of_tree(source_cat, large_optical_galaxies, "Large optical galaxies")
        if not large_optical_galaxies is None:
            tally_nearby += len(large_optical_galaxies)
    else:
        print("Skipping artefact and nearby galaxy filtering.")
        not_large_optical_galaxies = source_cat

    # Check if source is large
    large, not_large = lib.check_if_source_is_large(not_large_optical_galaxies,
                                                    PyBDSF_major_axis_arcsec)
    lib.slice_of_tree(source_cat, large, "Large radio objects")
    if not large is None:
        tally_large += len(large)

    ##[Large:yes] Check if source is bright
    bright, not_bright = lib.check_if_source_is_bright(large, total_flux_density_in_mJy)
    lib.end_of_tree(source_cat, bright, "Bright radio objects")
    if not bright is None:
        tally_large_and_bright += len(bright)

    # Choose decision tree subset to proceed with
    if must_be_large:
        final = large
        if must_be_bright == 1:
            print("Dataset must be large and bright.")
            final = bright
        elif must_be_bright == 0:
            print("Dataset must be large. Can be bright or faint.")
            final = large
        elif must_be_bright == -1:
            print("Dataset must be large and faint.")
            final = not_bright
    else:
        raise NotImplementedError("Implement the options for dataset to be returned when must_be_large is False.")

    # When in training_mode, check if sources appear in value-added catalog
    if training_mode:
        if new_DR2:
            final, _ = check_if_in_value_added_catalogue(final)
        else:
            final, vac_final = check_if_in_value_added_catalogue(final)
        print(f'or {len(final)} sources in the value added catalogue.')
        tally_in_vac += len(final)
    if len(final) == 0:
        print(f"Field {field_name} skipped as none of its sources appear in the merged catalogue.")
        continue
    # else:
    #    cras,cdecs = final.RA, final.DEC
    ras, decs = final.RA, final.DEC
    offset = 220 / 3600  # approx. half the diagonal axis of a 300 arcsec cutout
    field_range_dict[field_name] = (
    np.min(ras) - offset, np.max(ras) + offset, np.min(decs) - offset, np.max(decs) + offset)
    total_fluxes.append(final.Total_flux.values)
    peak_fluxes.append(final.Peak_flux.values)
    source_sizes.append(final.Maj.values)
    if training_mode:
        names.append(final.Component_Name.values)
    else:
        names.append(final.Source_Name.values)

    # Making cut-out objects for each source
    if training_mode:
        new_name_list = [n for n in final['Component_Name'].values]  # .str.decode('utf-8')]
        multis = [cn for cn in final[skey].values if cn in names_multiples]
        print(f"Er zijn {len(multis)} multi comp sources, bijv.", multis[:10])
        cutout_list = [cutout(row, i, ra, dec, va_sname=vac_row.Source_Name, mosaic_id=field_name) for
                       i, ((irow, row), (vac_irow, vac_row), ra, dec) in
                       enumerate(zip(final.iterrows(), vac_final.iterrows(), ras, decs))]

    else:
        new_name_list = [f'{field_name}_{n}' for n in final['Source_Name'].values]  # .str.decode('utf-8')]
        cutout_list = [cutout(row, i, ra, dec, mosaic_id=field_name) for i, ((irow, row), ra, dec) in
                       enumerate(zip(final.iterrows(), ras, decs))]

    # Assign paths to cutout_list
    for c in cutout_list:
        c.set_lofarname_DR2(field_path)
        # wisename = get_wise(c.c_source.ra,c.c_source.dec,1, wise_dir)
        # c.set_wisename(wisename)

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
print(
    f"\nIn total from the {tally_total} sources we start with, we get: \n{tally_artefacts} artefacts.\n{tally_nearby} nearby starforming ",
    f"galaxies.\n{tally_large} sources larger than {PyBDSF_major_axis_arcsec} arcsec,",
    f"\n{tally_large_and_bright} of those also have a total flux larger than {total_flux_density_in_mJy} mJy")
if training_mode:
    print(f"We require our final subset to be: large {must_be_large}, and bright {must_be_bright}.")
    print(f"{tally_in_vac} of the final subset appear in the LoTSS-DR1 value added catalogue.")
else:
    print(f"We require our final subset to be: large {must_be_large}, and bright {must_be_bright}.")
print(f'Scipt 1 Creating list of objects done. Time taken: {time.time() - start:.1f} sec\n\n')
