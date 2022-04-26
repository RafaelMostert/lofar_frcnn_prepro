""" Takes in a csv with the positios of all bounding boxes per cutout (in fits format),
converts the FITS cutouts to images (jpg or png),
converts the csv information to XML files for training CLARAN or faster-rcnn.pytorch,
converts the csv information to a json file for inspection using VIA (VGG Image Annotator)
which is Oxfords interactive webpage.
"""
import os
import pickle
from sys import argv, path

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('Agg')
path.insert(0, os.environ['PROJECTPATH'])
from lib.label_library import split_image_names_and_objects_standardized
from lib.label_library import convert_fits_to_images
from lib.label_library import create_COCO_style_directory_structure
from lib.label_library import create_dict_from_images_and_annotations_coco_version
# from lib.label_library import populate_VOC_structure_with_train_val_test_split
from lib.label_library import populate_coco_structure, _evaluate_gt_bboxes
import time

start = time.time()
# User input below
##########################################################
# This demo requires a set of fits files to be converted to images
# and a corresponding csv catalogue containing bounding boxes 
# and classes for each fits file
dr1_dr2_comparison_only = False
assert len(argv) == 15, f'Script expects 14 input arguments, not {len(argv) - 1}'
IMAGEDIR = os.environ['IMAGEDIR']
CACHE_DIR = os.environ['CACHE_PATH']
COMP_CAT_PATH = os.environ['LOTSS_COMP_CATALOGUE']
comp_cat = pd.read_hdf(COMP_CAT_PATH)
fits_directory = os.path.join(IMAGEDIR, 'cutouts')
contour_levels = [3, 5]  # draw 3 and 5 sigma contourfills in gb of rgb images
# train_val_test_split = [0.6,0.8,1]
cutout_key = 'Orig_Source_Name'
fixed_cutout_size = bool(int(argv[1]))
incl_diff = bool(int(argv[2]))
difficult_list_name = str(argv[3])
clip = bool(int(argv[4]))
if clip:
    clip_low = float(argv[5])
    clip_high = float(argv[6])
dataset_name = argv[7]
DEBUG_PATH = os.path.join(IMAGEDIR, dataset_name, 'debug')
os.makedirs(DEBUG_PATH, exist_ok=True)
# If imsize=None, there will be no rescaling
imsize = int(argv[8])
imsize_arcsec = int(argv[9])
precomputed_bboxes_enabled = bool(int(argv[10]))
training_mode = bool(int(argv[11]))
remove_unresolved = bool(int(argv[12]))
unresolved_threshold = str(argv[13])
sigma_box_fit = int(argv[14])
segmentation_dir = os.path.join(CACHE_DIR, 'segmentation_maps_' + unresolved_threshold)
root_directory_pytorch = os.path.join(IMAGEDIR, dataset_name)
os.makedirs(root_directory_pytorch, exist_ok=True)
# csv_path = os.path.join(root_directory_pytorch,'labeled_rotated_list.csv')
cutout_list_path = os.path.join(root_directory_pytorch, 'labeled_annotated_cutouts.pkl')
# For FasterRCNN (pytorch)
extension_pytorch = '.png'
# For CLARAN (tensorflow)
extension_CLARAN = '.png'
# Plot optical sources during train mode in debug images
plot_optical = False
root_directory_CLARAN = IMAGEDIR
print(f'''Start prepro script 4/4.
Fixed cut-out size set to: {fixed_cutout_size}
PNG images will be clipped: {clip}
Lower/upper limit (sigma): {clip_low}, {clip_high}
Final imsize in pixels {imsize}
Remove unresolved radio component: {remove_unresolved}
Precomputed bounding boxes enabled  {precomputed_bboxes_enabled}''')

#####################################################################
# Create dataset for Detectron2
#####################################################################

print(f'{"#" * 80} \nCreate and populate training directories for Detectron 2\n{"#" * 80}')
# Create a directory structure identical 
all_directory, train_directory, val_directory, test_directory, annotations_directory \
    = create_COCO_style_directory_structure(root_directory_pytorch)

# Get cutout list
with open(cutout_list_path, 'rb') as input:
    cutout_list = np.array(pickle.load(input))

# Get lists of sources per cutout
# csv = pd.read_csv(csv_path)
# cutout_names_old = list(csv[cutout_key].tolist())
cutout_names = np.array([c.c_source.sname for c in cutout_list])
# assert cutout_names_old == cutout_names
# cutout_names = np.array(sorted(set(cutout_names), key=cutout_names.index))


# List of image names
# save_appendix = '_radio_DR2.fits'
# save_appendix_rms = '_radio_rms_DR2.fits'
if remove_unresolved:
    save_appendix = f'_{imsize_arcsec}arcsec_large_radio_DR2_removed.fits'
    save_appendix_rms = f'_{imsize_arcsec}arcsec_large_radio_rms_DR2_removed.fits'
else:
    save_appendix = f'_{imsize_arcsec}arcsec_large_radio_DR2.fits'
    save_appendix_rms = f'_{imsize_arcsec}arcsec_large_radio_rms_DR2.fits'
rotation_angles = np.array([c.rotation_angle_deg for c in cutout_list])
fits_file_paths = np.array([os.path.join(fits_directory, str(name) + save_appendix)
                            for name in cutout_names])
rms_fits_file_paths = np.array([os.path.join(fits_directory, str(name) + save_appendix_rms) for name
                                in cutout_names])
image_names = np.array([f'{name}_radio_DR2_rotated{angle_deg}deg' for name, angle_deg in
                        zip(cutout_names, rotation_angles)])
"""
print('cutoutpat:', cutout_list_path)
print('len cutoutlist:', len(cutout_list))
print('cutout_names:', cutout_names) print('image_names:', image_names)
print('fitsfile_paths:', fits_file_paths)
print('rms_paths:', rms_fits_file_paths)
sdfsdf
"""

# Convert fits to pngs
# discard cutouts that contain sources where the number of reported components is higher than 5
# discard cutouts with outlying image widths in the case of fixed width cutouts
# cutout_list, image_names, cutout_names, fits_file_paths
idx = convert_fits_to_images(rotation_angles, CACHE_DIR, fits_file_paths,
                             image_names, cutout_list, cutout_names, root_directory_pytorch,
                             all_directory, extension_pytorch, cutout_key, fixed_cutout_size, scaling='sqrt',
                             max_component=5,
                             overwrite=True, contours=contour_levels, rms_fits_file_paths=rms_fits_file_paths,
                             clip_image=clip, upper_clip=clip_high, lower_clip=clip_low, resize=imsize,
                             save_appendix=dataset_name)

# Create a list of image objects (just a tuple containing x and y min and max, n_comp and n_peak) per cutout
cutout_list = cutout_list[idx]
cutout_names = cutout_names[idx]
fits_file_paths = fits_file_paths[idx]
rms_fits_file_paths = rms_fits_file_paths[idx]
rotation_angles = rotation_angles[idx]
image_names = image_names[idx]

# Check how well gt performs on our evaluation metric
# cutout_names = np.array([name.replace('_radio_DR2','') for name in image_names])
scale_factor = cutout_list[0].scale_factor
# print('Scale_factor', scale_factor)
# scale_factor = 1.492537313432836
# print('Scale_factor', scale_factor)
#        debug=False, scale_factor=1.492537313432836) 
# exit()
# Sanity check: Visualize labels and box size distributions
"""
sanity_check_visualization(DEBUG_PATH, image_names, all_directory, cutout_list,
            extension_pytorch,
            fixed_cutout_size)
"""

# Remove old preprocessing plots
if training_mode and plot_optical:
    prepro_plot_dir = os.path.join(DEBUG_PATH, 'boxes_with_optical')
    for s in ['train', 'val', 'test']:
        sub_dir = os.path.join(prepro_plot_dir, s)
        os.makedirs(sub_dir, exist_ok=True)
        for f in os.listdir(sub_dir):
            assert f.endswith('.png'), 'Directory should only contain images.'
        for f in os.listdir(sub_dir):
            os.remove(os.path.join(sub_dir, f))
else:
    prepro_plot_dir = os.path.join(DEBUG_PATH, 'boxes')
os.makedirs(prepro_plot_dir, exist_ok=True)
os.makedirs(os.path.join(DEBUG_PATH, 'DR1_DR2_comparison'), exist_ok=True)
if not plot_optical and os.path.exists(prepro_plot_dir):
    for f in os.listdir(prepro_plot_dir):
        assert f.endswith('.png'), 'Directory should only contain images.'
    for f in os.listdir(prepro_plot_dir):
        os.remove(os.path.join(prepro_plot_dir, f))
if dr1_dr2_comparison_only:
    for f in os.listdir(os.path.join(DEBUG_PATH, 'DR1_DR2_comparison')):
        assert f.endswith('.png'), 'Directory should only contain images.'
    for f in os.listdir(os.path.join(DEBUG_PATH, 'DR1_DR2_comparison')):
        os.remove(os.path.join(DEBUG_PATH, 'DR1_DR2_comparison', f))

if training_mode:
    save_optical = True
    if not 'Source_Name' in comp_cat.keys():
        splitted_index_list = [list(range(len(image_names)))]
    else:
        # Shuffle and split image names and objects into groups for train/test/val
        splitted_index_list = \
            split_image_names_and_objects_standardized(CACHE_DIR, cutout_list, image_names)

    for im_idx, im_dir, term in zip(splitted_index_list,
                                    [train_directory, val_directory, test_directory], ['train', 'val', 'test']):
        if im_idx == []:
            print(f"{term} data split is empty. Skipping.")
            continue

        # """
        # Create a JSON annotation file for each train/test/val split and place it inside the annot. folder
        create_dict_from_images_and_annotations_coco_version(image_names[im_idx],
                                                             fits_file_paths[im_idx], rms_fits_file_paths[im_idx],
                                                             cutout_list[im_idx],
                                                             extension_pytorch, image_dir=all_directory,
                                                             image_destination_dir=im_dir,
                                                             annotations_dir=annotations_directory,
                                                             json_dir=annotations_directory,
                                                             json_name=f'VIA_json_{term}.pkl', imsize=imsize,
                                                             debug_path=DEBUG_PATH,
                                                             debug=True,  # True
                                                             remove_unresolved=remove_unresolved,
                                                             save_optical=save_optical, plot_optical=plot_optical,
                                                             dr1_dr2_comparison_only=dr1_dr2_comparison_only,
                                                             precomputed_bboxes_enabled=precomputed_bboxes_enabled,
                                                             term=term)

        if dr1_dr2_comparison_only:
            continue
        # Copy images to their appropriate train/test/val directories
        populate_coco_structure(all_directory, im_dir, image_names[im_idx], extension_pytorch)
        # """

        # check bboxes again
        _evaluate_gt_bboxes(dataset_name, DEBUG_PATH, cutout_list[im_idx], cutout_names[im_idx],
                            im_dir, imsize, imsize_arcsec, remove_unresolved, flip_y=False, debug=True,  # True,
                            scale_factor=1, plot_misboxed=False,
                            plot_optical=plot_optical, segmentation_dir=segmentation_dir,
                            sigma_box_fit=sigma_box_fit)
        print()

        # Debug plot validation images
        """
        if term == 'val':
        convert_fits_to_images(rotation_angles, CACHE_DIR, fits_file_paths,
                image_names, cutout_list, cutout_names, root_directory_pytorch,
            all_directory, extension_pytorch, cutout_key,fixed_cutout_size,scaling='sqrt', max_component=5, 
            overwrite=True, contours=contour_levels, rms_fits_file_paths=rms_fits_file_paths,
            clip_image=clip, upper_clip=clip_high, lower_clip=clip_low, resize=imsize,
            save_appendix=dataset_name)
        """

    if not dr1_dr2_comparison_only:
        print("Over-all evaluation:")
        _evaluate_gt_bboxes(dataset_name, DEBUG_PATH, cutout_list, cutout_names, all_directory, imsize,
                            imsize_arcsec, remove_unresolved, debug=False, scale_factor=1, plot_misboxed=True,  # True,
                            segmentation_dir=segmentation_dir, sigma_box_fit=sigma_box_fit)
        print()
else:

    # Create a JSON annotation file for each train/test/val split and place it inside the annot. folder
    create_dict_from_images_and_annotations_coco_version(image_names,
                                                         fits_file_paths, rms_fits_file_paths, cutout_list,
                                                         extension_pytorch, image_dir=all_directory,
                                                         image_destination_dir=all_directory,
                                                         annotations_dir=annotations_directory,
                                                         json_dir=annotations_directory,
                                                         json_name=f'VIA_json_inference.pkl', imsize=imsize,
                                                         debug=False,
                                                         precomputed_bboxes_enabled=precomputed_bboxes_enabled,
                                                         remove_unresolved=remove_unresolved,
                                                         training_mode=False, debug_path=DEBUG_PATH)

print(f"Script 4 creating images and labels in coco format done. Time taken: {time.time() - start:.1f}")
