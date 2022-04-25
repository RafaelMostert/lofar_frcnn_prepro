import os
import pickle
from sys import argv, path, exit

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel

path.insert(0, os.environ['PROJECTPATH'])
from lib.label_utils import label_maker, plotter, rotate_points
from lib.label_utils import return_multi_comp_compnames, label_maker_angle
from lib.image_utils import load_fits
from copy import deepcopy
from skimage.transform import rotate
from skimage.util import crop
import time
import sqlite3

DATA_PATH = os.environ['IMAGEDIR']
CACHE_DIR = os.environ['CACHE_PATH']
os.makedirs(CACHE_DIR, exist_ok=True)
# Create cutout directory if it does not exist yet
CUT_OUT_PATH = os.path.join(DATA_PATH, 'cutouts')
os.makedirs(CUT_OUT_PATH, exist_ok=True)

if __name__ == '__main__':

    print("argv has length", len(argv))
    assert len(argv) == 21, 'Script expects 20 input arguments'

    start = time.time()
    # Train/predict mode flag, 1 is train mode, 0 is predict mode
    training_mode = bool(int(argv[1]))
    # Overwrite flag, 1 is True, 0 is False
    overwrite = bool(int(argv[3]))
    print(f'Overwrite flag is set to {overwrite}')
    pickle_name_labeled = argv[2] + '_labeled.pkl'
    pickle_name_labeled_annotated = argv[2] + '_labeled_annotated.pkl'

    # Determine what to do with the edge-cases
    edge_cases = bool(int(argv[4]))
    print(f'Allow central source to not fit in cutout: {edge_cases}')
    # Determine the box-scale
    box_scale = float(argv[5])
    print(f'Box scale set to: {box_scale}')
    # Debug or not
    debug = bool(int(argv[6]))
    print(f'Debug set to: {debug}')
    # Sigma 5 island size
    sig5_isl = int(argv[7])
    print(f'Number of pixels in sigma five islands: {sig5_isl}')
    # Low sigma sources (below 5) toggle:
    low_sig = bool(int(argv[8]))
    print(f'Including sources with sigma lower than 5: {low_sig}')
    # Determining if overlapping bounding boxes are allowed:
    overlap_allowed = bool(int(argv[9]))
    print(f'Overlapping bounding boxes are allowed: {overlap_allowed}')
    # Include difficult sources or not
    incl_difficult = bool(int(argv[10]))
    print(f'Difficult sources included: {incl_difficult}')
    # dataset name
    dataset_name = argv[11]
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    DEBUG_PATH = os.path.join(DATA_PATH, dataset_name, 'label_debug_im_hull')
    os.makedirs(DEBUG_PATH, exist_ok=True)
    # rotation enabled?
    rotation = bool(int(argv[14]))
    if not training_mode:
        rotation = False
        print("Rotation is always disabled in inference mode.")
    print(f'Rotation enabled: {rotation}')
    imsize = int(argv[15])
    imsize_arcsec = int(argv[16])
    precomputed = bool(int(argv[17]))
    unresolved_threshold = argv[18]
    remove_unresolved = bool(int(argv[19]))
    sigma_box_fit = int(argv[20])
    print("Remove unresolved is:", remove_unresolved)
    segment_dir = os.path.join(CACHE_DIR, f'segmentation_maps_{unresolved_threshold}')
    os.makedirs(segment_dir, exist_ok=True)

    if not training_mode and not precomputed:
        print('In predict mode without precomputed: skip labelmaking.')
        exit()
    if training_mode:
        comp_cat = pd.read_hdf(os.environ['LOTSS_COMP_CATALOGUE'], 'df')
    else:
        comp_cat = pd.read_hdf(os.environ['LOTSS_RAW_CATALOGUE_DR2'], 'df')
    if 'Source_Name' in comp_cat.keys():
        optical_cat = pd.read_hdf(os.environ['OPTICAL_CATALOGUE'], 'df')
        DR2_flag = False
    else:
        print("Assuming DR2 optical catalogue that includes legacy instead of PAN-STARRS.")
        DR2_flag = True
    print(f'Dataset name: {dataset_name}')
    label_list_dir = os.path.join(DATA_PATH, dataset_name)
    print(f'final list saved at : {label_list_dir}')
    os.makedirs(label_list_dir, exist_ok=True)
    # Specify how and what rotated copies should be produced (counter-clockwise in degrees)
    # Do not include 0. non-rotation is always included. 
    print(argv[12].split(','))
    print(argv[13].split(','))
    single_comp_rotation_angles_deg = list(map(int, argv[12].split(',')))
    multi_comp_rotation_angles_deg = list(map(int, argv[13].split(',')))

    # load fields
    fload = np.load(os.path.join(label_list_dir, 'fields.npz'))
    field_names, field_folders, field_paths, field_cats = fload['fname'], fload['ffolder'], \
                                                          fload['fpath'], fload['fcat']
    print('Fields included in this dataset:', field_names)
    all_fields_list = []
    cutout_n = 0

    # Path to text file containing sourcenames that require visual inspection as LoTSS-DR2 files
    # are missing
    skipped_path_image_missing = os.path.join(label_list_dir, "requires_visual_inspection_lotssdr2_missing.txt")
    with open(skipped_path_image_missing, 'w') as f:
        print('Sources with the following Source_Name require visual inspection:', file=f)
    image_missing = 0
    # Path to text file containing sourcenames that require visual inspection as they are too faint
    skipped_path_faint = os.path.join(label_list_dir, "requires_visual_inspection_central_source_too_faint.txt")
    with open(skipped_path_faint, 'w') as f:
        print('Sources with the following Source_Name require visual inspection:', file=f)
    faint = 0

    # Initialize coordinate dict
    # coord_dict = get_coord_dict(comp_cat, training_mode=training_mode)
    # idx_dict = get_idx_dict(comp_cat, training_mode=training_mode)
    multi_comp_source_names = return_multi_comp_compnames(comp_cat, CACHE_DIR,
                                                          save_appendix=dataset_name, training_mode=training_mode)

    # Iterate over all the fields
    for field_name, field_folder, field_path, field_cat in zip(field_names,
                                                               field_folders, field_paths, field_cats):

        pickle_labeled_annotated_path = os.path.join(field_folder, pickle_name_labeled_annotated)

        # Load field specific source list
        if not os.path.exists(pickle_labeled_annotated_path) or overwrite:
            # Import cut-out objects
            pickle_name = argv[2] + '.pkl'
            pickle_path = os.path.join(field_folder, pickle_name)

            print(f'Opening {pickle_path}')
            with open(pickle_path, 'rb') as input:
                l = deepcopy(pickle.load(input))
            print(f'Field {field_name} containing {len(l)} cutouts.')

            # Create optical cat 
            if training_mode and not 'Source_Name' in comp_cat.keys():
                field_range_dict = np.load(os.path.join(dataset_path, 'field_ranges.npy'),
                                           allow_pickle=True).item()
                minra, maxra, mindec, maxdec = field_range_dict[field_name]
                print("DEBUG boundaries for this field RA and DEC:", minra, maxra, mindec, maxdec)
                con = sqlite3.connect(os.environ['OPTICAL_CATALOGUE'])
                sql = f'''SELECT * FROM OPTICAL 
                WHERE RA BETWEEN {minra} AND {maxra} AND
                DEC BETWEEN {mindec} AND {maxdec};'''
                start = time.time()
                optical_cat = pd.read_sql_query(sql, con)
                optical_cat.rename(columns={'RA': 'ra', 'DEC': 'dec'}, inplace=True)
                print(f"SQL query for all optical sources for this field took {time.time() - start:.2f} sec.")
                print(optical_cat.info())

            # l2 will contain all source in l and rotated copies
            l2 = []

            # loop over each cutout 
            # for i in range(len(l)):
            for i, lii in enumerate(l):
                sname = str(lii.c_source.sname)
                cutout_n += 1
                debug = False
                # if not sname in ['ILTJ123215.99+530444.2', 'ILTJ123527.84+531457.1']:
                #    continue
                # if not sname.startswith('ILTJ135332.18+534731.0'):
                #    continue
                list_skipped = []
                # Check for duplicates
                names_in_cutout = []
                for ss in lii.other_components:
                    names_in_cutout.append(ss.sname)
                assert len(set(names_in_cutout)) == len(names_in_cutout), (f'doubles',
                                                                           f' {[os.sname for os in lii.other_sources]}')

                # Opening DR2 radio data
                if remove_unresolved:
                    save_appendix = f'_{imsize_arcsec}arcsec_large_radio_DR2_removed.fits'
                    save_appendix_rms = f'_{imsize_arcsec}arcsec_large_radio_rms_DR2_removed.fits'
                else:
                    save_appendix = f'_{imsize_arcsec}arcsec_large_radio_DR2.fits'
                    save_appendix_rms = f'_{imsize_arcsec}arcsec_large_radio_rms_DR2.fits'

                if os.path.exists(os.path.join(CUT_OUT_PATH, sname + save_appendix)) and hasattr(lii,
                                                                                                 'crop_offset'):
                    data_DR2, hdr = load_fits(os.path.join(CUT_OUT_PATH,
                                                           sname + save_appendix), dimensions_normal=True)
                    # print(f'\"{os.path.join(CUT_OUT_PATH,sname+save_appendix)}\",')
                else:
                    print('Warning: No DR2 data found. Skipping cutout.')
                    data_DR2 = None
                    with open(skipped_path_image_missing, 'a') as f:
                        print(lii.c_source.sname, file=f)
                        image_missing += 1
                    continue

                if os.path.exists(os.path.join(CUT_OUT_PATH, sname + save_appendix_rms)):
                    rms_DR2 = fits.getdata(os.path.join(CUT_OUT_PATH, sname + save_appendix_rms))
                else:
                    print('Warning: No DR2 rms data found. Proceding while assuming constant rms.')
                    rms_DR2 = None
                """
                # Opening infrared data
                if os.path.exists(os.path.join(CUT_OUT_PATH, sname+'_IR.npy')):
                    infrared = np.load(os.path.join(CUT_OUT_PATH, sname+'_IR.npy'))
                    print('IR debug shape:', np.shape(infrared), type(infrared))
                else:
                    infrared = None
                    print('Warning: No infrared data found')
                """
                infrared = None

                # Calculates the centers of all the sources in the cut-out in pixel coordinates
                wcs = WCS(hdr)
                # lii.calculate_pixel_centers(wcs)

                # Create labels for each rotated source
                # In our LGZ_v2 training set there are 4278 single component sources and 1401 multi.
                # Thus 75% of the dataset is single component.
                if sname in multi_comp_source_names:
                    angle_degrees = multi_comp_rotation_angles_deg
                else:
                    angle_degrees = single_comp_rotation_angles_deg

                # loop_start = time.time()
                full_pixel_size = np.shape(data_DR2)[0]
                if training_mode:
                    plot_data_DR2 = np.flip(crop(data_DR2, lii.crop_offset), axis=0)
                    plot_rms_DR2 = np.flip(crop(rms_DR2, lii.crop_offset), axis=0)
                else:
                    plot_data_DR2 = np.flip(data_DR2, axis=0)
                    plot_rms_DR2 = np.flip(rms_DR2, axis=0)

                # add component pixel locations to the cutout
                # retrieve RA and DEC of sources that make up this value added source
                if training_mode:
                    cRA, cDEC = lii.c_source.ra, lii.c_source.dec
                    # print("cRA, cDEC:", cRA, cDEC)
                    # Get optical catalog entries for cutout
                    search_radius_arcsec = imsize_arcsec * 1.5
                    search_radius_degree = search_radius_arcsec / 3600
                    # if 'Source_Name' in comp_cat.keys():

                    condition = (optical_cat.ra < (cRA + search_radius_degree)) & \
                                (optical_cat.ra > (cRA - search_radius_degree)) & \
                                (optical_cat.dec < (cDEC + search_radius_degree)) & \
                                (optical_cat.dec > (cDEC - search_radius_degree))
                    optical_cat_cutout = optical_cat[condition]
                    # print("DEBUG nan values in W1, W2, R mag",
                    #        np.sum([np.isnan(v) for v in optical_cat_cutout.MAG_W1.values]),
                    #        np.sum([np.isnan(v) for v in optical_cat_cutout.MAG_W2.values]),
                    #        np.sum([np.isnan(v) for v in optical_cat_cutout.MAG_R.values]))
                    optical_coords = [[r, d] for r, d in zip(optical_cat_cutout.ra.values,
                                                             optical_cat_cutout.dec.values)]
                    # turn coordinates into x,y for this cutout
                    optical_pixel_locations = [[], []]
                    if len(optical_coords) > 0:
                        # optical_skycoords = SkyCoord([optical_coords], unit='deg')
                        optical_skycoords = \
                            SkyCoord(optical_cat_cutout.ra.values, optical_cat_cutout.dec.values, unit='deg')
                        optical_pixel_locations = skycoord_to_pixel(optical_skycoords, wcs, origin=0)
                    lii.set_optical_sources(optical_pixel_locations, optical_cat_cutout, imsize, data_DR2.shape[0],
                                            DR2=DR2_flag)

                # The first source needs to perform slightly more operations
                # like finding and setting the number of peaks 
                # Labeling the central source
                label_maker(lii, lii.c_source, box_scale, plot_data_DR2, plot_rms_DR2,
                            CUT_OUT_PATH, sig5_isl, wcs,
                            remove_unresolved, segment_dir,
                            training_mode=training_mode, sigma_box_fit=sigma_box_fit)
                if lii.c_source.low_sigma_flag > 0:
                    with open(skipped_path_faint, 'a') as f:
                        print(lii.c_source.sname + f', {lii.c_source.low_sigma_flag}', file=f)
                        faint += 1
                    continue

                # Label all other sources in the cutout
                for j in range(len(lii.other_components)):
                    label_maker(lii, lii.other_components[j], box_scale, plot_data_DR2,
                                plot_rms_DR2, CUT_OUT_PATH, sig5_isl, wcs,
                                remove_unresolved, segment_dir,
                                training_mode=training_mode, sigma_box_fit=sigma_box_fit)
                li = deepcopy(lii)

                # Remove unwanted sources
                """
                print("before removal focus, rel, unrel")
                print(lii.get_focussed_comp())
                print(lii.get_related_comp())
                print(lii.get_unrelated_comp())
                print([(s_val.in_bounds, s_val.low_sigma_flag) for s_idx, s_val in
                    enumerate(lii.other_components)])
                if lii.c_source.sname == 'ILTJ140638.08+552810.0':

                    print([(s_val.unresolved,hasattr(s_val, 'in_bounds'), s_val.in_bounds,
                        s_val.low_sigma_flag) 
                        for s_idx, s_val in enumerate(lii.other_components) if s_val.related])
                """

                unwanted_idxs = [s_idx for s_idx, s_val in enumerate(lii.other_components)
                                 if not hasattr(s_val, 'in_bounds') or not s_val.in_bounds or \
                                 (s_val.low_sigma_flag > 0 and not s_val.unresolved)]
                lii.remove_components(unwanted_idxs)

                """
                if lii.c_source.sname == 'ILTJ140638.08+552810.0':
                    print("For source:", lii.c_source.sname)
                    print(lii.get_related_comp())
                    print([s.unresolved for s in lii.other_components if s.related])
                print("focus, rel, unrel")
                print(lii.get_focussed_comp())
                print(lii.get_related_comp())
                print(lii.get_unrelated_comp())
                """
                if not overlap_allowed:
                    lii.box_overlap()

                # Make debug images
                if debug and training_mode:
                    plotter(lii, plot_data_DR2, plot_rms_DR2, infrared, DEBUG_PATH, edge_cases,
                            low_sig, overlap_allowed, incl_difficult, field_name, save=True,
                            save_appendix=f'rotated0deg')

                l2.append(lii)

                # If rotation is enabled, repeat the labelling process for rotated copies of the
                # cutouts
                if rotation:
                    for angle_deg in angle_degrees:
                        # Copy the source
                        ll = deepcopy(li)
                        ll.set_rotation_angle(angle_deg)

                        # Rotate data and rms
                        if data_DR2.dtype.byteorder == '>':
                            data_DR2 = data_DR2.byteswap().newbyteorder()
                        data_r = rotate(data_DR2, -angle_deg, resize=False, center=None,
                                        order=1, mode='constant', cval=0, clip=True,
                                        preserve_range=True)
                        full_pixel_size = np.shape(data_r)[0]
                        crop_amount = (full_pixel_size - lii.size_pixels) / 2
                        if crop_amount < 0:
                            continue
                        data_r = np.flip(crop(data_r, crop_amount), axis=0)
                        if rms_DR2.dtype.byteorder == '>':
                            rms_DR2 = rms_DR2.byteswap().newbyteorder()
                        rms_r = rotate(rms_DR2, -angle_deg, resize=False, center=None,
                                       order=1, mode='constant', cval=0, clip=True,
                                       preserve_range=True)
                        rms_r = np.flip(crop(rms_r, crop_amount), axis=0)

                        # Rotate focus, related and unrelated
                        focus = deepcopy(ll.get_focussed_comp())
                        related = deepcopy(ll.get_related_comp())
                        unrelated = deepcopy(ll.get_unrelated_comp())
                        focus_r = rotate_points(focus[0], focus[1],
                                                -angle_deg, (imsize / 2, imsize / 2))
                        # angle_deg, (data_DR2.shape[0]/2,data_DR2.shape[1]/2))
                        related_r = rotate_points(np.array(related[0]), np.array(related[1]),
                                                  -angle_deg, (imsize / 2, imsize / 2))
                        # angle_deg, (ll.size_pixels/2,ll.size_pixels/2))
                        unrelated_r = rotate_points(np.array(unrelated[0]),
                                                    np.array(unrelated[1]),
                                                    -angle_deg, (imsize / 2, imsize / 2))
                        # angle_deg, (ll.size_pixels/2,ll.size_pixels/2))

                        """
                        plt.plot(data.shape[0]/2,data.shape[1]/2,'g*')
                        plt.plot(cutout.unrelated_comp[0][0], cutout.unrelated_comp[1][0], 'k.')
                        plt.plot(unrelated_r[0][0], unrelated_r[1][0], 'r.')
                        #unrelated_r = translate_to_new_midpoint(unrelated_r[0], unrelated_r[1],
                        #        old_shape, data.shape)
                        #print("unrelated comps translated")
                        #print(unrelated_r)
                        #plt.plot(unrelated_r[0][0], unrelated_r[1][0], 'b.')
                        plt.gca().set_aspect('equal', adjustable='box')
                        plt.grid()
                        plt.xlim(0,200)
                        plt.ylim(0,200)
                        plt.show()

                        print("focus, rel, unrel")
                        print(lii.get_focussed_comp())
                        print(lii.get_related_comp())
                        print(lii.get_unrelated_comp())
                        """

                        ll.update_pixel_locs(focus_r, related_r, unrelated_r)

                        label_maker_angle(ll, ll.c_source, box_scale,
                                          data_r, rms_r, CUT_OUT_PATH, sig5_isl, wcs, angle_deg,
                                          remove_unresolved, sigma_box_fit=sigma_box_fit,
                                          training_mode=training_mode)

                        if ll.c_source.low_sigma_flag > 0:
                            print("Skipping rotated source as it is below our sigma threshold.")
                            continue

                        # Labels all other sources present in the image only if 
                        # the central source is in bounds with angle

                        for j in range(len(ll.other_components)):
                            label_maker_angle(ll, ll.other_components[j], box_scale,
                                              data_r, rms_r, CUT_OUT_PATH, sig5_isl, wcs, angle_deg,
                                              remove_unresolved, sigma_box_fit=sigma_box_fit,
                                              training_mode=training_mode)

                        # Remove unwanted sources
                        unwanted_idxs = [s_idx for s_idx, s_val in enumerate(ll.other_components)
                                         if not s_val.in_bounds or s_val.low_sigma_flag > 0]
                        ll.remove_components(unwanted_idxs)

                        if not overlap_allowed:
                            ll.box_overlap()

                        # Make debug images
                        if debug and training_mode:
                            plotter(ll, data_r, rms_r, infrared, DEBUG_PATH, edge_cases,
                                    low_sig, overlap_allowed, incl_difficult, field_name, save=True,
                                    angle_deg=angle_deg, save_appendix=f'rotated{angle_deg}deg')
                        l2.append(ll)
                # print('loop_end. time taken:', time.time()-loop_start)
                """
                if i==0:
                    np.save('/data/mostertrij/lofar_frcnn_tools/keys3_cutout.npy',
                            list(lii.__slots__))
                    np.save('/data/mostertrij/lofar_frcnn_tools/keys3_source.npy',
                            list(lii.c_source.__slots__))
                """
            with open(pickle_labeled_annotated_path, 'wb') as output:
                pickle.dump(l2, output, pickle.HIGHEST_PROTOCOL)
                print(f'Saved annotated labeled list to {pickle_labeled_annotated_path}')

        else:
            print(f'Found existing labeled list. Opening {pickle_labeled_annotated_path}')
            with open(pickle_labeled_annotated_path, 'rb') as input:
                l2 = pickle.load(input)
        all_fields_list += l2
        print("all_fields:", len(all_fields_list))

    with open(os.path.join(label_list_dir, 'labeled_annotated_cutouts.pkl'), 'wb') as output:
        pickle.dump(all_fields_list, output, pickle.HIGHEST_PROTOCOL)

    # make_list(all_fields_list,label_list_dir,'labeled_rotated_list.csv',edge_cases,low_sig,overlap_allowed,incl_difficult)

print(f"{image_missing} sources skipped because LoTSS-DR2 stokes I is missing.")
print(f"{faint} sources skipped because central source is too faint.")
print(f"Script 3 Done. Time taken: {time.time() - start:.1f}\n\n")
