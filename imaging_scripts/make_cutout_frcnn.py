# from astropy.nddata.utils import Cutout2D
import os
import pickle
import time
from sys import argv, path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

path.insert(0, os.environ['PROJECTPATH'])
from lib.subim import extract_subim
from lib.image_utils import find_bbox, remove_unresolved_sources_from_fits
from lib.label_utils import get_idx_dict
from lib.decision_tree_lib import return_nn_within_radius

# Create cutout directory in imagedirectory
imagedir = os.environ['TEMP_RESULTS']
UNRESOLVED_PATH = os.environ['LIKELY_UNRESOLVED_CATALOGUE']
CUTOUT_DIR = os.path.join(imagedir, 'cutouts')
os.makedirs(CUTOUT_DIR, exist_ok=True)

if __name__ == '__main__':

    assert len(argv) == 11, 'Script expects 11 input arguments'
    start = time.time()
    # Train/predict mode flag, 1 is train mode, 0 is predict mode
    training_mode = bool(int(argv[1]))
    # Import cut-out objects
    pickle_name = argv[2] + '.pkl'
    # Determines the number of cut-outs to be made from given input
    list_len = int(argv[3])
    # Overwrite flag, 1 is True, 0 is False
    overwrite = bool(int(argv[4]))
    print(f'Overwrite flag is set to {overwrite}')
    final_pixel_size = int(argv[6])
    dataset_name = argv[7]
    rotation = bool(int(argv[8]))
    unresolved_threshold = str(argv[9])
    remove_unresolved = bool(int(argv[10]))
    print("Start prepro script 2/4.")
    print("Path to file with names of unresolved objects:", UNRESOLVED_PATH)
    print("Remove unresolved sources is set to:", remove_unresolved)
    dataset_dir = os.path.join(imagedir, dataset_name)


    # Checks to see if a fixed box size is given or not
    if argv[5] is 0:
        fixed_box_size = False
        print('========================================')
        print('Using variable cutout size of {} arcseconds.'.format(fixed_box_size))
        print('========================================')
        raise NotImplementedError("Variable cutout size is not implemented.")
    else:
        fixed_box_size = int(argv[5])
        print('========================================')
        print('Using fixed cutout size of {} arcseconds.'.format(fixed_box_size))
        if training_mode:
            print(f'Or {int(fixed_box_size * np.sqrt(2))} before rotation&cropping.')
        print('========================================')

    # Annotated PyBDSF table, used for finding other sources
    if training_mode:
        Thisbelowisstillinfullcatmode
        compcat = pd.read_hdf(os.environ['LOTSS_COMP_CATALOGUE'], 'df')
        lt = pd.read_hdf(os.environ['LOTSS_RAW_CATALOGUE'], 'df')
        # comp_name_to_source_name_dict = {n:i for i,n in zip(compcat.Source_Name.values,
        #                                                    compcat.Component_Name.values)}
    else:
        #try: 
        #    DR1_testset_inference = os.environ['DR1_TESTSET_INFERENCE']
        #    lt = pd.read_hdf(os.environ['LOTSS_RAW_CATALOGUE'], 'df')
        #except:
        #    lt = pd.read_hdf(os.environ['LOTSS_RAW_CATALOGUE_DR2'], 'df')
        #compcat = lt
        compcat = pd.read_hdf(os.path.join(os.getenv('TEMP_RESULTS'),os.getenv('SRL_NAME').replace('.fits','.h5')))

    # Load catalogue containing likely unresolved sources to remove them later
    if os.path.exists(UNRESOLVED_PATH):
        if UNRESOLVED_PATH.endswith('.h5'):
            unresolved_cat = pd.read_hdf(UNRESOLVED_PATH)
        else:
            raise Exception("Unexpected file type, expected hdf5:", UNRESOLVED_PATH)
        unresolved_dict = {s: p for s, p in
                           zip(unresolved_cat.index.values, unresolved_cat[unresolved_threshold].apply(
                               lambda x: bool(x)).values)}
        # Load gaussian component cat
        if training_mode:
            gauss_cat = pd.read_hdf(os.environ['LOTSS_GAUSS_CATALOGUE'])
        else:
            print("Loading DR2 gaussian catalogue")
            #gauss_cat = pd.read_hdf(os.environ['LOTSS_GAUSS_CATALOGUE_DR2'])
            gauss_cat = pd.read_hdf(os.path.join(os.getenv('TEMP_RESULTS'),os.getenv('GAUS_NAME').replace('.fits','.h5')))

        # Turn Gauss cat into dict
        gauss_dict = {s: [] for s in gauss_cat['Source_Name'].values}
        for s, idx in zip(gauss_cat['Source_Name'].values, gauss_cat.index):
            gauss_dict[s].append(idx)

    else:
        print("Catalogue containing likely unresolved sources expected here:", UNRESOLVED_PATH,
              "does not exist. Removing likely unresolved sources will therefore be skipped.")
        unresolved_dict = None
    # Initialize coordinate dict 
    idx_dict = get_idx_dict(compcat, training_mode=training_mode)


    # load fields
    fload = np.load(os.path.join(dataset_dir, 'fields.npz'))
    field_names, local_field_folders, field_paths, field_cats = fload['fname'], fload['ffolder'], \
                                                                fload['fpath'], fload['fcat']
    print("We will be iterating over the following fields:", field_names)

    # loop over fields
    cutout_n = 0

    for field_name, local_field_folder, field_path in zip(field_names,
                                                          local_field_folders, field_paths):
        if cutout_n >= list_len:
            break

        print(f'Opening {pickle_name} in field {field_name}')
        with open(os.path.join(imagedir, pickle_name), 'rb') as input:
            l = pickle.load(input)

        local_count = 0
        for counter_index, s in enumerate(l):

            if cutout_n >= list_len:
                break

            sourcename = f'{s.c_source.sname}'
            # print(f'Source index: {cutout_n} (within this field {local_count}:{len(l)-1}) - {sourcename}')

            ## Settings names for infrared cut-out
            # irimage = sourcename+'_IR.png'
            # infrared_numpy_name = sourcename+'_IR.npy'
            # infrared_numpy_path = os.path.join(CUTOUT_DIR,sourcename+'_IR.npy')

            # Only proceed if cutout does not exist yet or if overwrite == 1
            # if os.path.exists(os.path.join(CUTOUT_DIR,sourcename+'_radio.fits')) and \
            #        os.path.exists(os.path.join(CUTOUT_DIR,irimage)) and \
            #        os.path.exists(infrared_numpy_path) and overwrite == 0:
            if os.path.exists(os.path.join(CUTOUT_DIR, sourcename + '_radio_DR2.fits')) and \
                    not overwrite:
                print('Skipping: {} exists and overwrite is False'.format(irimage))
                cutout_n += 1
                local_count += 1
                continue

            try:
                remote = bool(int(os.environ['REMOTE_IMAGES']))
            except:
                remote = False
            if remote:
                lofarname_DR2_0h = os.path.join(os.environ['MOSAICS_PATH_DR2'], "RA0h_field", s.lofarname_DR2)
                lofarname_DR2_13h = os.path.join(os.environ['MOSAICS_PATH_DR2'], "RA13h_field", s.lofarname_DR2)
                if os.path.exists(lofarname_DR2_0h):
                    lofarname_DR2 = lofarname_DR2_0h
                    lofarrms_DR2 = lofarname_DR2_0h.replace('-blanked.fits', '.fits').replace('.fits', '.rms.fits')
                else:
                    lofarname_DR2 = lofarname_DR2_13h
                    lofarrms_DR2 = lofarname_DR2_13h.replace('-blanked.fits', '.fits').replace('.fits', '.rms.fits')
            # elif augmented_flux:
            #    lofarname_DR2_0h = os.path.join(os.environ['MOSAICS_PATH_DR2'],
            #            "RA0h_field",s.lofarname_DR2,"augmented")
            #    lofarname_DR2_13h = os.path.join(os.environ['MOSAICS_PATH_DR2'],"RA13h_field",
            #            s.lofarname_DR2, "augmented")
            #    if os.path.exists(lofarname_DR2_0h):
            #        lofarname_DR2 = lofarname_DR2_0h
            #        lofarrms_DR2 = lofarname_DR2_0h.replace('-blanked.fits','.rms.fits')
            #    else:
            #        lofarname_DR2 = lofarname_DR2_13h
            #        lofarrms_DR2 = lofarname_DR2_13h.replace('-blanked.fits','.rms.fits')
            else:
                lofarname_DR2 = os.path.join(os.environ['LOCAL_MOSAICS_PATH_DR2'], s.lofarname_DR2)
                lofarrms_DR2 = os.path.join(os.environ['LOCAL_MOSAICS_PATH_DR2'],
                                            s.lofarname_DR2.replace('-blanked.fits', '.fits').replace('.fits',
                                                                                                      '.rms.fits'))
            # wisename = s.wisename

            # Everything under if statement used to vary box size to accomodate interesting neighbours
            if fixed_box_size == 0:

                # Repeat nn_search_attempt times
                ras, decs = lt.RA, lt.DEC
                old_ra, old_dec = s.RA, s.DEC,
                for j in range(nn_search_attempts):

                    # Gather all sources within nn_search_radius_arcsec
                    indices = return_nn_within_radius(ras, decs, old_ra, old_dec,
                                                      nn_search_radius_arcsec)[0]

                    # Set new center based on their mean RA and DEC
                    new_ra = np.mean(lt.loc[indices].RA.values)
                    new_dec = np.mean(lt.loc[indices].DEC.values)
                    if old_ra == new_ra and old_dec == new_dec:
                        new_ras[i] = new_ra
                        new_decs[i] = new_dec
                        break
                    else:
                        old_ra = new_ra
                        old_dec = new_dec

                # Find bounding box for the sources 
                ra, dec, size_arcsec = find_bbox(lt.loc[indices])

                # Ensuring that the cutout stays within certain smits
                if np.isnan(size_arcsec):
                    ra, dec = source.RA, source.DEC
                    size_arcsec = 60

                if size_arcsec < 60:
                    size_arcsec = 60

                if size_arcsec > 300.0:
                    # revert just to original
                    ra, dec = source.RA, source.DEC
                    size_arcsec = 300.0
                    ra, dec, size_arcsec = find_bbox([source])
                size_degree = size_arcsec / 3600

            else:
                # All commands for a fixed box size
                # Determine box dimensions
                ra, dec = s.RA, s.DEC,
                size_degree = fixed_box_size / 3600.0
            s.set_size(size_degree)
            if training_mode:
                size_degree *= np.sqrt(2)

            # Calculating bounds based on image
            hdu = fits.open(lofarname_DR2)
            psize = int(size_degree / hdu[0].header['CDELT2'])
            wcs = WCS(hdu[0].header)
            ndims = hdu[0].header['NAXIS']
            pvect = np.zeros((1, ndims))
            pvect[0][0] = ra
            pvect[0][1] = dec
            # the origin should be 1 for FITS files
            imc = wcs.wcs_world2pix(pvect, 1)
            imc0 = imc - psize / 2
            skyc0 = wcs.wcs_pix2world(imc0, 1)
            imc1 = imc + psize / 2
            skyc1 = wcs.wcs_pix2world(imc1, 1)

            min_RA, min_DEC = skyc1[0][0], skyc0[0][1]
            max_RA, max_DEC = skyc0[0][0], skyc1[0][1]

            s.set_bounds_degrees(min_RA, max_RA, min_DEC, max_DEC)
            newly_created = False
            if remove_unresolved:
                save_appendix = f'_{fixed_box_size}arcsec_large_radio_DR2_removed.fits'
                save_appendix_rms = f'_{fixed_box_size}arcsec_large_radio_rms_DR2_removed.fits'
            else:
                save_appendix = f'_{fixed_box_size}arcsec_large_radio_DR2.fits'
                save_appendix_rms = f'_{fixed_box_size}arcsec_large_radio_rms_DR2.fits'

            cutout_path = os.path.join(CUTOUT_DIR, sourcename + save_appendix)
            if not os.path.exists(cutout_path) or overwrite:  # or overwrite:
                lhdu = extract_subim(lofarname_DR2, ra, dec, size_degree, verbose=False)
                lhdu.writeto(cutout_path, overwrite=overwrite)
                newly_created = True
                try:
                    lhdu = extract_subim(lofarname_DR2, ra, dec, size_degree, verbose=False)
                    lhdu.writeto(cutout_path, overwrite=overwrite)
                    newly_created = True
                except:
                    print(
                        'Warning: Failed to make DR2 cut-out. File path may have changed or the required position is not on the map. Proceding without.')
            else:
                lhdu = fits.open(cutout_path)

            # LOFAR - DR2 rms
            rms_path = os.path.join(CUTOUT_DIR, sourcename + save_appendix_rms)
            # print("final cutoutsize:", final_pixel_size, size_degree,'overwrite', overwrite)
            if not os.path.exists(rms_path) or overwrite:  # or overwrite:
                try:
                    lrms = extract_subim(lofarrms_DR2, ra, dec, size_degree, verbose=False)
                    lrms.writeto(rms_path, overwrite=overwrite)

                except:
                    print(
                        'Warning: Failed to make DR2 rms cut-out. File path may have changed or the required position is not on the map. Proceding without.')
            else:
                lrms = fits.open(rms_path)

            # Saving beam
            s.set_beam(wcs, lhdu)
            cutout_hdu = lhdu.pop(0)
            rms_hdu = lrms.pop(0)
            try:
                s.set_scale_factor(final_pixel_size, cutout_hdu.shape[0])
            except:
                print(s.c_source.sname, "has an unwanted cutout shape:", cutout_hdu.shape[0],
                      "and will therefore be skipped.")
                continue
            if not final_pixel_size <= rms_hdu.shape[0]:
                print(s.c_source.sname, "has an unwanted rms shape:", rms_hdu.shape[0],
                      "and will therefore be skipped.")
                continue
            cutout_wcs = WCS(cutout_hdu.header)

            # Select all sources which fall within the bounds of the box
            box_dim = (compcat['RA'] >= min_RA) & (compcat['RA'] <= max_RA) \
                      & (compcat['DEC'] >= min_DEC) & (compcat['DEC'] <= max_DEC)
            # Get accompanying value added catalogue datarow
            compcat_subset = compcat[box_dim]
            # Filter out central component
            if training_mode:
                compcat_subset = compcat_subset[[s.c_source.sname != n
                                                 for n in compcat_subset.Component_Name.values]]
            else:
                compcat_subset = compcat_subset[[s.c_source.sname != n
                                                 for n in compcat_subset.Source_Name.values]]
            # Add other radio components to cutout object
            s.save_other_components(cutout_wcs, idx_dict, compcat_subset,
                                    unresolved_dict, remove_unresolved=remove_unresolved, training_mode=training_mode)
            """
            print('focus, related, unrelated pixel locs:', s.c_source.sname)
            print(s.get_focussed_comp())
            print(s.get_related_comp())
            print(s.get_unrelated_comp())
            """
            # Remove unresolved sources
            if remove_unresolved and newly_created:
                # Remove unresolved sources from fits file
                remove_unresolved_sources_from_fits(s, os.path.join(CUTOUT_DIR,
                                                                    sourcename + save_appendix), gauss_cat, gauss_dict)
                """
                # Remove unresolved sources from cutout object
                unwanted_idxs = [s_idx for s_idx, s_val in enumerate(s.other_components)
                        if s_val.unresolved]
                s.remove_components(unwanted_idxs)
                """

            # Calculate the convex hull
            s.c_source.set_convex_hull(compcat, idx_dict, training_mode=training_mode)
            [oc.set_convex_hull(compcat, idx_dict, training_mode=training_mode)
             for oc in s.other_components]

            """
            # Making numpy cut-out
            radio_x, radio_y = np.shape(lhdu[0].data)
            wcs_radio = WCS(lhdu[0].header)
            #wcs_infrared = WCS(whdu[0].header)
            # Go from radio central pixel to skycoord
            radio_skycoord = SkyCoord.from_pixel(radio_x/2, radio_y/2, wcs_radio) 
            # Go from skycoord to IR pixel 
            infrared_x, infrared_y = radio_skycoord.to_pixel(wcs_infrared) 
            # Create infrared cutout that matches the radio fits file
            #infrared_cutout = Cutout2D(whdu[0].data, (infrared_x, infrared_y), (radio_x, radio_y))
            # Save infrared to numpy array
            print('INFRARED cutout shape:', np.shape(infrared_cutout.data), 'radio cutout shape:', radio_x,radio_y)
            if infrared_cutout != np.array(None):
                np.save(infrared_numpy_path, infrared_cutout.data)
            
            
            #if isinstance(tcopy, pd.Series):
            #    print('Writing to file')
            #    print(sourcename,tcopy['Source_Name'].decode(),file=outfile)
            #else:
            #    print('Writing to file')
            #    for i in range(len(tcopy)): 
            #        print(sourcename,tcopy['Source_Name'][i],file=outfile)

            try:
                peak==r['Peak_flux']/1000.0
            except:
                peak=None

            # Making infrared cutout
            # To do: Pull function into this script? Rest is not necessary 
            show_overlay(lhdu,whdu,ra,dec,size_degree,lw=1,save_name=os.path.join(CUTOUT_DIR,irimage),title=None,peak=peak,noisethresh=0,show_lofar=False)
            """
            cutout_n += 1
            local_count += 1
            """
            if counter_index ==1:
                np.save('/data/mostertrij/lofar_frcnn_tools/keys2_cutout.npy', list(s.__slots__))
            """

        print(f'Processed {len(l)} sources in this field.')
        with open(os.path.join(imagedir, pickle_name), 'wb') as output:
            pickle.dump(l, output, pickle.HIGHEST_PROTOCOL)

    # end of cut-out for loop
    print(f'Script 2 Making cutouts done. Time taken: {time.time() - start:.1f} sec.\n\n')
