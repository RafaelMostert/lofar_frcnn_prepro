"""
Given that we remove out of bound (out of cutout field of view)
components from the source objects, we need to penalize our catalogue
accuracies for the sources with components that fall outside the bound.
Throughout, 'comp' is an abbreviation of 'component'
"""
import os
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from copy import deepcopy
from time import time
from astropy.coordinates import SkyCoord
from astropy.visualization import PercentileInterval
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs import utils
from collections import Counter
from label_library import load_fits


def non_unique(l):
    c = Counter(l)
    return [x for x in l if c[x] > 1]

class OutOfFOVTally(object):
    def __init__(self,
            dataset_path='/data2/mostertrij/data/frcnn_images/uLB300_precomputed_removed',
            gt_comp_cat_path='/data2/mostertrij/data/catalogues/LoTSS_DR1_corrected_cat.comp.h5',
            imsize_pixels=200,
            sourcename_key='Source_Name',
            fits_dir_path='/data2/mostertrij/data/frcnn_images/cutouts',
            fits_files_suffix='_300arcsec_large_radio_DR2.fits'):

        self.datasetname=dataset_path.split('/')[-1]
        self.sourcename_key=sourcename_key
        self.json_annotation_dir_path=os.path.join(dataset_path,'LGZ_COCOstyle/annotations')
        self.gt_comp_cat = pd.read_hdf(gt_comp_cat_path)
        self.imsize_pixels=imsize_pixels
        self.fits_dir_path=fits_dir_path
        self.fits_files_suffix=fits_files_suffix
        self.save_path=dataset_path
        save_path_txt = os.path.join(dataset_path,'out_of_fov_tally.txt')
        self.f = open(save_path_txt,'w')
        print(f"Working on dataset:", self.datasetname)
        print("Saving output to:", self.save_path)
        self.datasets=['train','val','test']
        # Execute main target
        print(f"List of components that fall outside FOV for dataset {self.datasetname}",
                file=self.f)
        self.tally_out_of_FOV()
        

    def get_relevant_compnames(self):
        # Get final component names from json annotation files

        # Some datasets only have train or test/val data, so limit ourselves to the existing sets
        new_datasets=[]
        for dataset in self.datasets:
            if os.path.exists(os.path.join(self.json_annotation_dir_path,f'VIA_json_{dataset}.pkl')):
                new_datasets.append(dataset)
        self.datasets = new_datasets

        self.annotations = [pd.read_pickle(
            os.path.join(self.json_annotation_dir_path,f'VIA_json_{dataset}.pkl')) 
            for dataset in self.datasets]
        self.relevant_compnames_list = [[ann['file_name'].split('/')[-1].split('_')[0]
                for ann in annotation_subset 
                if ann['file_name'].endswith('rotated0deg.png')] # for annotation in annotations 
                for annotation_subset in self.annotations] # for annotations in train, val, test
        self.relevant_compnames_flattened = [item for sublist in self.relevant_compnames_list 
                for item in sublist]
        # Informative prints
        for dataset_i, dataset in enumerate(self.datasets):
            print(f"Retrieved {len(self.relevant_compnames_list[dataset_i])} components "
                    f"in the {dataset} set (the pickled annotation-JSON).",
                file=self.f)
            print(f"Retrieved {len(self.relevant_compnames_list[dataset_i])} components "
                f"in the {dataset} set (the pickled annotation JSON).")


    def get_relevant_sourcenames(self):
        # Use gt comp cat to link compnames to sourcenames
        # Note: each compname exists at most once
        non_unique_compnames = non_unique(self.gt_comp_cat.Component_Name.values)
        print(f"Note: There are {len(non_unique_compnames)} non-unique compnames in the gt_comp_cat.")
        print("Those will only give a single compname:sourcename pair (possibly erroneously).")
        self.gt_compname_to_sourcename_dict = {compname:sourcename for
                compname,sourcename in zip(self.gt_comp_cat.Component_Name.values,
                    self.gt_comp_cat[self.sourcename_key].values)}

        # Get relevant sourcenames per dataset (train, val, test)
        self.relevant_sourcenames_list = [[self.gt_compname_to_sourcename_dict[relevant_compname]
            for relevant_compname in relevant_compnames]
                for relevant_compnames in self.relevant_compnames_list]
        self.relevant_sourcenames_flattened = [item for sublist in self.relevant_sourcenames_list 
                for item in sublist]


    def link_sourcenames_to_comp_coordinates(self):
        # Use gt comp cat to link sourcenames to its comp-coords

        # initialize dict
        # Note:[self.sourcename_key] list contains duplicates, but python dicts will only retain a single
        # unique key per unique name (so similar to using set(bla[self.sourcename_key].values))

        # initialize dicts
        self.gt_sourcename_to_compcoords_dict = {sourcename:[] for
                sourcename in self.gt_comp_cat[self.sourcename_key].values}
        self.relevant_sourcename_to_compcoords_dict = {sourcename:[] for
                sourcename in self.relevant_sourcenames_flattened}
        self.relevant_sourcename_to_skycoords_dict = deepcopy(self.relevant_sourcename_to_compcoords_dict)
        # Fill dict with coords
        for sourcename,ra,dec in zip(self.gt_comp_cat[self.sourcename_key].values,
                    self.gt_comp_cat.RA.values, self.gt_comp_cat.DEC.values):
            self.gt_sourcename_to_compcoords_dict[sourcename].append((ra,dec))
        # Only keep the relevant sourcename entries
        for sourcename in self.relevant_sourcenames_flattened:
            self.relevant_sourcename_to_compcoords_dict[sourcename].append(self.gt_sourcename_to_compcoords_dict[sourcename])

        # Convert coordinates to skycoords
        start = time()
        failcount=0
        for sourcename, coords in self.relevant_sourcename_to_compcoords_dict.items():
            if coords != []:

                ras,decs = np.array(coords[0]).T
                skycoordlist = SkyCoord(ras,decs,unit='deg')
                self.relevant_sourcename_to_skycoords_dict[sourcename]=skycoordlist
            else:
                failcount+=1
        print(f"Converted coords to skycoords. Failcount {failcount}. Time taken: {time()-start:.2f}")


    def check_if_coords_fall_outside_imsize(self):

        impath = os.path.join(self.save_path,'out_of_bound_debug_images')
        os.makedirs(impath,exist_ok=True)
        imsize_pixels =self.imsize_pixels
        m = 50
        print(f"Assuming the fits cutouts are {2*m +imsize_pixels} pixels wide.")
        print(f"And assuming the final image are {imsize_pixels} pixels wide.")
        # For each train/val/test subset
        for dataset,relevant_compnames in zip(self.datasets,self.relevant_compnames_list):

            print(f"The following component-prediction in the {dataset} set "
                    f"contains out of bound components:",
                file=self.f)
            print(f"Checking {dataset} set on out of bound components.")

            # For each relevant component
            out_of_bound_compnames = []
            for relevant_compname in relevant_compnames:
                out_of_bound_flag=False
                # Load corresponding fits cutouts
                fits_path = os.path.join(self.fits_dir_path,
                        relevant_compname+self.fits_files_suffix)
                hdu, hdr = load_fits(fits_path, dimensions_normal=True)
                wcs=WCS(hdr, naxis=2)

                # Get corresponding relevant sourcename
                relevant_sourcename = self.gt_compname_to_sourcename_dict[relevant_compname]

                # Get corresponding relevant skycoords
                relevant_coords = self.relevant_sourcename_to_compcoords_dict[relevant_sourcename]
                #print("relevant coords,", relevant_coords)
                relevant_skycoords = self.relevant_sourcename_to_skycoords_dict[relevant_sourcename]
                #print("relevant skycoores,", relevant_skycoords)
                # Transform skycoords to pixellocations
                xs, ys = utils.skycoord_to_pixel(relevant_skycoords, wcs, 0)
                #print("xs",xs,"ys",ys)


                # Check if pixellocations are within imsize
                for x,y in zip(xs,ys):
                    if (not m <= x <= imsize_pixels+m) or (not m <= y <= imsize_pixels+m):
                        out_of_bound_flag=True

                if out_of_bound_flag:
                    out_of_bound_compnames.append(relevant_compname)
                    # Debug show
                    interval = PercentileInterval(99.)
                    image = interval(hdu)
                    plt.figure()
                    plt.imshow(image)
                    plt.plot([m,imsize_pixels+m,imsize_pixels+m,m,m],
                            [m,m,imsize_pixels+m,imsize_pixels+m,m],
                            linewidth=1,color='r')
                    plt.plot(xs,ys,'*',linestyle='none',color='red')
                    title = ''
                    for title_i, (x,y) in enumerate(zip(xs,ys)):
                        title+= f'{x:.0f};{y:.0f}|'
                        if title_i>4:
                            title+='\n'

                    plt.title(title)
                    this_image_path = os.path.join(impath,f'{relevant_compname}_percentile99.png')
                    plt.savefig(this_image_path,bbox_inches='tight')
                    plt.close('all')

                # If not write compname to file
                for out_of_bound_compname in out_of_bound_compnames:
                    print(out_of_bound_compname, file=self.f)
            print(f"Found {len(out_of_bound_compnames)} out of bound components.")

            # Save to npy for easy access
            npy_path = os.path.join(self.save_path,f'out_of_bound_{dataset}.npy')
            np.save(npy_path, out_of_bound_compnames)


    def tally_out_of_FOV(self):
        # Execute all class-functions that lead to the tallying of 
        # out of FOV components
        start = time()
        self.get_relevant_compnames()
        self.get_relevant_sourcenames()
        self.link_sourcenames_to_comp_coordinates()
        self.check_if_coords_fall_outside_imsize()
        print(f"Done. Time taken is {time()-start:.0f} sec.\n")




uLB300_viridis_out_of_tally = OutOfFOVTally(dataset_path='/data2/mostertrij/data/frcnn_images/uLB300_precomputed_removed_viridis')
uLB300_viridis1_to_30sigma_out_of_tally = OutOfFOVTally(dataset_path='/data2/mostertrij/data/frcnn_images/uLB300_precomputed_removed_viridis1_to_30sigma')
uLB300_out_of_tally_notRemoved = OutOfFOVTally(dataset_path='/data2/mostertrij/data/frcnn_images/uLB300_precomputed')
uLB300_out_of_tally = OutOfFOVTally(dataset_path='/data2/mostertrij/data/frcnn_images/uLB300_precomputed_removed')
uLF300_out_of_tally = OutOfFOVTally(dataset_path='/data2/mostertrij/data/frcnn_images/uLF300_precomputed_removed',
        fits_files_suffix='_300arcsec_large_radio_DR2_removed.fits')
uL300_out_of_tally = OutOfFOVTally(
        dataset_path='/data2/mostertrij/data/frcnn_images/uL300_precomputed_removed',
        fits_files_suffix='_300arcsec_large_radio_DR2_removed.fits')
uLB300_DR2_out_of_tally = OutOfFOVTally(
        dataset_path='/data2/mostertrij/data/frcnn_images/LB300_Spring-60-65-corrected_precomputed_removed',
        gt_comp_cat_path='/data2/mostertrij/data/catalogues/Spring-60-65/LoTSS_DR2_corrected_cat.comp.h5',
        sourcename_key='Parent_Source')
