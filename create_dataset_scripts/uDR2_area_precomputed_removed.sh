#!/bin/bash
#-----------------------------------------------------------------------------------
#These scripts are used for the following purposes:
#

#NoteL The sample_list is created by saving a subselection of sources from the LOFAR value added catalog.
#See 'sample_list_create.py' for an example on how to do this. 
#-----------------------------------------------------------------------------------
#Set paths (change if necessary):

DATASET_NAME='LB300_Spring-60-65-corrected_precomputed_removed'
SAMPLE_LEN=100
SAMPLE_LIST='LB300_Spring-60-65-corrected_precomputed_removed'
N_FIELDS=1 # Number of fields to include. Set to 1e9 to include all fields
ROTATION=0 # 0 is False, 1 is True
single_comp_rotation_angles_deg='25,50,100'
multi_comp_rotation_angles_deg='25,50,100' #,105,125,145,165,185,205'
FIXED_CUTOUT_SIZE=1 # 0 is False, 1 is True
CUTOUT_SIZE_IN_ARCSEC=300 # Size of cut-out in DEC, will be slightly larger in RA (due to square cut-out)
RESCALED_CUTOUT_SIZE_IN_PIXELS=200
POINTING_NAME='mosaic-blanked' # Excluding the .fits extension
EDGE_CASES=0 # 0 is False, 1 is True
BOX_SCALE=1 #Constant with which the bounding box will be scaled
DEBUG=1 # 0 is False, 1 is True
SIG5_ISLAND_SIZE=1 # Number of pixels that need to be above 5 sigma for component to be detected
INCLUDE_LOW_SIG=0 # Determines wether low sigma sources are also labeled
ALLOW_OVERLAP=1 # Determines if we allow overlaping bounding boxes or not
INCL_DIFF=1 # Determines if difficult sources are allowed or not
DIFFICULT_LIST_NAME='difficult_1000.txt'
CLIP=1 # Determines if the cut-outs are clipped or not
CLIP_LOW=1 # lower clip value (sigma)
CLIP_HIGH=30 # upper clip value (sigma)
SIGMA_BOX_FIT=5 # Region to fit bounding box to
MUST_BE_LARGE=1 # Require sources to be > 15 arcsec or not? [0 is False, 1 is True]
MUST_BE_BRIGHT=1 # Require sources to have total flux > 10mJy or not? [0 is False, 1 is True]
UNRESOLVED_THRESHOLD=0.20
REMOVE_UNRESOLVED=1
TRAINING_MODE=1 
PRECOMPUTED_BBOXES=1 # 0 is False; 1 is True
OVERWRITE=1 # 0 is False, 1 is True

export PROJECTPATH=/data2/mostertrij/lofar_frcnn_tools
#export IMAGEDIR=/data2/mostertrij/data/frcnn_images # Where the folders with datasets will end up
export IMAGEDIR=/data2/mostertrij/data/runtest # Where the folders with datasets will end up
export CACHE_PATH=/data2/mostertrij/data/cache # Cache
export MOSAICS_PATH_DR2=/disks/paradata/shimwell/LoTSS-DR2/mosaics
export LOCAL_MOSAICS_PATH_DR2=/data2/mostertrij/data/LoTSS_DR2
export MOSAICS_PATH_DR1=/data2/mostertrij/pink-basics/data_LoTSS_DR1
export LOTSS_RAW_CATALOGUE=/data2/mostertrij/data/catalogues/LoTSS_DR2_v100.srl.h5 #LOFAR_HBA_T1_DR1_catalog_v1.0.srl.h5
export LOTSS_GAUSS_CATALOGUE=/data2/mostertrij/data/catalogues/LoTSS_DR2_v100.gaus.h5
export LOTSS_COMP_CATALOGUE=/data2/mostertrij/data/catalogues/Spring-60-65/LoTSS_DR2_corrected_cat.comp.h5 #LoTSS_DR1_corrected_cat.comp.h5
export LOTSS_SOURCE_CATALOGUE=/data2/mostertrij/data/catalogues/Spring-60-65/LoTSS_DR2_corrected_merged.h5 #LoTSS_DR1_corrected_merged.h5
export LOTSS_RAW_CATALOGUE_DR2=/data2/mostertrij/data/catalogues/LoTSS_DR2_v100.srl.h5
export LIKELY_UNRESOLVED_CATALOGUE=/data2/mostertrij/data/catalogues/GradientBoostingClassifier_A1_31504_18F_TT1234_B1_exp3_DR2_slim.h5
export OPTICAL_CATALOGUE=/data1/mostertrij/dr2_combined.db
export PATH=/soft/Montage_v3.3/bin:$PATH

#1 - Make a source list: 
# Go through decision tree (fig. 5 in Williams et al 2018) and select sources that are large and
# bright
# Where n is the number of sources that you want in the list and 'list_name.fits' is the name of the 
# fits file that you want to produce. Change paths as needed. Edit script for different selection criteria.
python $PROJECTPATH/imaging_scripts/multi_field_decision_tree.py $TRAINING_MODE $SAMPLE_LEN \
$SAMPLE_LIST 1 $DATASET_NAME $N_FIELDS $MUST_BE_LARGE $MUST_BE_BRIGHT #&> logs/uit1_LB300_Spring-60-65-corrected_precomputed_removed.txt

#2 - Generate cut-outs using: 
#(where N and M are the index range)
#Note: The images will be placed in the directory from which you run the commands
python -W ignore $PROJECTPATH/imaging_scripts/make_cutout_frcnn.py $TRAINING_MODE $SAMPLE_LIST $SAMPLE_LEN \
    $OVERWRITE $CUTOUT_SIZE_IN_ARCSEC $RESCALED_CUTOUT_SIZE_IN_PIXELS $DATASET_NAME $ROTATION \
    $UNRESOLVED_THRESHOLD $REMOVE_UNRESOLVED #&> logs/uit2_LB300_Spring-60-65-corrected_precomputed_removed.txt


#3 - Determine box size and component numbers using: 
python $PROJECTPATH/labeling_scripts/labeler_rotation.py $TRAINING_MODE $SAMPLE_LIST $OVERWRITE \
    $EDGE_CASES $BOX_SCALE $DEBUG $SIG5_ISLAND_SIZE $INCLUDE_LOW_SIG $ALLOW_OVERLAP $INCL_DIFF \
   $DATASET_NAME $single_comp_rotation_angles_deg $multi_comp_rotation_angles_deg $ROTATION \
    $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES \
    $UNRESOLVED_THRESHOLD $REMOVE_UNRESOLVED $SIGMA_BOX_FIT #&> logs/uit3_LB300_Spring-60-65-corrected_precomputed_removed.txt

#echo """
#4 - Create labels in XML format and structure data in correct folder hierarchy
python $PROJECTPATH/labeling_scripts/create_and_populate_initial_dataset_rotation.py \
   $FIXED_CUTOUT_SIZE $INCL_DIFF $DIFFICULT_LIST_NAME $CLIP $CLIP_LOW $CLIP_HIGH \
   $DATASET_NAME $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES \
   $TRAINING_MODE $REMOVE_UNRESOLVED $UNRESOLVED_THRESHOLD $SIGMA_BOX_FIT #&> logs/uit4_LB300_Spring-60-65-corrected_precomputed_removed.txt
#echo """
#"""
