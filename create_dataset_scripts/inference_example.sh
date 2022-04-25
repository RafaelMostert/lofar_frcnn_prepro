#!/bin/bash
#-----------------------------------------------------------------------------------
# Dataset creation for use in a Fast R-CNN
#-----------------------------------------------------------------------------------

# Change these variables to suit your needs
DATASET_NAME='inference_1field_DR2'
SAMPLE_LEN=999999999999999999999 # Limit number of sources to this number. Set very high to have no limit
N_FIELDS=1 # Number of fields to include. Set to 1e9 to include all fields
INCLUDE_LOW_SIG=1 # Determines whether low sigma sources are also labeled
REMOVE_UNRESOLVED=1 #  0 is False; 1 is True
MUST_BE_LARGE=1 # Require sources to be > 15 arcsec or not? [0 is False, 1 is True]
MUST_BE_BRIGHT=-1 # Require sources to have total flux > 10mJy or not? [0 is False, 1 is True]
#Change paths to match your setup
CATALOGUE_PATH=/data2/mostertrij/data/catalogues
export PROJECTPATH=/data2/mostertrij/lofar_frcnn_tools # location of project
export IMAGEDIR=/data2/mostertrij/data/frcnn_images # Where the folders with datasets will end up
export DEBUG_PATH=/data2/mostertrij/data/frcnn_images/$DATASET_NAME/debug # Where the folders with datasets will end up
export CACHE_PATH=/data2/mostertrij/data/cache # Cache
export MOSAICS_PATH_DR2=/disks/paradata/shimwell/LoTSS-DR2/mosaics
export LOCAL_MOSAICS_PATH_DR2=/data2/mostertrij/data/LoTSS_DR2
export MOSAICS_PATH_DR1=/data2/mostertrij/pink-basics/data_LoTSS_DR1
export LOTSS_RAW_CATALOGUE=$CATALOGUE_PATH/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.h5
export LOTSS_GAUSS_CATALOGUE=$CATALOGUE_PATH/LOFAR_HBA_T1_DR1_catalog_v0.99.gaus.h5
export LOTSS_COMP_CATALOGUE=$CATALOGUE_PATH/LoTSS_DR1_corrected_cat.comp.h5
export LOTSS_SOURCE_CATALOGUE=$CATALOGUE_PATH/LoTSS_DR1_corrected_merged.h5
export LOTSS_RAW_CATALOGUE_DR2=$CATALOGUE_PATH/LoTSS_DR2_v100.srl.h5
export LOTSS_GAUSS_CATALOGUE_DR2=$CATALOGUE_PATH/LoTSS_DR2_v100.gaus.h5
export LIKELY_UNRESOLVED_CATALOGUE=$CATALOGUE_PATH/GradientBoostingClassifier_A1_31504_18F_TT1234_B1_exp3_DR2.csv
export OPTICAL_CATALOGUE=$CATALOGUE_PATH/combined_panstarrs_wise.h5
export PATH=/soft/Montage_v3.3/bin:$PATH
export EXCLUDE_DR1_AREA=1

# Leave everything below this line untouched
####################################
TRAINING_MODE=0 
ROTATION=0 # 0 is False, 1 is True
single_comp_rotation_angles_deg='25,50,100' # Ignored when rotation==0
multi_comp_rotation_angles_deg='25,50,100' # Ignored when rotation==0
SAMPLE_LIST=$DATASET_NAME
FIXED_CUTOUT_SIZE=1 # 0 is False, 1 is True
CUTOUT_SIZE_IN_ARCSEC=300 # Size of cut-out in DEC, will be slightly larger in RA (due to square cut-out)
RESCALED_CUTOUT_SIZE_IN_PIXELS=200
ALLOW_OVERLAP=1 # Determines if we allow overlaping bounding boxes or not
INCL_DIFF=1 # Determines if difficult sources are allowed or not
EDGE_CASES=0 # 0 is False, 1 is True
DIFFICULT_LIST_NAME='difficult_1000.txt'
BOX_SCALE=1 #Constant with which the bounding box will be scaled
SIG5_ISLAND_SIZE=1 # Number of pixels that need to be above 5 sigma for component to be detected
UNRESOLVED_THRESHOLD=0.20 # Threshold to use for Alegre's desicion tree
PRECOMPUTED_BBOXES=1 # 0 is False; 1 is True
MUST_BE_LARGE=1
MUST_BE_BRIGHT=1
CLIP=1 # Determines if the cut-outs are clipped or not
CLIP_LOW=1 # lower clip value (sigma)
CLIP_HIGH=30 # upper clip value (sigma)
DEBUG=1 # 0 is False, 1 is True
OVERWRITE=1 # 0 is False, 1 is True
####################################


#1 - Make a source list: 
# Go through decision tree (fig. 5 in Williams et al 2018) and select sources that are large and
# bright
# Where n is the number of sources that you want in the list and 'list_name.fits' is the name of the 
# fits file that you want to produce. Change paths as needed. Edit script for different selection criteria.
python $PROJECTPATH/imaging_scripts/make_cutout_objects_given_decision_tree_output.py $TRAINING_MODE $SAMPLE_LEN \
    $SAMPLE_LIST 1 $DATASET_NAME $N_FIELDS $MUST_BE_LARGE \
    $MUST_BE_BRIGHT &> logs/1inference_$DATASET_NAME.txt


#2 - Generate cut-outs using: 
#(where N and M are the index range)
#Note: The images will be placed in the directory from which you run the commands
#echo """
python -W ignore $PROJECTPATH/imaging_scripts/make_cutout_frcnn.py $TRAINING_MODE $SAMPLE_LIST $SAMPLE_LEN \
    $OVERWRITE $CUTOUT_SIZE_IN_ARCSEC $RESCALED_CUTOUT_SIZE_IN_PIXELS $DATASET_NAME $ROTATION \
    $UNRESOLVED_THRESHOLD $REMOVE_UNRESOLVED &> logs/2inference_$DATASET_NAME.txt
#"""


#3 - Determine box size and component numbers using: 
#echo """
python $PROJECTPATH/labeling_scripts/labeler_rotation.py $TRAINING_MODE $SAMPLE_LIST $OVERWRITE \
    $EDGE_CASES $BOX_SCALE $DEBUG $SIG5_ISLAND_SIZE $INCLUDE_LOW_SIG $ALLOW_OVERLAP $INCL_DIFF \
   $DATASET_NAME $single_comp_rotation_angles_deg $multi_comp_rotation_angles_deg $ROTATION \
    $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES $UNRESOLVED_THRESHOLD \
        $REMOVE_UNRESOLVED  &> logs/3inference_$DATASET_NAME.txt
#"""

#4 - Create labels in XML format and structure data in correct folder hierarchy
#echo """
python $PROJECTPATH/labeling_scripts/create_and_populate_initial_dataset_rotation.py \
   $FIXED_CUTOUT_SIZE $INCL_DIFF $DIFFICULT_LIST_NAME $CLIP $CLIP_LOW $CLIP_HIGH \
   $DATASET_NAME $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES \
   $TRAINING_MODE $REMOVE_UNRESOLVED &> logs/4inference_$DATASET_NAME.txt
#"""

#sleep 30
#rsync -a /data2/mostertrij/data/frcnn_images/inference_$DATASET_NAME lgm4:/data1/mostertrij/data/frcnn_images
#rsync -a /data2/mostertrij/data/cache/segmentation_maps* lgm4:/data1/mostertrij/data/cache
