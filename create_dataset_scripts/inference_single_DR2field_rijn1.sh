#!/bin/bash
#-----------------------------------------------------------------------------------
# Dataset creation for use in a Fast R-CNN
#-----------------------------------------------------------------------------------

# Change these variables to suit your needs
DATASET_NAME='inference_P21_DR2_rijn1' 
SAMPLE_LEN=9999999999999999999999999999999999999 # Limit number of sources to this number. Set very high to have no limit
N_FIELDS=1 # Number of fields to include. Set to 1e9 to include all fields
INCLUDE_LOW_SIG=1 # Determines whether low sigma sources are also labeled
REMOVE_UNRESOLVED=1 #  0 is False; 1 is True
#Change paths to match your setup
BASE_PROJECT_PATH=/data1/intema/lofarpipeline
CATALOGUE_PATH=$BASE_PROJECT_PATH/Data/Catalogues/Pybdsf
export PROJECTPATH=$BASE_PROJECT_PATH/lofar_frcnn_prepro # location of project
export IMAGEDIR=$BASE_PROJECT_PATH/Data/frcnn_images # Where the folders with datasets will end up
export DEBUG_PATH=$BASE_PROJECT_PATH/frcnn_images/$DATASET_NAME/debug # Where the folders with datasets will end up
export CACHE_PATH=$BASE_PROJECT_PATH/Data/cache # Cache
#export MOSAICS_PATH_DR2=/disks/paradata/shimwell/LoTSS-DR2/mosaics
#export LOCAL_MOSAICS_PATH_DR2=/data2/mostertrij/data/LoTSS_DR2
export LOCAL_MOSAICS_PATH_DR2=$BASE_PROJECT_PATH/Data/Images
#export MOSAICS_PATH_DR1=/data2/mostertrij/pink-basics/data_LoTSS_DR1
#export LOTSS_RAW_CATALOGUE=$CATALOGUE_PATH/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.h5
#export LOTSS_GAUSS_CATALOGUE=$CATALOGUE_PATH/LOFAR_HBA_T1_DR1_catalog_v0.99.gaus.h5
#export LOTSS_COMP_CATALOGUE=$CATALOGUE_PATH/LoTSS_DR1_corrected_cat.comp.h5
#export LOTSS_SOURCE_CATALOGUE=$CATALOGUE_PATH/LoTSS_DR1_corrected_merged.h5
#export LOTSS_RAW_CATALOGUE_DR2=$CATALOGUE_PATH/LoTSS_DR2_v100.srl.h5
export LOTSS_RAW_CATALOGUE_DR2=$CATALOGUE_PATH/LoTSS_DR2_v110_masked.srl.P21.h5
#export LOTSS_GAUSS_CATALOGUE_DR2=$CATALOGUE_PATH/LoTSS_DR2_v100.gaus.h5
export LOTSS_GAUSS_CATALOGUE_DR2=$CATALOGUE_PATH/LoTSS_DR2_v110.gaus.P21.h5
#export LIKELY_UNRESOLVED_CATALOGUE=$CATALOGUE_PATH/GradientBoostingClassifier_A1_31504_18F_TT1234_B1_exp3_DR2.h5
export LIKELY_UNRESOLVED_CATALOGUE=$BASE_PROJECT_PATH/Results/GBC/P21_pred_thresholds.h5
#export OPTICAL_CATALOGUE=$CATALOGUE_PATH/combined_panstarrs_wise.h5
# Uncomment to exclude DR1 area
#export EXCLUDE_DR1_AREA=1

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
SIGMA_BOX_FIT=5
UNRESOLVED_THRESHOLD=0.20 # Threshold to use for Alegre's desicion tree
PRECOMPUTED_BBOXES=1 # 0 is False; 1 is True
CLIP=1 # Determines if the cut-outs are clipped or not
CLIP_LOW=1 # lower clip value (sigma)
CLIP_HIGH=30 # upper clip value (sigma)
DEBUG=1 # 0 is False, 1 is True
OVERWRITE=1 # 0 is False, 1 is True
REMOTE=0
####################################
mkdir -p logs

python $BASE_PROJECT_PATH/lofar_frcnn_prepro/lib/fits_in_folder_to_hdf5.py $BASE_PROJECT_PATH/Results/GBC &> logs/out00_$DATASET_NAME.txt
python $BASE_PROJECT_PATH/lofar_frcnn_prepro/lib/fits_in_folder_to_hdf5.py $CATALOGUE_PATH &> logs/out01_$DATASET_NAME.txt

#1 - Make a source list: 
# Given a source list, create source objects 
python $PROJECTPATH/imaging_scripts/make_cutout_objects_single_pointing.py $TRAINING_MODE $SAMPLE_LEN \
    $SAMPLE_LIST 1 $DATASET_NAME $N_FIELDS $REMOTE  $UNRESOLVED_THRESHOLD &> logs/out1_$DATASET_NAME.txt


#2 - Generate cut-outs using: 
#(where N and M are the index range)
#Note: The images will be placed in the directory from which you run the commands
#echo """
python -W ignore $PROJECTPATH/imaging_scripts/make_cutout_frcnn.py $TRAINING_MODE $SAMPLE_LIST $SAMPLE_LEN \
    $OVERWRITE $CUTOUT_SIZE_IN_ARCSEC $RESCALED_CUTOUT_SIZE_IN_PIXELS $DATASET_NAME $ROTATION \
    $UNRESOLVED_THRESHOLD $REMOVE_UNRESOLVED &> logs/out2_$DATASET_NAME.txt
#"""


#3 - Determine box size and component numbers using: 
#echo """
python $PROJECTPATH/labeling_scripts/labeler_rotation.py $TRAINING_MODE $SAMPLE_LIST $OVERWRITE \
    $EDGE_CASES $BOX_SCALE $DEBUG $SIG5_ISLAND_SIZE $INCLUDE_LOW_SIG $ALLOW_OVERLAP $INCL_DIFF \
   $DATASET_NAME $single_comp_rotation_angles_deg $multi_comp_rotation_angles_deg $ROTATION \
    $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES $UNRESOLVED_THRESHOLD \
        $REMOVE_UNRESOLVED $SIGMA_BOX_FIT &> logs/out3_$DATASET_NAME.txt
#"""

#4 - Create labels in XML format and structure data in correct folder hierarchy
#echo """
python $PROJECTPATH/labeling_scripts/create_and_populate_initial_dataset_rotation.py \
   $FIXED_CUTOUT_SIZE $INCL_DIFF $DIFFICULT_LIST_NAME $CLIP $CLIP_LOW $CLIP_HIGH \
   $DATASET_NAME $RESCALED_CUTOUT_SIZE_IN_PIXELS $CUTOUT_SIZE_IN_ARCSEC $PRECOMPUTED_BBOXES \
   $TRAINING_MODE $REMOVE_UNRESOLVED $UNRESOLVED_THRESHOLD $SIGMA_BOX_FIT &> logs/out4_$DATASET_NAME.txt
#"""

#sleep 10
#rsync -a /data2/mostertrij/data/frcnn_images/$DATASET_NAME lgm4:/data1/mostertrij/data/frcnn_images
#rsync -a /data2/mostertrij/data/cache/segmentation_maps* lgm4:/data1/mostertrij/data/cache
