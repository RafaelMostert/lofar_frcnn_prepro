This folder contains two scripts, usually ran as script 3 and 4:

labeler_rotation.py

Enriches list of cutout_objects with bounding boxes encapsulating the five or three sigma
emission of all radio emission (for different rotation angles)

create_and_populate_initial_dataset_rotation.py

Converts FITS cutouts to three-channel png-images as required for the Fast R-CNN.
Creates precomputed bounding boxes or Regions of Interest by taking all possible
combinations of single radio objects in the cutout and prunes the regions that are
duplicates.
Saves all required label-information in the required JSON format for the Fast R-CNN
implementation that we use.
