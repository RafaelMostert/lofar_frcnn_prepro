This folder contains two scripts, usually ran as script 1 and 2:

multi_field_decision_tree.py

Gets a number of observing pointings to process through a bash script in ../create_datasets
It filters these sources through a flowchart replicated from Williams et al. 2019.
It turns the resulting sources into a list of cutout_object class objects (see ../lib/cutout_object.py)

make_cutout_frcnn.py  

Goes down the list of cutout_objects and creates corresponding LoTSS-DR2 FITS cutouts.
Enriches the cutout_objects with the locations of all PyBDSF sources within the cutout and 
(when TRAIN_MODE is true) labels them as belonging to the central PyBDSF source or unrelated neighbours.
If REMOVE_UNRESOLVED in the ../create_datasets script was true, unresolved sources are removed from
the cutout at this point.
