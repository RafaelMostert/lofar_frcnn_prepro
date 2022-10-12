<!-- ABOUT THE PROJECT -->

## About The Project

This code generates a dataset that can be used to either train or infer radio component association predictions from a
Fast R-CNN.


<!-- GETTING STARTED -->
test2

## Getting Started

### Prerequisites

The scripts in this project require the following python packages:

* Within a python >=3.7 environment, install the following prerequisites using either pip
  ```sh
  pip install -r requirements.txt
  ```
  or conda
  ```sh
  conda env create -f requirements_conda.yml
  conda activate lofar_astro
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/RafaelMostert/lofar_frcnn_tools.git
   ```

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->

## Usage

1. Enter the create_dataset_scripts directory.
2. Copy one of the template bash-scripts (the one for inference or the one for training)
3. Modify all paths and parameters to satisfy your needs
4. Run the bash-script (make sure it is runnable using chmod if needed)

<p align="right">(<a href="#top">back to top</a>)</p>

## Script details

Running the bash-scripts in the create_dataset_scripts directory will sequentially run four scripts:

1. imaging_scripts/make_cutout_objects.py
2. imaging_scripts/make_cutout_frcnn.py
3. labeling_scripts/labeler_rotation.py
4. labeling_scripts/create_and_populate_initial_dataset_rotation.py

make_cutout_objects.py:
Checks which observed pointings are available and also listed in the initial source catalogue and then turns
these sources within that observed pointing into a list of cutout_object
class objects (see ../lib/cutout_object.py)

make_cutout_frcnn.py:
Goes down the list of cutout_objects and creates corresponding LoTSS-DR2 FITS cutouts. Enriches the cutout_objects with
the locations of all PyBDSF sources within the cutout and
(when TRAIN_MODE is true) labels them as belonging to the central PyBDSF source or unrelated neighbours. If
REMOVE_UNRESOLVED in the ../create_datasets script was true, unresolved sources are removed from the cutout at this
point.

labeler_rotation.py:
Enriches list of cutout_objects with bounding boxes encapsulating the five or three sigma emission of all radio
emission (for different rotation angles)

create_and_populate_initial_dataset_rotation.py:
Converts FITS cutouts to three-channel png-images as required for the Fast R-CNN. Creates precomputed bounding boxes or
Regions of Interest by taking all possible combinations of single radio objects in the cutout and prunes the regions
that are duplicates. Saves all required label-information in the required JSON format for the Fast R-CNN implementation
that we use.

<!-- LICENSE 
## License

Distributed under the x License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
-->


<!-- CONTACT -->

## Contact

Rafael Mostert - mostert @strw.leidenuniv.nl

<p align="right">(<a href="#top">back to top</a>)</p>
