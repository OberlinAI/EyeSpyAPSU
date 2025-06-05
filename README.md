# EyeSpyAPSU
Eye Spy a PSU: Automating Sampling Frame Construction from Aerial Images using Machine Learning

This GitHub repository contains the code necessary to replicate the study evaluating how supervised machine learning models (specifically CNNs) might be used to automatically discover places, objects, or locations (POL) from aerial images in order to construct sampling frames for social science research.  This study is currently under review as the manuscript "Eye Spy a PSU! Automating Sampling Frame Construction from Aerial Images using Machine Learning" in the journal (*Methods, Data, and Analyses*)[https://mda.gesis.org/index.php/mda].

## Data to Download

To replicate our study, you will need to use to download two data sources:

1. USDA Images
2. U.S. Wind Turbine Database

## Repository Organization 

The contents of this repository are organized as follows:

* The **data** folder contains three additional data files we've created for the project:

1. `Windmills_Polygons_256.csv` contains the pixel coordinates of the polygons representing each windmill in each image of the study
2. `counties_utm.csv` contains the UTM coordinates of each county in Iowa (where each county is identified by the last three digits of its FIPS code)
3. `windmills_IA.csv` contains a subset of the U.S. Wind Turbine Database mentioned above for the windmills present in the state of Iowa

* The **src** folder contains the Python code that learns the machine learning models and conducts our experiments

* The **scripts** folder contains Bash scripts that run our Python code to train multiple models with different random seeds (that affect the initial state of each learned neural network)

* The **R** folder contains the R code that was used to evaluate the results of our study

## Running the Study

#### Creating Image Files

In order to convert the original county-wide image files from the USDA in `sid` file format into `tif` file format, use the `src/ConvertTIF.py` program.  You will need to specify where the original `*.sid` image files are downloaded and where you want to store the processed `*.tif` files as varaibles in the Python program.

Next, to splice the original county-wide image files in `tif` format into a contiguous, non-overlapping grid of `256 x 256` (or other sized) meter images, use the `src/SliceImages.py` program.  Again, you will need to specify where you saved the `*.tif` files created by the `ConvertTIF.py` program, as well as where you want to store the sliced county images.  Both locations are stored as variables in the Python program.

Finally, to create the windmill and non-windmill images used for hyperparameter tuning, run the `src/CreateAllWindmillImages.py` program.

#### Training and Evaluating Models

To train and evaluate the ZFNet, VGG-16, or U-Net models, you should use the `src/WindmillZFNet.py`, `src/WindmillVGG.py`, and `src/WindmillUNet.py` programs, respectively.  Variables at the top of each program are used to specify default hyperparameter values (which can also be passed in as command line arguments), as well as the locations of the images of interest.  For advice on how to run the programs, please consult the Bash scripts in the `scripts` folder.
