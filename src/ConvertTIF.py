'''
This program converts the original *.sid image files from the USDA into *.tif format).  It retains one image per county.
'''

import glob
import os
import subprocess
from multiprocessing import Pool

# this is the folder containing the original images, downloaded as *.sid files from the USDA
ORIGINAL_IMAGES_FOLDER = "/mnt/windmills/images/2017/IA/"
EXTENSION = "ortho_*.sid"

# this is the new folder to store the images as *.tif files
NEW_IMAGES_FOLDER = ORIGINAL_IMAGES_FOLDER + "tif/"

# the number of threads to use (for parallelizing the work)
NUM_THREADS = 12


def main():
    # get the original filenames
    os.chdir(ORIGINAL_IMAGES_FOLDER)
    filenames = list(glob.glob(EXTENSION))

    # create the pool
    with Pool(NUM_THREADS) as pool:
        results = pool.map(convertImage, filenames)

        print("\n".join(results))


def convertImage(filename):
    # save the CWD
    CWD = os.getcwd()

    # change directories
    os.chdir(NEW_IMAGES_FOLDER)

    # create the command
    # NOTE: the path below is where we assume you have MrSID installed to provide the mrsiddecode program
    # /opt/MrSID_DSDK-9.5.4.4703-rhel6.x86-64.gcc531/Raster_DSDK/bin/mrsiddecode -wf -i ortho_1-1_1n_s_ia089_2017_1.sid -o ortho_1-1_1n_s_ia089_2017_1.tif
    inputFile = ORIGINAL_IMAGES_FOLDER + filename
    outputFile = NEW_IMAGES_FOLDER + filename.replace(".sid", ".tif")
    command = ["/opt/MrSID_DSDK-9.5.4.4703-rhel6.x86-64.gcc531/Raster_DSDK/bin/mrsiddecode", "-wf", "-i", inputFile, "-o", outputFile]

    # run the command
    result = subprocess.run(command)

    # return to the old directory
    os.chdir(CWD)

    return outputFile + "\n" + str(result)


if __name__ == "__main__":
    main()
    
