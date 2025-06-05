'''
This program slices entire counties (each a single large *.tif file) into smaller, gridded, non-overlapping images (in *tif format).
'''

import glob
from multiprocessing import Manager, Pool
import os
import subprocess
import WindmillUtils

SIZE = 256    # how wide and tall to make each image (e.g., 256x256 pixels)
NUM_THREADS = 12    # the number of threads to use (for parallelizing the work)

# the folder containing the original images (as *.tif files), created by the ConvertTIF.py program
ORIGINAL_IMAGES_FOLDER = "/mnt/windmills/images/2017/IA/tif/"

# the location where the sliced images should be located (as *.tif files)
NEW_IMAGES_FOLDER_BASE = ORIGINAL_IMAGES_FOLDER + str(SIZE) + "/counties/"


def splitCounty(countyNum):
    # make sure the folder exists for this county
    countyFolder = NEW_IMAGES_FOLDER_BASE + str(countyNum) + "/"
    if not os.path.exists(countyFolder):
        os.mkdir(countyFolder)

    # create the string version of the county
    countyStr = str(countyNum)
    if len(countyStr) == 1:
        countyStr = "00" + countyStr
    elif len(countyStr) == 2:
        countyStr = "0" + countyStr

    # get the county
    county = counties[countyNum]
    localWindmills = windmillsByCounty[countyNum]

    # read in the images
    os.chdir(ORIGINAL_IMAGES_FOLDER)
    filename = "ortho_1-1_1n_s_ia" + countyStr + "_2017_1.tif"

    x = 0
    while x < county[2] - SIZE:
        y = 0
        while y < county[3] - SIZE:
            hasWindmill = containsWindmill(x + county[0], county[1] - y, localWindmills)

            # create the command
            # gdal_translate -srcwin 2365 17523 4000 4000 ortho_1-1_1n_s_ia089_2017_1.tif Image_7.tif
            outputStart = "windmill" if hasWindmill else "notwindmill"
            outputFile = countyFolder + outputStart + "_" + str(countyNum) + "_" + str(county[0] + x) \
                         + "_" + str(county[1] - y) + ".tif"
            command = ["gdal_translate", "-srcwin", str(x), str(y), str(SIZE), str(SIZE), filename, outputFile]

            # run the command
            result = subprocess.run(command)

            y += SIZE
        x += SIZE


def containsWindmill(leftX, topY, windmills):
    for windmill in windmills:
        center = (windmill[9], windmill[10])
        if center[0] >= leftX and center[0] <= leftX + SIZE and center[1] <= topY and center[1] >= topY - SIZE:
            return True

    return False


if __name__ == "__main__":
    # make sure the new folder base is made
    if not os.path.exists(NEW_IMAGES_FOLDER_BASE):
        os.mkdir(NEW_IMAGES_FOLDER_BASE)

    # read in the county data
    allCounties = WindmillUtils.readCounties()

    # read in the windmills
    windmills, tl, br = WindmillUtils.readWindmillsAndLabels()

    # create the Manager
    manager = Manager()

    # split the windmills by county
    wbc = {}
    for windmill in windmills:
        county = windmill[0]

        if county not in wbc:
            wbc[county] = []

        wbc[county].append(windmill)

    # copy the windmills
    windmillsByCounty = manager.dict()
    counties = manager.dict()
    for county in wbc:
        windmillsByCounty[county] = wbc[county]
        counties[county] = allCounties[county]

    countyNums = list(windmillsByCounty.keys())
    countyNums.sort()

    # create the pool
    with Pool(NUM_THREADS) as pool:
        results = pool.map(splitCounty, countyNums)

