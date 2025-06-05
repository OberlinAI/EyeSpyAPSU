'''
This program provides functions helpful for working with windmills and images, not necessarily part of the learning process.
'''

import glob
import os

'''replace these with the apporpriate locations on your computer.  NOTE: ~/Code/EyeSpyAPSU is assumed to be the GitHub repository'''
COUNTIES_FILENAME = "~/Code/EyeSpyAPSU/data/counties_utm.csv"
CENTERS_FILENAME = "~/Code/EyeSpyAPSU/data/WindmillCenters.csv"
ALL_WINDMILLS_FILE = "~/Code/EyeSpyAPSU/data/windmills_IA.csv"
WINDMILLS_FOLDER = "~/Code/EyeSpyAPSU/data/"
WINDMILLS_FILENAME_REGEX = "Windmills_*.csv"  # from the WindmillLocationsPerCounty.zip file in the data folder in the GitHub repository
LABELS_FOLDER = "~/Code/EyeSpyAPSU/data/"
LABELS_FILENAME_REGEX = "Windmill_Labels_*.txt"   # from the WindmillPolygonsPerCounty.zip file in the data folder in the GitHub repository


def readCounties():
    with open(COUNTIES_FILENAME, "r") as file:
        lines = file.readlines()

    counties = {}
    for line in lines[1:]:
        sp = line.strip().split(",")
        county = int(sp[0])

        counties[county] = (int(sp[1]), int(sp[2]), int(sp[3]), int(sp[4]))

    return counties


def readWindmillsAndLabels():
    windmills = []
    counties = []
    topLefts = {}
    bottomRights = {}

    # go to the folder with the windmills
    CWD = os.getcwd()
    os.chdir(LABELS_FOLDER)

    # read in the windmills

    with open(ALL_WINDMILLS_FILE, "r") as file:
        lines = file.readlines()

    for line in lines[1:]:
        windmill = line.split(";")

        # convert the fields
        windmill[0] = int(windmill[0])
        windmill[2] = float(windmill[2])
        windmill[3] = float(windmill[3])
        windmill[4] = int(windmill[4])
        windmill[5] = float(windmill[5])
        windmill[6] = float(windmill[6])
        windmill[7] = float(windmill[7])
        windmill[8] = float(windmill[8])
        windmill[9] = int(windmill[9])
        windmill[10] = int(windmill[10])

        # save the windmill
        windmills.append(windmill)

        if windmill[0] not in counties:
            counties.append(windmill[0])

    counties.sort()
    minX = 9999999999999999999
    maxX = 0
    minY = 9999999999999999999
    maxY = 0
    for county in counties:
        topLefts[county] = (minX, maxY)
        bottomRights[county] = (maxX, minY)


    for windmill in windmills:
        x = windmill[9]
        y = windmill[10]

        minX, maxY = topLefts[windmill[0]]
        maxX, minY = bottomRights[windmill[0]]

        if x < minX:
            minX = x
        if x > maxX:
            maxX = x

        if y < minY:
            minY = y
        if y > maxY:
            maxY = y

        topLefts[windmill[0]] = (minX, maxY)
        bottomRights[windmill[0]] = (maxX, minY)

    # reset the CWD
    os.chdir(CWD)


    return windmills, topLefts, bottomRights


def readWindmills():
    windmills = []
    topLefts = {}
    bottomRights = {}

    # go to the folder with the windmills
    CWD = os.getcwd()
    os.chdir(WINDMILLS_FOLDER)

    # read in the windmills
    for filename in glob.glob(WINDMILLS_FILENAME_REGEX):
        county = findCounty(filename)

        with open(filename, "r") as file:
            lines = file.readlines()

        minX = 9999999999999999999
        maxX = 0
        minY = 9999999999999999999
        maxY = 0

        for line in lines[1:]:
            sp = line.split(",")
            windmill = (int(sp[3]), int(sp[4]), float(sp[0]), float(sp[1]), county)
            windmills.append(windmill)

            x = windmill[0]
            y = windmill[1]

            if x < minX:
                minX = x
            elif x > maxX:
                maxX = x

            if y < minY:
                minY = y
            elif y > maxY:
                maxY = y

        topLefts[county] = (minX, maxY)
        bottomRights[county] = (maxX, minY)

    # reset the CWD
    os.chdir(CWD)


    return windmills, topLefts, bottomRights


def findCounty(filename):
    return int(filename.split("_")[1].replace(".csv", ""))


def organizeByCounty(windmills):
    countyWindmills = {}

    for windmill in windmills:
        county = windmill[4]

        if county not in countyWindmills:
            countyWindmills[county] = []

        countyWindmills[county].append(windmill)

    return countyWindmills


def readCenters():
    with open(CENTERS_FILENAME, "r") as file:
        lines = file.readlines()

    centers= {}
    for line in lines[1:]:
        sp = line.strip().split(",")

        windmill = (float(sp[0]), float(sp[1]))

        centers[windmill] = (int(sp[2]), int(sp[3]))

    return centers


def readLabels(folder=LABELS_FOLDER, regex=LABELS_FILENAME_REGEX):
    labels = {}

    # go to the folder with the windmills
    CWD = os.getcwd()
    os.chdir(folder)

    labels = {}
    for filename in glob.glob(regex):
        with open(filename, "r") as file:
            lines = file.readlines()

        for line in lines:
            if line.startswith("Polygon"):
                sp = line.split("))")
                coords = sp[-1].strip()
                coords = coords.split(",")

                windmill = (float(coords[0]), float(coords[1]))
                label = sp[0] + "))"
                labels[windmill] = label

    # reset the CWD
    os.chdir(CWD)

    return labels
