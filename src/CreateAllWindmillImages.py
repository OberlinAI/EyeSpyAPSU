'''
This program creates training images of windmill and non-windmill locations.

For windmills, it finds every location with a known polygon, randomly offsets the center of that polygon from the center of
the windmill image (so that windmills are not always in the center), and crops a desired size around the windmill.

For non-windmills, it randomly samples locations not immediately near windmills, and crops a desired size around the randomly sampled
non-windmill location.
'''

import math
from multiprocessing import Manager, Pool
import numpy as np
import subprocess
import sys
import WindmillUtils
import UTMConverter

SEED = 12345    # a random seed to use for randomization
COUNTY_BUFFER = 1000    # a buffer to use to make sure that images do not span counties
SIZE = 256    # the number of pixels wide and tall to make each image (e.g., 256x256)
SHIFT = SIZE // 2 - 48   # a maximum amount to shift each windmill from the center of the image
RATIO = 9   # the ratio of non-windmill locations to grab for every one windmill
ZONE = "15"   # Iowa's "zone" within UTM coordinates
NUM_THREADS = 12    # the number of threads to use (for parallelizing the work)

# the name of the file that will contain the polygons of all identified windmills
WINDMILL_LABELS_FILENAME = "Windmills_Polygons_" + str(SIZE) + ".csv"

# the name of the file that will contain all non-windmill image details
NOT_WINDMILL_LABELS_FILENAME = "NotWindmills_Images_" + str(SIZE) + ".csv"


def findNearby(x, y, windmills, T):
    nearby = []

    for windmill in windmills:
        center = (windmill[9], windmill[10])

        otherX = center[0] # windmill[0]
        otherY = center[1] # windmill[1]

        xdiff = x - otherX
        ydiff = y - otherY
        dist = math.sqrt(xdiff * xdiff + ydiff * ydiff)

        if dist < T:
            nearby.append(windmill)

    return nearby, len(nearby)


def getPoints(label):
    # get the points as a string
    label = label.replace('"', '')
    label = label.replace("Polygon ((", "")
    label = label.replace("))", "")
    sp = label.split(",")

    points = []
    for pt in sp:
        pt = pt.strip()
        ptSp = pt.split(" ")
        point = (float(ptSp[0]), float(ptSp[1]))
        points.append(point)

    return points


def sliceWindmill(windmill, num, counties):
    center = (windmill[9], windmill[10])

    # come up with the top left corner
    # and randomly shift the image so it isn't centered
    origTopX = center[0] - SIZE // 2 + np.random.randint(-SHIFT, SHIFT + 1)
    origTopY = center[1] + SIZE // 2 + np.random.randint(-SHIFT, SHIFT + 1)

    # offset by the county image top left
    county = windmill[0]
    countyCoords = counties[county]
    topX = origTopX - countyCoords[0]
    topY = countyCoords[1] - origTopY

    county = str(county)
    if len(county) == 1:
        county = "00" + county
    elif len(county) == 2:
        county = "0" + county

    # gdal_translate -srcwin 2365 17523 4000 4000 ortho_1-1_1n_s_ia089_2017_1.tif Image_7.tif
    inputFile = "ortho_1-1_1n_s_ia" + county + "_2017_1.tif"
    outputFile = "windmill_" + str(windmill[4]) + "_" + county + "_" + str(num) + ".tif"
    command = ["gdal_translate", "-srcwin", str(topX), str(topY), str(SIZE), str(SIZE), inputFile, outputFile]

    result = subprocess.run(command)

    return origTopX, origTopY, outputFile


def sliceOpenLand(county, num, windmills, counties):
    countyCoords = counties[county]
    topLeft = (countyCoords[0], countyCoords[1])
    bottomRight = (countyCoords[0] + countyCoords[2], countyCoords[1] - countyCoords[3])
    buffer = COUNTY_BUFFER + SIZE // 2

    # make more than one notwindmill image per windmill
    labels = []
    for i in range(RATIO):
        newNum = num + i

        # randomly generate a center that is too far away from windmills
        done = False
        while not done:
            x = np.random.randint(topLeft[0] + buffer, bottomRight[0] - buffer)
            y = np.random.randint(bottomRight[1] + buffer, topLeft[1] - buffer)

            nearby, count = findNearby(x, y, windmills, SIZE)
            if count == 0:
                done = True

        # convert to the top left coordinate
        x = x - countyCoords[0] - SIZE // 2
        y = countyCoords[1] - y + SIZE // 2

        tl = UTMConverter.convertToLatLong(countyCoords[0] + x, countyCoords[1] - y, ZONE)
        br = UTMConverter.convertToLatLong(countyCoords[0] + x + SIZE - 1, countyCoords[1] - y - SIZE + 1, ZONE)

        countyStr = str(county)
        if len(countyStr) == 1:
            countyStr = "00" + countyStr
        elif len(countyStr) == 2:
            countyStr = "0" + countyStr

        # gdal_translate -srcwin 2365 17523 4000 4000 ortho_1-1_1n_s_ia089_2017_1.tif Image_7.tif
        inputFile = "ortho_1-1_1n_s_ia" + countyStr + "_2017_1.tif"
        outputFile = "notwindmill_" + countyStr + "_" + str(newNum) + ".tif"
        command = ["gdal_translate", "-srcwin", str(x), str(y), str(SIZE), str(SIZE), inputFile, outputFile]

        result = subprocess.run(command)

        label = (county, newNum, outputFile, tl[0], tl[1], br[0], br[1])
        labels.append(label)

    return newNum, labels


def calculateOverlap(polygon, topLeft, bottomRight):
    top = topLeft[0]
    left = topLeft[1]
    bottom = bottomRight[0]
    right = bottomRight[1]

    # are any points inside or outside of the image?
    contains = False
    outside = False
    for point in polygon:
        # longitude is first in WKT
        lat = point[1]
        long = point[0]

        if inside(lat, long, top, left, bottom, right):
            contains = True
        else:
            outside = True

    if not contains:
        return []

    # if no point is outside, then the label is fine as is!
    if not outside:
        return list(polygon)

    overlap = []
    for i in range(len(polygon)):
        one = polygon[i-1] # exploit negative indexing to get the edge between first and last points
        two = polygon[i]

        # long is first then lat in each WKT point
        if inside(one[1], one[0], top, left, bottom, right) and inside(two[1], two[0], top, left, bottom, right):
            overlap.append(two)
        elif inside(one[1], one[0], top, left, bottom, right):
            # the ith is outside and the i-1th is inside
            overlap.append(getMidpoint(one, two, top, left, bottom, right))
            #print("one inside", getMidpoint(one, two, top, left, bottom, right))
        elif inside(two[1], two[0], top, left, bottom, right):
            # the ith is inside and the i-1th is outside
            overlap.append(getMidpoint(one, two, top, left, bottom, right))
            #print("two inside", getMidpoint(one, two, top, left, bottom, right))
            overlap.append(two)

    # verify that every point is in the image
    for point in overlap:
        if not inside(point[1], point[0], top, left, bottom, right):
            print("Point not inside image:", point, topLeft, bottomRight)
            sys.exit(-1)

    return overlap



def getMidpoint(one, two, top, left, bottom, right):
    # in case some points are inside and some are outside
    # if xl < left  and bottom <= yl <= top then xk = left and yk = m (left - xl) + yl
    # if xr > right and bottom <= yr <= top then xk = right and yk = m (right - xl) + yl
    # if yl < bottom and left <= xl <= right then yk = bottom and xk = (bottom - yl) / m + xl
    # if yl > top and left <= xl <= right then yk = top and xk = (top - yl) / m + xl

    # if xl < left and yl < bottom then test both xk = left and yk = bottom and use the point inside the image
    # if xl < left and yl > top then test both xk = left and yk = top and use the point inside the image
    # if xl > right and yl < bottom then test both xk = right and yk = bottom and use the point inside the image
    # if xl > right and yl > top then test both xk = right and yk = top and use the point inside the image

    # which is left and which is right?
    # long is first, then lat in each WKT point
    if one[0] < two[0]:
        l = one
        r = two
    else:
        r = one
        l = two

    # find the slope
    m = (r[1] - l[1]) / (r[0] - l[0])

    if l[0] < left:
        #print("l[0] < left")
        # find the intercept
        xk = left
        yk = m * (xk - l[0]) + l[1]

        if bottom <= l[1] and l[1] <= top:
            # we only fell off one side
            return (xk, yk)
        elif l[1] < bottom:
            # find both intercepts, pick the valid one (only one should be inside the image)
            if inside(yk, xk, top, left, bottom, right):
                return (xk, yk)
            else:
                yk = bottom
                xk = (yk - l[1]) / m + l[0]

                if inside(yk, xk, top, left, bottom, right):
                    return (xk, yk)
                else:
                    print("No midpoint found: ", one, two, top, left, bottom, right)
        elif l[1] > top:
            # find both intercepts, pick the valid one (only one should be inside the image)
            if inside(yk, xk, top, left, bottom, right):
                return (xk, yk)
            else:
                yk = top
                xk = (yk - l[1]) / m + l[0]

                if inside(yk, xk, top, left, bottom, right):
                    return (xk, yk)
                else:
                    print("No midpoint found: ", one, two, top, left, bottom, right)
    elif r[0] > right:
        #print("r[0] > right")
        # find the intercept
        xk = right
        yk = m * (xk - l[0]) + l[1]

        if bottom <= r[1] and r[1] <= top:
            # we only fell off one side
            return (xk, yk)
        elif r[1] < bottom:
            # find both intercepts, pick the valid one (only one should be inside the image)
            if inside(yk, xk, top, left, bottom, right):
                return (xk, yk)
            else:
                yk = bottom
                xk = (yk - l[1]) / m + l[0]

                if inside(yk, xk, top, left, bottom, right):
                    return (xk, yk)
                else:
                    print("No midpoint found: ", one, two, top, left, bottom, right)
        elif r[1] > top:
            # find both intercepts, pick the valid one (only one should be inside the image)
            if inside(yk, xk, top, left, bottom, right):
                return (xk, yk)
            else:
                yk = top
                xk = (yk - l[1]) / m + l[0]

                if inside(yk, xk, top, left, bottom, right):
                    return (xk, yk)
                else:
                    print("No midpoint found: ", one, two, top, left, bottom, right)
    elif l[1] < bottom:
        #print("l[1] < bottom")
        # find the intercept
        yk = bottom
        xk = (yk - l[1]) / m + l[0]
        return (xk, yk)
    elif l[1] > top:
        #print("l[1] > top")
        # find the intercept
        yk = top
        xk = (yk - l[1]) / m + l[0]
        return (xk, yk)
    elif r[1] < bottom:
        #print("r[1] < bottom")
        # find the intercept
        yk = bottom
        xk = (yk - l[1]) / m + l[0]
        return (xk, yk)
    elif r[1] > top:
        #print("r[1] > top")
        # find the intercept
        yk = top
        xk = (yk - l[1]) / m + l[0]
        return (xk, yk)
    else:
        print(l, r, top, left, bottom, right)


def inside(lat, long, top, left, bottom, right):
    return lat <= top and lat >= bottom and long >= left and long <= right


def convertLabel(points):
    label = "Polygon (("
    label += ",".join([" ".join([str(i) for i in p]) for p in points])
    label += "))"

    return label


def createLabels(topLeft, bottomRight, x, y, windmills):
    # find each windmill in the image
    nearby, num = findNearby(x, y, windmills, SIZE * 2)

    localLabels = []
    for windmill in nearby:
        polygon = getPoints(windmill[-1])
        overlap = calculateOverlap(polygon, topLeft, bottomRight)

        if len(overlap) > 1:
            localLabels.append(overlap)

    if len(localLabels) == 0:
        print("PROBLEM!", topLeft, bottomRight, x, y, num)
        sys.exit(-1)

    return localLabels


def createImages(county):
    np.random.seed(SEED + county)

    # get the windmills for this county
    global windmillsByCounty, counties
    windmills = windmillsByCounty[county]

    # create the images for each windmill
    num = 0
    windmillLabels = []
    for windmill in windmills:
        num += 1
        topX, topY, imageName = sliceWindmill(windmill, num, counties)

        topLeft = UTMConverter.convertToLatLong(topX, topY, ZONE)
        bottomRight = UTMConverter.convertToLatLong(topX + SIZE - 1, topY - SIZE + 1, ZONE)

        #print(windmill)

        t = (windmill[0], num, windmill[4], imageName, topLeft[0], topLeft[1], bottomRight[0], bottomRight[1],
             createLabels(topLeft, bottomRight, windmill[9], windmill[10], windmills))
        windmillLabels.append(t)

    # randomly create images without windmills (one from each county for each windmill)
    lastNum = 0
    notwindmillLabels = []
    for windmill in windmills:
        lastNum, labels = sliceOpenLand(windmill[0], lastNum + 1, windmills, counties)
        notwindmillLabels.extend(labels)

    return (windmillLabels, notwindmillLabels)


if __name__ == "__main__":
    # read in the data from file
    counties = WindmillUtils.readCounties()
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
    for county in wbc:
        windmillsByCounty[county] = wbc[county]

    countyNums = list(windmillsByCounty.keys())
    countyNums.sort()

    # create the pool
    with Pool(NUM_THREADS) as pool:
        results = pool.map(createImages, countyNums)

    # combine the results
    windmillLabels = []
    notwindmillLabels = []
    for r in results:
        windmillLabels.extend(r[0])
        notwindmillLabels.extend(r[1])
    windmillLabels.sort()
    notwindmillLabels.sort()

    # output the image descriptions
    with open(WINDMILL_LABELS_FILENAME, "w") as file:
        file.write(
            "County;Num;case_id;ImageFile;ImageTop_Lat;ImageLeft_Long;ImageBottom_Lat;ImageRight_Long;Polygons\n")
        for windmill in windmillLabels:
            file.write(";".join([str(s) for s in windmill[:-1]]) + ";" + " ".join(
                [convertLabel(points) for points in windmill[-1]]) + "\n")

    # output the image descriptions
    with open(NOT_WINDMILL_LABELS_FILENAME, "w") as file:
        file.write(
            "County;Num;ImageFile;ImageTop_Lat;ImageLeft_Long;ImageBottom_Lat;ImageRight_Long\n")
        for notwindmill in notwindmillLabels:
            file.write(";".join([str(s) for s in notwindmill]) + "\n")

