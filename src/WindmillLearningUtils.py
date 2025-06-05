'''
This program provides many functions helpful to the learning process, not unique to a particular type of model.
'''

import copy
import glob
import math
from matplotlib.path import Path
import numpy as np
import os
import PIL
import random
import sys


def readOnlyIds(imageFolder):
    # create the lists to return
    ids = []

    # save the original working directory
    CWD = os.getcwd()

    # go into the image folder
    os.chdir(imageFolder)

    # read in the ids
    for filename in glob.glob("windmill*.tif"):
        ids.append(filename)

    for filename in glob.glob("notwindmill*.tif"):
        ids.append(filename)

    return ids

def readAllImages(imageFolder, binaryFolder = "", batchSize = 10000):
    # create the lists to return
    images = []
    labels = []
    ids = []

    # save the original working directory
    CWD = os.getcwd()

    # go into the image folder
    os.chdir(imageFolder)

    # read in the images
    num = 0
    for filename in glob.glob("windmill*.tif"):
        im = PIL.Image.open(filename)
        npIm = np.array(im) / 255

        images.append(npIm)
        labels.append([1])
        ids.append(filename)

        if num % 1000 == 0:
            print("windmill:", num)
        num += 1

    for filename in glob.glob("notwindmill*.tif"):
        im = PIL.Image.open(filename)
        npIm = np.array(im) / 255

        images.append(npIm)
        labels.append([0])
        ids.append(filename)

        if num % 1000 == 0:
            print("not windmill: ", num)

        num += 1

    if binaryFolder != "":
        # save the data to files in batches
        start = 0
        end = min(start + batchSize, len(images))
        batch = 0
        while start < end:
            np.save(binaryFolder + "images" + str(batch) + ".npy", images[start:end])
            np.save(binaryFolder + "labels" + str(batch) + ".npy", labels[start:end])
            np.save(binaryFolder + "ids" + str(batch) + ".npy", ids[start:end])

            batch += 1
            start = end
            end = min(start + batchSize, len(images))

    # go back to the original working directory
    os.chdir(CWD)

    return images, labels, ids


def readImagesByIds(imageFolder, ids):
    # create the lists to return
    images = []

    # save the original working directory
    CWD = os.getcwd()

    # go into the image folder
    os.chdir(imageFolder)

    # read in the images
    num = 0
    for filename in ids:
        im = PIL.Image.open(filename)
        npIm = np.array(im) / 255

        images.append(npIm)

    # go back to the original working directory
    os.chdir(CWD)

    return images


def readLabelsByIds(ids):
    # create the list to return
    labels = []
    for filename in ids:
        if filename.startswith("windmill"):
            labels.append([1])
        else:
            labels.append([0])

    return labels


def readIdsOneLabel(imageFolder, getWindmills):
    # create the lists to return
    ids = []

    # save the original working directory
    CWD = os.getcwd()

    # go into the image folder
    os.chdir(imageFolder)

    # read in the images
    regex = "windmill*.tif" if getWindmills else "notwindmill*.tif"
    label = 1 if getWindmills else 0
    for filename in glob.glob(regex):
        ids.append(filename)

    # go back to the original working directory
    os.chdir(CWD)

    return ids


def readImagesOneLabel(imageFolder, getWindmills):
    # get the ids
    ids = readIdsOneLabel(imageFolder, getWindmills)

    # get the images
    images = readImagesByIds(imageFolder, ids)

    # get the labels
    labels = readLabelsByIds(ids)

    return images, labels, ids


def readBatch(num, folder):
    # check if we have already saved the arrays as binary files
    if os.path.exists(folder + "images" + str(num) + ".npy"):
        images = np.load(folder + "images" + str(num) + ".npy")
        labels = np.load(folder + "labels" + str(num) + ".npy")
        ids = np.load(folder + "ids" + str(num) + ".npy")

        return images, labels, ids
    else:
        print("Sorry, but batch", num, "does not exist!")
        sys.exit(-1)


def splitByCounties(images, labels, ids):
    imagesByCounty = {}
    labelsByCounty = {}
    idsByCounty = {}

    for i in range(len(ids)):
        id = ids[i]

        if id.startswith("notwindmill"):
            county = int(id.split("_")[1])
        else:
            county = int(id.split("_")[2])

        if county not in idsByCounty:
            imagesByCounty[county] = []
            labelsByCounty[county] = []
            idsByCounty[county] = []

        imagesByCounty[county].append(images[i])
        labelsByCounty[county].append(labels[i])
        idsByCounty[county].append(ids[i])

    return imagesByCounty, labelsByCounty, idsByCounty


def splitCounties(ids, percentForTraining, maxForTraining):
    idsByCounty = {}

    for i in range(len(ids)):
        id = ids[i]

        if id.startswith("notwindmill"):
            county = int(id.split("_")[1])
        else:
            county = int(id.split("_")[2])

        if county not in idsByCounty:
            idsByCounty[county] = []

        idsByCounty[county].append(ids[i])

    # sort the counties by size
    sizes = []
    totalSize = 0
    for county in idsByCounty:
        t = (len(idsByCounty[county]), county)
        sizes.append(t)
        totalSize += t[0]
    sizes.sort()

    # are we starting from the front or back?
    if maxForTraining:
        sizes.reverse()

    # find the trainingCounties
    trainingCounties = []
    trainingSize = 0
    trainingFinalCount = percentForTraining * totalSize
    while trainingSize < trainingFinalCount:
        t = sizes.pop(0)
        trainingCounties.append(t)
        trainingSize += t[0]

    testingCounties = list(sizes) # after popping, these are the testing counties

    return trainingCounties, testingCounties


def splitTrainingIdsByCounties(ids, percentForTraining, maxForTraining):
    # first, split the data by county
    _, _, idsByCounty = splitByCounties(ids, ids, ids)

    # find the training counties
    trainingCounties, testingCounties = splitCounties(ids, percentForTraining, maxForTraining)

    # create the new lists of data for the desired counties
    newIds = []
    counties = []
    for t in trainingCounties:
        county = t[1]
        counties.append(county)
        for i in range(len(idsByCounty[county])):
            newIds.append(idsByCounty[county][i])

    return newIds, counties


def splitTrainingDataAfterSorting(images, labels, ids, percentForTraining, maxForTraining):
    # first, split the data by county
    imagesByCounty, labelsByCounty, idsByCounty = splitByCounties(images, labels, ids)

    # find the training counties
    trainingCounties, testingCounties = splitCounties(ids, percentForTraining, maxForTraining)

    # create the new lists of data for the desired counties
    newImages = []
    newLabels = []
    newIds = []
    counties = []
    for t in trainingCounties:
        county = t[1]
        counties.append(county)
        for i in range(len(imagesByCounty[county])):
            newImages.append(imagesByCounty[county][i])
            newLabels.append(labelsByCounty[county][i])
            newIds.append(idsByCounty[county][i])

    return newImages, newLabels, newIds, counties


def sampleImbalance(images, labels, ids, ratio):
    # first, split the instances by label
    positiveImages = []
    positiveLabels = []
    positiveIds = []
    negativeImages = []
    negativeLabels = []
    negativeIds = []

    for i in range(len(images)):
        if ids[i].startswith("windmill"):
            positiveImages.append(images[i])
            positiveLabels.append(labels[i])
            positiveIds.append(ids[i])
        else:
            negativeImages.append(images[i])
            negativeLabels.append(labels[i])
            negativeIds.append(ids[i])

    # randomly shuffle the indices of the negative instances
    indices = list(range(len(negativeImages)))
    random.shuffle(indices)

    # start the final sets
    images = list(positiveImages)
    labels = list(positiveLabels)
    ids = list(positiveIds)

    # add the correct number of instances to build a final set
    end = min(len(indices), ratio*len(positiveImages))
    for i in indices[:end]:
        images.append(negativeImages[i])
        labels.append(negativeLabels[i])
        ids.append(negativeIds[i])

    # print(len(images), len(positiveImages), len(negativeImages), len(positiveImages) + len(negativeImages))
    # print(len(positiveImages), len(images), ratio, (ratio + 1) * len(positiveImages))

    return images, labels, ids


def sampleIdsImbalance(ids, ratio):
    # first, split the instances by label
    positiveIds = []
    negativeIds = []

    for i in range(len(ids)):
        if ids[i].startswith("windmill"):
            positiveIds.append(ids[i])
        else:
            negativeIds.append(ids[i])

    # randomly shuffle the indices of the negative instances
    indices = list(range(len(negativeIds)))
    random.shuffle(indices)

    # start the final sets
    ids = list(positiveIds)

    # add the correct number of instances to build a final set
    end = min(len(indices), ratio*len(positiveIds))
    for i in indices[:end]:
        ids.append(negativeIds[i])

    # print(len(images), len(positiveImages), len(negativeImages), len(positiveImages) + len(negativeImages))
    # print(len(positiveImages), len(images), ratio, (ratio + 1) * len(positiveImages))

    return ids


def splitData(images, labels, ids, trainingPercent, validationPercent):
    x = [[] for i in range(3)]
    y = [[] for i in range(3)]
    z = [[] for i in range(3)]

    indices = list(range(len(images)))
    random.shuffle(indices)

    trainSize = int(len(images) * trainingPercent)
    if trainingPercent + validationPercent == 1.0:
        validSize = len(images) - trainSize
    else:
        validSize = int(len(images) * validationPercent)


    for i in range(len(images)):
        index = indices[i]
        if i < trainSize:
            x[0].append(images[index])
            y[0].append(labels[index])
            z[0].append(ids[index])
        elif i < trainSize + validSize:
            x[1].append(images[index])
            y[1].append(labels[index])
            z[1].append(ids[index])
        else:
            x[2].append(images[index])
            y[2].append(labels[index])
            z[2].append(ids[index])

    return x, y, z


def batch(x, y, z, batchSize, nextIndex):
    batchX = []
    batchY = []
    batchZ = []

    n = len(x)
    for i in range(batchSize):
        if nextIndex < n:
            batchX.append(x[nextIndex])
            batchY.append(y[nextIndex])
            batchZ.append(z[nextIndex])
            nextIndex = (nextIndex + 1)

    if nextIndex >= n:
        nextIndex = -1

    return batchX, batchY, batchZ, nextIndex


def batchIds(ids, batchSize, nextIndex):
    batchIds = []

    n = len(ids)
    for i in range(batchSize):
        if nextIndex < n:
            batchIds.append(ids[nextIndex])
            nextIndex = (nextIndex + 1)

    if nextIndex >= n:
        nextIndex = -1

    return batchIds, nextIndex


def calcResults(x, predict, sess, images, labels, filenames, batchSize):
    matrix = [[0, 0], [0, 0]]
    wrongFiles = {}

    correct = 0
    nextIndex = 0
    while nextIndex >= 0:
        imagesSub, labelsSub, filenamesSub, nextIndex = batch(images, labels, filenames, batchSize, nextIndex)

        p = sess.run(predict, feed_dict={x: imagesSub})
        matrixSub, _, wrongFilesSub = calcPerformance(p, labelsSub, filenamesSub)

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                matrix[i][j] += matrixSub[i][j]

                if i == j:
                    correct += matrixSub[i][j]

            if i not in wrongFiles:
                wrongFiles[i] = []
            if i in wrongFilesSub:
                wrongFiles[i].extend(wrongFilesSub[i])

    return matrix, correct / len(labels), wrongFiles


def calcPerformance(rawPredictions, labels, filenames):
    predictions = []
    for rawPred in rawPredictions:
        if rawPred[0] >= 0.5:
            predictions.append(1)
        else:
            predictions.append(0)

    total = 0
    correct = 0
    matrix = [[0, 0], [0, 0]]
    wrongFiles = {}
    for i in range(len(labels)):
        pred = predictions[i]
        actual = int(labels[i][0])

        matrix[actual][pred] += 1

        if actual == pred:
            correct += 1
        else:
            if actual not in wrongFiles:
                wrongFiles[actual] = []
            wrongFiles[actual].append(filenames[i])

        total += 1

    return matrix, correct / total, wrongFiles


def printMatrix(matrix):
    tp = matrix[1][1]
    fp = matrix[0][1]
    fn = matrix[1][0]
    tn = matrix[0][0]

    acc = (tp + tn) / (tp + fp + fn + tn)
    recall_p = tp / (tp + fn) if tp + fn > 0 else 0
    precision_p = tp / (tp + fp) if tp + fp > 0 else 0
    recall_n = tn / (tn + fp) if tn + fp > 0 else 0
    precision_n = tn / (tn + fn) if tn + fn > 0 else 0
    f1 = 2 * recall_p * precision_p / (recall_p + precision_p) if recall_p + precision_p > 0 else 0
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (
                tn + fn)) if tp + fp > 0 and tn + fn > 0 and tp + fn > 0 and tn + fp > 0 else 0
    balanced = 0 if (tp == 0 and fn == 0) or (tn == 0 and fp == 0) else 0.5 * (recall_p + recall_n)
    recallProd = 0 if tp == 0 or tn == 0 else recall_p * recall_n

    print("Acc: " + "{0:.4f}".format(acc) \
           + " RecallP: " + "{0:.4f}".format(recall_p) \
           + " PrecisionP: " + "{0:.4f}".format(precision_p) \
           + " RecallN: " + "{0:.4f}".format(recall_n) \
           + " PrecisionN: " + "{0:.4f}".format(precision_n) \
           + " F1: " + "{0:.4f}".format(f1) \
           + " MCC: " + "{0:.4f}".format(mcc) \
           + " BalSum: " + "{0:.4f}".format(balanced) \
           + " BalProd: " + "{0:.4f}".format(recallProd))
    print("\tNW\tW")
    print("NW", matrix[0][0], matrix[0][1], sep="\t")
    print("W", matrix[1][0], matrix[1][1], sep="\t")


def createSegmentationLabels(ids, TRAINING_IMAGE_FOLDER, IMAGE_SIZE):
    # read in the label CSV
    labelCSV = {}
    with open(TRAINING_IMAGE_FOLDER + "Windmills_Polygons_" + str(IMAGE_SIZE) + ".csv", "r") as file:
        for line in file:
            sp = line.strip().split(";")
            filename = sp[3]
            labelCSV[filename] = sp

    # create the labels
    labels = []
    zeros = np.array([[[1., 0.] for j in range(IMAGE_SIZE)] for i in range(IMAGE_SIZE)])
    for i in range(len(ids)):
        # get the filename
        filename = ids[i]

        if filename.startswith("windmill_"):
            # get the polygons for the label
            csvLine = labelCSV[filename]
            labelPolygons = csvLine[-1]
            polygonsStr = labelPolygons.replace("Polygon ((", "").replace("))", "|")[:-1].split("|")
            polygons = parsePolygons(polygonsStr)

            # convert the polygons into pixels
            polygonPixels = convertToPixels(polygons, float(csvLine[4]), float(csvLine[5]), float(csvLine[6]),
                                            float(csvLine[7]), IMAGE_SIZE)

            # create the label
            label = createLabel(polygonPixels, IMAGE_SIZE)

            labels.append(np.array(convertLabel(label)))
        else:
            labels.append(copy.deepcopy(zeros))

    return labels


def createLabel(polygonPixels, IMAGE_SIZE):
    label = [[0 for j in range(IMAGE_SIZE)] for i in range(IMAGE_SIZE)]

    # find the labels at the pixel level
    # credit to https://stackoverflow.com/questions/21339448/how-to-get-list-of-points-inside-a-polygon-in-python
    x, y = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    for polygon in polygonPixels:
        p = Path(polygon)
        grid = p.contains_points(points)
        maskBool = grid.reshape(IMAGE_SIZE, IMAGE_SIZE)

        polyLabel = [[1 if maskBool[i][j] else 0 for j in range(len(maskBool[i]))] for i in range(len(maskBool))]
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = max(label[i][j], polyLabel[i][j])

    return label


def convertLabel(label):
    newLabel = []
    for i in range(len(label)):
        newLabel.append([])
        for j in range(len(label[0])):
            if label[i][j] == 0:
                newLabel[i].append([1., 0.])
            else:
                newLabel[i].append([0., 1.])

    return newLabel


def parsePolygons(polygons):
    ret = []

    for pStr in polygons:
        polygon = []

        pStr = pStr.strip()  # remove trailing whitespace
        for pt in pStr.split(","):
            sp = pt.split(" ")
            t = (float(sp[0]), float(sp[1]))
            polygon.append(t)

        ret.append(polygon)

    return ret


def convertToPixels(polygons, topLat, leftLong, botLat, rightLong, IMAGE_SIZE):
    xDist = (rightLong - leftLong) / IMAGE_SIZE
    yDist = (topLat - botLat) / IMAGE_SIZE

    ret = []

    for p in polygons:
        newP = []

        for pt in p:
            t = (int((pt[0] - leftLong) // xDist), int((topLat - pt[1]) // yDist))

            # make sure we are within the boarder
            if t[0] == IMAGE_SIZE:
                t = (IMAGE_SIZE - 1, t[1])
            if t[1] == IMAGE_SIZE:
                t = (t[0], IMAGE_SIZE - 1)

            newP.append(t)

            if t[0] < 0 or t[0] > IMAGE_SIZE - 1 or t[1] < 0 or t[1] > IMAGE_SIZE - 1:
                print(t, pt)

        ret.append(newP)

    return ret

