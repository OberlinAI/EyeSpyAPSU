'''
This program performs the learning for creating and using UNet models.
'''

from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, UpSampling2D, concatenate
from keras import backend as K

import numpy as np
import os
import math
import PIL
import random
from shapely.geometry import Polygon, Point
import sys
import tensorflow as tf
import time
import UTMConverter
import WindmillLearningUtils

# define the constants of our experiments
ALGORITHM = "UNet"
IMAGE_SIZE = 256   # the width/length of each image in pixels (256 x 256)
RANDOM_SEED = 12345    # a random seed to use for randomization
RATIO = 1    # the undersampling ratio to use
PERCENT_ALL_FOR_TRAINING = 0.5    # use 50% of all of the data for training
TRAIN_FROM_MAX_COUNTIES = False    # use the smallest counties for training, if training with specific counties (i.e., using the trainNetwork function below)
TRAINING_PERCENT = 0.8    # how much of the training images to use for actual training (vs. validation)
VALIDATION_PERCENT = 0.2    # how much of the training images to use for validation (vs. actual training)
LAYERS = 4    # number of layers in the UNet
LEARNING_RATE = 0.00007    # the learning rate to use
MAX_STEPS = 1000    # maximum number of epochs to run
ADDITIONAL_STEPS = 25    # patience = number of epochs to keep running after a new best model is found
BATCH_SIZE = 16    # the batch size to use during training
THRESHOLD = 0.5    # percentage threshold used to convert a predicted probability to whether a pixel has a windmill or not


'''Replace these folders with the locations that you prefer'''
# the location of the training images
TRAINING_IMAGE_FOLDER = "/work/windmills/" + str(IMAGE_SIZE) + "/"
LABELS_FILE = TRAINING_IMAGE_FOLDER + "Windmills_Polygons_" + str(IMAGE_SIZE) + ".csv"    # from the data folder in the GitHub repository

# the location of the testing images (organized by county)
TESTING_IMAGE_FOLDER = TRAINING_IMAGE_FOLDER + "counties/"

# the location to store the trained models
MODEL_FOLDER = "/mnt/windmills/models/" + ALGORITHM + "/"
MODEL_START = "model_ep"
MODEL_EXTENSION = ".ckpt"

# the location to store the predictions
RESULTS_FOLDER = "/mnt/windmills/results/" + ALGORITHM + "/"

# the location to store the predicted images (for image segmentation)
TRAINING_RESULT_FOLDER = "/mnt/windmills/models/trainImages/" + ALGORITHM + "/"


def make_unet(input_tensor, depth=1, img_size=256, img_depth=1, output_depth=2):
    layers = []
    layer_depth = 64
    conv_size = 3  # TODO come back if depth gets more than 5
    numConvsPerLayer = 2
    x = input_tensor
    for i in range(depth):
        for j in range(numConvsPerLayer):
            x = Conv2D(layer_depth, (conv_size, conv_size), padding='same', activation='relu')(x)
        layers.append(x)
        x = MaxPooling2D((2, 2))(x)
        layer_depth *= 2
    x = Conv2D(layer_depth, (conv_size, conv_size), activation="relu", padding="same")(x)
    for i in range(depth):
        layer_depth //= 2
        x = UpSampling2D(size=(2, 2))(x)
        x = concatenate([x, layers[depth - i - 1]])
        for j in range(numConvsPerLayer):
            x = Conv2D(layer_depth, (conv_size, conv_size), padding="same", activation='relu')(x)
    x = Conv2D(output_depth, (1, 1), activation="relu", padding="same")(x)
    return x


def trainNetwork(splitImages, splitLabels, splitIds):
    # reset the graph everytime we want to rebuild this window so we don't hit errors
    K.clear_session()
    tf.reset_default_graph()

    # create the data pipeline
    segmentation_input = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    segmentation_labels = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 2))

    # create the network
    segmentation_output = make_unet(segmentation_input, depth=LAYERS, img_size=IMAGE_SIZE, img_depth=3)
    seg_logs = tf.nn.softmax(segmentation_output)

    # create the training process

    # we need a fancier loss function
    # 0 is our empty class
    segmentation_loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(labels=segmentation_labels, logits=segmentation_output))

    # keep track of how many steps we've taken for weight decay
    starter_learning_rate = LEARNING_RATE
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               MAX_STEPS * int(math.ceil(len(splitImages[0]) / BATCH_SIZE)),
                                               0.9, staircase=True)

    # create the optimizer
    segmentation_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(segmentation_loss,
                                                                            global_step=global_step)

    # create the saver
    segmentation_saver = tf.train.Saver()

    # create the model directory if needed
    minMax = "max" if TRAIN_FROM_MAX_COUNTIES else "min"
    modelFolder = MODEL_FOLDER + str(LAYERS) + "Layers/" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + "/"
    if not os.path.exists(modelFolder):
        os.mkdir(modelFolder)

    # save the current working directory
    CWD = os.getcwd()

    # go to the directory where we want to log the training data
    trainFolder = TRAINING_RESULT_FOLDER + str(LAYERS) + "Layers/" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + "/"
    if not os.path.exists(trainFolder):
        os.mkdir(trainFolder)
    os.chdir(trainFolder)

    # setup the GPU resources
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("Beginning to train segmentation...")

        drawnLabels = False
        epoch = 0
        bestValidLoss = math.inf
        lastBest = 0
        while epoch < MAX_STEPS and epoch < lastBest + ADDITIONAL_STEPS:
            batch_start = time.time()

            # calculate the test loss
            if len(splitImages[2]) > 0:
                test_loss = 0
                nextIndex = 0
                while nextIndex >= 0:
                    batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[2], splitLabels[2],
                                                                                      splitIds[2], BATCH_SIZE, nextIndex)

                    # evaluate the batch
                    feed_dict = {segmentation_input: batchX, segmentation_labels: batchY}
                    test_loss_batch = sess.run(segmentation_loss, feed_dict=feed_dict)
                    test_loss += test_loss_batch

                print("Test loss for epoch " + str(epoch) + ": " + str(test_loss / len(splitImages[2])))

            # calculate the validation loss
            valid_loss = 0
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[1], splitLabels[1],
                                                                                  splitIds[1], BATCH_SIZE, nextIndex)

                feed_dict = {segmentation_input: batchX, segmentation_labels: batchY}
                valid_loss_batch, seg_logits = sess.run([segmentation_loss, seg_logs], feed_dict=feed_dict)
                valid_loss += valid_loss_batch

                ###### H A C K #######
                if nextIndex == BATCH_SIZE:
                    numImages = len(batchY)
                    ending = "_ep" + str(epoch) + ".jpeg"
                    for j in range(numImages):
                        im = PIL.Image.fromarray(seg_logits[j][:, :, 0] * 255.)
                        im = im.convert("RGB")
                        im.save(batchIds[j].replace(".tif", ending))

                        if not drawnLabels:
                            lab = PIL.Image.fromarray(batchY[j][:, :, 0] * 255.)
                            lab = lab.convert("RGB")
                            lab.save(batchIds[j].replace(".tif", "_label.jpeg"))

            # note that we've for sure drawn the true labels
            drawnLabels = True

            print("Valid loss for epoch " + str(epoch) + ": " + str(valid_loss / len(splitImages[1])))

            if valid_loss < bestValidLoss:
                # update that we found a new best model
                bestValidLoss = valid_loss
                lastBest = epoch

                # save the model to file
                save_path = segmentation_saver.save(sess, modelFolder + MODEL_START + str(epoch) + MODEL_EXTENSION)
                print("Model saved!", save_path)

            avg_loss = 0
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[0], splitLabels[0],
                                                                                  splitIds[0], BATCH_SIZE, nextIndex)

                feed_dict = {segmentation_input: batchX, segmentation_labels: batchY}
                _, current_loss, seg_logits = sess.run([segmentation_optimizer, segmentation_loss, seg_logs], feed_dict=feed_dict)
                avg_loss += current_loss

            print("epoch " + str(epoch) + " complete with loss " + str(avg_loss / len(splitImages[0])))
            print("epoch time:" + str(time.time() - batch_start))

            epoch += 1

        # go back to the original working directory
        os.chdir(CWD)


def testNetwork(counties):
    # reset the graph everytime we want to rebuild this window so we don't hit errors
    K.clear_session()
    tf.reset_default_graph()

    # create the data pipeline
    segmentation_input = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))

    # create the network
    segmentation_output = make_unet(segmentation_input, depth=LAYERS, img_size=IMAGE_SIZE, img_depth=3)
    seg_logs = tf.nn.softmax(segmentation_output)
    finalPred = tf.cast(seg_logs >= THRESHOLD, dtype=tf.int32)
    sumPred = tf.reduce_sum(finalPred [:,:,:,1:], axis=(1, 2, 3))

    # create the restorer
    saver = tf.train.Saver()

    # create the model directory if needed
    minMax = "max" if TRAIN_FROM_MAX_COUNTIES else "min"
    modelFolder = MODEL_FOLDER + str(LAYERS) + "Layers/" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" +  \
                  str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + "/"

    # setup the GPU resources
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # restore the network
        print(modelFolder)
        lastCheckpoint = tf.train.get_checkpoint_state(modelFolder)
        modelPath = lastCheckpoint.model_checkpoint_path
        saver.restore(sess, modelPath)

        for county in counties:
            # read in the correct data
            countyFolder = TESTING_IMAGE_FOLDER + str(county) + "/"
            countyIds = WindmillLearningUtils.readOnlyIds(countyFolder)
            print(countyFolder, len(countyIds))

            # make predictions for each test instance
            counts = []

            nextIndex = 0
            while nextIndex >= 0:
                batchIds, nextIndex = WindmillLearningUtils.batchIds(countyIds, BATCH_SIZE, nextIndex)
                batchX = WindmillLearningUtils.readImagesByIds(countyFolder, batchIds)
                batchY = WindmillLearningUtils.readLabelsByIds(batchIds)
                seg_logits, sums = sess.run([seg_logs, sumPred], feed_dict={segmentation_input: batchX})

                for i in range(len(batchY)):
                    actual = batchY[i][0]
                    predicted = sums[i]

                    counts.append((predicted, actual, batchIds[i]))

            # create the ROC data
            limits = [100 * i for i in range(100)]
            bestWrong = []
            bestBalanced = 0
            fewestErrors = len(counts)
            fewestWrong = []
            results = []
            for limit in limits:
                tp = 0
                fp = 0
                fn = 0
                tn = 0
                wrong = []
                for count in counts:
                    if count[0] > limit:
                        if count[1] > 0:
                            tp += 1
                        else:
                            fp += 1
                            wrong.append(count[2])
                    else:
                        if count[1] > 0:
                            fn += 1
                            wrong.append(count[2])
                        else:
                            tn += 1

                print("Limit:", limit, "TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

                acc = (tp + tn) / (tp + fp + fn + tn)
                recallP = tp / (tp + fn) if tp + fn > 0 else 0
                precisionP = tp / (tp + fp) if tp + fp > 0 else 0
                recallN = tn / (tn + fp) if tn + fp > 0 else 0
                precisionN = tn / (tn + fn) if tn + fn > 0 else 0
                balanced = 0.5 * (recallP + recallN)
                f1 = 2 * recallP * precisionP / (recallP + precisionP) if recallP + precisionP > 0 else 0
                mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if tp + fp > 0 and tn + fn > 0 and tp + fn > 0 and tn + fp > 0 else 0
                print("Acc:", acc, "RecallP:", recallP, "PrecisionP:", precisionP, "RecallN:", recallN, "PrecisionN:", precisionN, "Balanced:", balanced, "F1:", f1, "MCC:", mcc)

                t = (limit, tp, fp, fn, tn, acc, recallP, precisionP, recallN, precisionN, balanced, f1, mcc)
                results.append(t)

                if balanced > bestBalanced:
                    bestBalanced = balanced
                    bestWrong = wrong

                if fp + fn < fewestErrors:
                    fewestErrors = fp + fn
                    fewestWrong = wrong

            with open(RESULTS_FOLDER + "bestBalanced_county" + str(county) + "_" + ALGORITHM + "_" + str(LAYERS) + "_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + ("_v3" if extra else "") + ".csv", "w") as file:
                    #RESULTS_FOLDER + "bestBalanced_UNet_" + str(LAYERS) + "Layers_" + str(RANDOM_SEED) + ".csv", "w") as file:
                for wrongId in bestWrong:
                    file.write(wrongId + "\n")

            with open(RESULTS_FOLDER + "fewestWrong_county" + str(county) + "_" + ALGORITHM + "_" + str(LAYERS) + "_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + ("_v3" if extra else "") + ".csv", "w") as file:
                    #RESULTS_FOLDER + "fewestWrong_UNet_" + str(LAYERS) + "Layers_" + str(RANDOM_SEED) + ".csv", "w") as file:
                for wrongId in fewestWrong:
                    file.write(wrongId + "\n")

            with open(RESULTS_FOLDER + "results_county" + str(county) + "_" + ALGORITHM + "_" + str(LAYERS) + "_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + ("_v3" if extra else "") + ".csv", "w") as file:
                    #RESULTS_FOLDER + "results_UNet_" + str(LAYERS) + "Layers_" + str(RANDOM_SEED) + ".csv", "w") as file:
                file.write("Limit,TP,FP,FN,TN,Acc,Recall_P,Precision_P,Recall_N,Precision_N,BalancedAcc,F1,MCC\n")
                for result in results:
                    file.write(str(result[0]) + "," + ",".join(["{0:.6f}".format(num) for num in result[1:]]) + "\n")

            with open(RESULTS_FOLDER + "predictions_county" + str(county) + "_" + ALGORITHM + "_" + str(LAYERS) + "_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + ("_v3" if extra else "") + ".csv", "w") as file:
                file.write("Id,Actual,Prediction\n")
                for count in counts:
                    L = [count[2], str(count[1]), "1" if count[0] > 100 else "0"]
                    file.write(",".join(L) + "\n")


def readWindmills():
    windmills = {}

    with open(LABELS_FILE, "r") as file:
        skipHeader = True
        for line in file:
            data = line.strip().split(";")

            if skipHeader:
                skipHeader = False
            else:
                county = int(data[0])

                polygons = []
                polygonStr = data[-1]
                split = polygonStr.split("))")
                for i in range(len(split) - 1):
                    polygons.append(split[i].replace("Polygon ((", "").replace(", ", ","))
                data[-1] = WindmillLearningUtils.parsePolygons(polygons)

                if county not in windmills:
                    windmills[county] = []

                windmills[county].append(data)

    return windmills



def tune(splitImages, splitLabels, splitIds):
    # reset the graph everytime we want to rebuild this window so we don't hit errors
    K.clear_session()
    tf.reset_default_graph()

    # create the data pipeline
    segmentation_input = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    segmentation_labels = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 2))

    # create the network
    segmentation_output = make_unet(segmentation_input, depth=LAYERS, img_size=IMAGE_SIZE, img_depth=3)
    seg_logs = tf.nn.softmax(segmentation_output)
    finalPred = tf.cast(seg_logs >= THRESHOLD, dtype=tf.int32)
    sumPred = tf.reduce_sum(finalPred [:,:,:,1:], axis=(1, 2, 3))

    # create the training process

    # we need a fancier loss function
    # 0 is our empty class
    segmentation_loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(labels=segmentation_labels, logits=segmentation_output))

    # keep track of how many steps we've taken for weight decay
    starter_learning_rate = LEARNING_RATE
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               MAX_STEPS * int(math.ceil(len(splitImages[0]) / BATCH_SIZE)),
                                               0.9, staircase=True)

    # create the optimizer
    segmentation_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(segmentation_loss,
                                                                            global_step=global_step)

    # create the saver
    # segmentation_saver = tf.train.Saver()

    # create the model directory if needed
    # modelFolder = MODEL_FOLDER + str(LAYERS) + "Layers/" + str(RATIO) + "Ratio_" + str(RANDOM_SEED) + "TrainTest/"
    # if not os.path.exists(modelFolder):
    #     os.mkdir(modelFolder)

    # setup the GPU resources
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        print("Beginning to train segmentation...")

        epoch = 0
        bestValidLoss = math.inf
        lastBest = 0
        bestCounts = []
        while epoch < MAX_STEPS and epoch < lastBest + ADDITIONAL_STEPS:
            batch_start = time.time()

            avg_loss = 0
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[0], splitLabels[0],
                                                                                  splitIds[0], BATCH_SIZE, nextIndex)

                feed_dict = {segmentation_input: batchX, segmentation_labels: batchY}
                _, current_loss, seg_logits = sess.run([segmentation_optimizer, segmentation_loss, seg_logs],
                                                       feed_dict=feed_dict)
                avg_loss += current_loss

            print("epoch " + str(epoch) + " complete with loss " + str(avg_loss / len(splitImages[0])))
            print("epoch time:" + str(time.time() - batch_start))

            # calculate the validation loss
            valid_loss = 0
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[1], splitLabels[1],
                                                                                  splitIds[1], BATCH_SIZE, nextIndex)

                valid_loss_batch = sess.run(segmentation_loss, feed_dict={segmentation_input: batchX, segmentation_labels: batchY})
                valid_loss += valid_loss_batch

            #print("Valid loss for epoch " + str(epoch) + ": " + str(valid_loss / len(splitImages[1])))

            if valid_loss < bestValidLoss:
                # update that we found a new best model
                bestValidLoss = valid_loss
                lastBest = epoch

                # save the model to file
                # save_path = segmentation_saver.save(sess, modelFolder + MODEL_START + str(epoch) + MODEL_EXTENSION)
                # print("Model saved!", save_path)

                # make predictions for each test instance
                bestCounts = []
                nextIndex = 0
                while nextIndex >= 0:
                    batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[2], splitLabels[2],
                                                                                      splitIds[2], BATCH_SIZE,
                                                                                      nextIndex)
                    seg_logits, sums = sess.run([seg_logs, sumPred], feed_dict={segmentation_input: batchX})

                    for i in range(len(batchY)):
                        actual = batchY[i][0]
                        predicted = sums[i]

                        bestCounts.append((predicted, actual, batchIds[i]))

            print("Valid loss for epoch " + str(epoch) + ": " + str(valid_loss / len(splitImages[1])) + "\tBest: " + str(bestValidLoss / len(splitImages[1])) + " (Epoch: " + str(lastBest) + ")")

            # calculate the test loss
            # if len(splitImages[2]) > 0:
            #     test_loss = 0
            #     nextIndex = 0
            #     while nextIndex >= 0:
            #         batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[2],
            #                                                                           splitLabels[2],
            #                                                                           splitIds[2], BATCH_SIZE,
            #                                                                           nextIndex)
            #
            #         test_loss_batch = sess.run(segmentation_loss,
            #                                    feed_dict={segmentation_input: batchX, segmentation_labels: batchY})
            #         test_loss += test_loss_batch
            #
            #     print("Test loss for epoch " + str(epoch) + ": " + str(test_loss / len(splitImages[2])))

            epoch += 1

        # restore the network
        # lastCheckpoint = tf.train.get_checkpoint_state(modelFolder)
        # modelPath = lastCheckpoint.model_checkpoint_path
        # segmentation_saver.restore(sess, modelPath)

        # create the ROC data
        limits = [100 * i for i in range(100)]
        bestWrong = []
        bestBalanced = 0
        fewestErrors = len(bestCounts)
        fewestWrong = []
        results = []
        for limit in limits:
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            wrong = []
            for count in bestCounts:
                if count[0] > limit:
                    if count[2].startswith("windmill"):
                        tp += 1
                    else:
                        fp += 1
                        wrong.append(count[2])
                else:
                    if count[2].startswith("windmill"):
                        fn += 1
                        wrong.append(count[2])
                    else:
                        tn += 1

            print("Limit:", limit, "TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)

            acc = (tp + tn) / (tp + fp + fn + tn)
            recallP = tp / (tp + fn) if tp + fn > 0 else 0
            precisionP = tp / (tp + fp) if tp + fp > 0 else 0
            recallN = tn / (tn + fp) if tn + fp > 0 else 0
            precisionN = tn / (tn + fn) if tn + fn > 0 else 0
            balanced = 0.5 * (recallP + recallN)
            f1 = 2 * recallP * precisionP / (recallP + precisionP) if recallP + precisionP > 0 else 0
            mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (
                        tn + fn)) if tp + fp > 0 and tn + fn > 0 and tp + fn > 0 and tn + fp > 0 else 0
            print("Acc:", acc, "RecallP:", recallP, "PrecisionP:", precisionP, "RecallN:", recallN, "PrecisionN:",
                  precisionN, "Balanced:", balanced, "F1:", f1, "MCC:", mcc)

            t = (limit, tp, fp, fn, tn, acc, recallP, precisionP, recallN, precisionN, balanced, f1, mcc)
            results.append(t)

            if balanced > bestBalanced:
                bestBalanced = balanced
                bestWrong = wrong

            if fp + fn < fewestErrors:
                fewestErrors = fp + fn
                fewestWrong = wrong

        with open(RESULTS_FOLDER + "bestBalanced_" + ALGORITHM + "TrainTest_" + str(LAYERS) + "Layers_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(RANDOM_SEED) + ".csv", "w") as file:
            for wrongId in bestWrong:
                file.write(wrongId + "\n")

        with open(RESULTS_FOLDER + "fewestWrong_" + ALGORITHM + "TrainTest_" + str(LAYERS) + "Layers_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(RANDOM_SEED) + ".csv", "w") as file:
            for wrongId in fewestWrong:
                file.write(wrongId + "\n")

        with open(RESULTS_FOLDER + "results_" + ALGORITHM + "TrainTest_" + str(LAYERS) + "Layers_" + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(RANDOM_SEED) + ".csv", "w") as file:
            file.write("Limit,TP,FP,FN,TN,Acc,Recall_P,Precision_P,Recall_N,Precision_N,BalancedAcc,F1,MCC\n")
            for result in results:
                file.write(str(result[0]) + "," + ",".join(["{0:.6f}".format(num) for num in result[1:]]) + "\n")


def main():
    if len(sys.argv) <= 5:
        print("Usage: Windmill" + ALGORITHM + " <seed> <mode> <ratio> <trainPerc> <other>")
        print("<mode> can be either --train or --test or --tune")
        print("<ratio> is the desired ratio between notwindmill and windmill images")
        print("<trainPerc> is the percentage of all windmills to use for training data (not including validation for --tune)")
        print("<other> is either max or min for --train and --test, else it is validation percent for --tune")
        sys.exit(-1)

    # get the random seed
    global RANDOM_SEED, RATIO, PERCENT_ALL_FOR_TRAINING, TRAIN_FROM_MAX_COUNTIES, TRAINING_PERCENT, VALIDATION_PERCENT, LEARNING_RATE, BATCH_SIZE, LAYERS
    RANDOM_SEED = int(sys.argv[1])
    random.seed(RANDOM_SEED)

    # are we in training or test mode?
    if sys.argv[2] == "--train":
        RATIO = int(sys.argv[3])

        PERCENT_ALL_FOR_TRAINING = float(sys.argv[4])
        if sys.argv[5] == "max":
            TRAIN_FROM_MAX_COUNTIES = True
        elif sys.argv[5] == "min":
            TRAIN_FROM_MAX_COUNTIES = False
        else:
            print("<other> needs to be max or min")

        if len(sys.argv) > 6:
            BATCH_SIZE = int(sys.argv[6])

        if len(sys.argv) > 7:
            LEARNING_RATE = float(sys.argv[7])

        if len(sys.argv) > 8:
            LAYERS = int(sys.argv[8])

        # process the training data
        allIds = WindmillLearningUtils.readOnlyIds(TRAINING_IMAGE_FOLDER)
        trainingIds, counties = WindmillLearningUtils.splitTrainingIdsByCounties(allIds, PERCENT_ALL_FOR_TRAINING,
                                                                                 TRAIN_FROM_MAX_COUNTIES)

        ids = WindmillLearningUtils.sampleIdsImbalance(trainingIds, RATIO)
        images = WindmillLearningUtils.readImagesByIds(TRAINING_IMAGE_FOLDER, ids)
        labels = WindmillLearningUtils.createSegmentationLabels(ids, TRAINING_IMAGE_FOLDER, IMAGE_SIZE)
        splitImages, splitLabels, splitIds = WindmillLearningUtils.splitData(images, labels, ids,
                                                                             TRAINING_PERCENT, VALIDATION_PERCENT)

        # train the network
        trainNetwork(splitImages, splitLabels, splitIds)
    elif sys.argv[2] == "--test":
        RATIO = int(sys.argv[3])
        PERCENT_ALL_FOR_TRAINING = float(sys.argv[4])
        if sys.argv[5] == "max":
            TRAIN_FROM_MAX_COUNTIES = True
        elif sys.argv[5] == "min":
            TRAIN_FROM_MAX_COUNTIES = False
        else:
            print("<other> needs to be max or min")
        if len(sys.argv) > 6:
            BATCH_SIZE = int(sys.argv[6])

        if len(sys.argv) > 7:
            LEARNING_RATE = float(sys.argv[7])

        if len(sys.argv) > 8:
            LAYERS = int(sys.argv[8])

        # process the testing data
        ids = WindmillLearningUtils.readOnlyIds(TRAINING_IMAGE_FOLDER)
        #trainingCounties, testingCounties = WindmillLearningUtils.splitCounties(ids, PERCENT_ALL_FOR_TRAINING, TRAIN_FROM_MAX_COUNTIES)
        trainingCounties, testingCounties = WindmillLearningUtils.splitCounties(ids, 0.5, TRAIN_FROM_MAX_COUNTIES) # TODO hardcoded for consistency when experimenting with smaller training percentages
        counties = [t[1] for t in testingCounties]

        # test on the testingCounties
        print("Test Counties:", testingCounties)
        testNetwork(counties)
    elif sys.argv[2] == "--tune":
        RATIO = int(sys.argv[3])
        TRAINING_PERCENT = float(sys.argv[4])
        VALIDATION_PERCENT = float(sys.argv[5])
        if len(sys.argv) > 6:
            BATCH_SIZE = int(sys.argv[6])

        if len(sys.argv) > 7:
            LEARNING_RATE = float(sys.argv[7])

        if len(sys.argv) > 8:
            LAYERS = int(sys.argv[8])

        # process the training
        allIds = WindmillLearningUtils.readOnlyIds(TRAINING_IMAGE_FOLDER)
        ids = WindmillLearningUtils.sampleIdsImbalance(allIds, RATIO)
        images = WindmillLearningUtils.readImagesByIds(TRAINING_IMAGE_FOLDER, ids)
        labels = WindmillLearningUtils.createSegmentationLabels(ids, TRAINING_IMAGE_FOLDER, IMAGE_SIZE)
        splitImages, splitLabels, splitIds = WindmillLearningUtils.splitData(images, labels, ids,
                                                                             TRAINING_PERCENT, VALIDATION_PERCENT)

        tune(splitImages, splitLabels, splitIds)
    elif sys.argv[2] == "--createBatches":
        WindmillLearningUtils.readAllImages(TRAINING_IMAGE_FOLDER, TRAINING_IMAGE_FOLDER + "binary/", 100000)


if __name__ == "__main__":
    main()

