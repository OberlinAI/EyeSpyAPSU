'''
This program performs the learning for creating and using VGG-16 models.
'''

import numpy as np
import os
import math
import random
import sys
import tensorflow as tf
import time
import WindmillLearningUtils

# define the constants of our experiments
ALGORITHM = "VGG"
IMAGE_SIZE = 256   # the width/length of each image in pixels (256 x 256)
RANDOM_SEED = 12345    # a random seed to use for randomization
RATIO = 9    # the undersampling ratio to use
PERCENT_ALL_FOR_TRAINING = 0.5    # use 50% of all of the data for training
TRAIN_FROM_MAX_COUNTIES = False    # use the smallest counties for training, if training with specific counties (i.e., using the trainNetwork function below)
TRAINING_PERCENT = 0.8    # how much of the training images to use for actual training (vs. validation)
VALIDATION_PERCENT = 0.2    # how much of the training images to use for validation (vs. actual training)
LEARNING_RATE = 0.0002    # the learning rate to use
MAX_STEPS = 1000    # maximum number of epochs to run
ADDITIONAL_STEPS = 25    # patience = number of epochs to keep running after a new best model is found
BATCH_SIZE = 128    # the batch size to use during training
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


def createNetwork(x):
    # Code by Han Shao
    conv1 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[7, 7], strides=2, padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)

    conv2 = tf.layers.conv2d(inputs=pool1, filters=256, kernel_size=[5, 5], strides=2, padding="same",
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=384, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv4 = tf.layers.conv2d(inputs=conv3, filters=384, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

    conv5 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[15, 15], padding="same", activation=tf.nn.relu)
    pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], strides=2)

    poolSize = IMAGE_SIZE // 32 - 1
    pool5_flat = tf.reshape(pool5, shape=[-1, poolSize * poolSize * 256])

    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    dropout1 = tf.layers.dropout(dense1, rate=0.5)
    dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)
    dropout2 = tf.layers.dropout(dense2, rate=0.5)

    output = tf.layers.dense(inputs=dropout2, units=1)
    predict = tf.nn.sigmoid(output)

    return output, predict


def trainNetwork(splitImages, splitLabels, splitIds):
    # create the data pipeline
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # create the network
    output, predict = createNetwork(x)

    # create the training process
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))

    # keep track of how many steps we've taken for weight decay
    starter_learning_rate = LEARNING_RATE
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               MAX_STEPS * int(math.ceil(len(splitImages[0]) / BATCH_SIZE)),
                                               0.9, staircase=True)

    # create the optimizer
    trainer = tf.train.AdamOptimizer(learning_rate)
    train_step = trainer.minimize(loss, global_step=global_step)

    # create the saver
    saver = tf.train.Saver()

    # create the model directory if needed
    minMax = "max" if TRAIN_FROM_MAX_COUNTIES else "min"
    modelFolder = MODEL_FOLDER + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" \
                  + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(RANDOM_SEED) + "/"
    if not os.path.exists(modelFolder):
        os.mkdir(modelFolder)

    # setup the GPU resources
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        epoch = 0
        elapsedTime = 0
        bestValidLoss = math.inf
        lastBest = 0
        while epoch < MAX_STEPS and epoch < lastBest + ADDITIONAL_STEPS:
            epoch += 1

            # get the training batch
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[0], splitLabels[0],
                                                                                  splitIds[0], BATCH_SIZE, nextIndex)
                # train on the batch
                startTime = time.perf_counter()
                sess.run(train_step, feed_dict={x: batchX, y: batchY})
                endTime = time.perf_counter()
                elapsedTime += endTime - startTime

            # calculate the validation loss
            valid_loss = 0
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[1], splitLabels[1],
                                                                                  splitIds[1], BATCH_SIZE, nextIndex)
                valid_loss += sess.run(loss, feed_dict={x: batchX, y: batchY})

            # is this our best validation loss?
            if valid_loss < bestValidLoss:
                # update that we found a new best model
                bestValidLoss = valid_loss
                lastBest = epoch

                # save the model to file
                save_path = saver.save(sess, modelFolder + MODEL_START + str(epoch) + MODEL_EXTENSION)
                print("Model saved!", save_path)

            if epoch % 10 == 0:
                trainMatrix, trainAcc, _ = WindmillLearningUtils.calcResults(x, predict, sess,
                                                                             splitImages[0], splitLabels[0],
                                                                             splitIds[0],
                                                                             BATCH_SIZE)

                validMatrix, validAcc, _ = WindmillLearningUtils.calcResults(x, predict, sess,
                                                                             splitImages[1], splitLabels[1],
                                                                             splitIds[1],
                                                                             BATCH_SIZE)

                print("Epoch", epoch, "Time:", elapsedTime / epoch)
                print("Train:", trainAcc, "Valid:", validAcc)

                print("TRAINING:")
                WindmillLearningUtils.printMatrix(trainMatrix)
                print("VALIDATION:")
                WindmillLearningUtils.printMatrix(validMatrix)


def testNetwork(counties):
    # create the data pipeline
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # create the network
    output, predict = createNetwork(x)

    # create the saver
    saver = tf.train.Saver()

    # create the model directory if needed
    minMax = "max" if TRAIN_FROM_MAX_COUNTIES else "min"
    modelFolder = MODEL_FOLDER + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + \
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
                p = sess.run(predict, feed_dict={x: batchX})

                for i in range(len(batchY)):
                    actual = batchY[i][0]
                    predicted = p[i]

                    counts.append((predicted, actual, batchIds[i]))

            # create the ROC data
            limits = [0.001 * i for i in range(1000)]
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

            with open(RESULTS_FOLDER + "bestBalanced_county" + str(county) + "_" + ALGORITHM + "_" + str(
                    RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(
                    RANDOM_SEED) +  ".csv", "w") as file:
                # RESULTS_FOLDER + "bestBalanced_UNet_" + str(LAYERS) + "Layers_" + str(RANDOM_SEED) + ".csv", "w") as file:
                for wrongId in bestWrong:
                    file.write(wrongId + "\n")

            with open(RESULTS_FOLDER + "fewestWrong_county" + str(county) + "_" + ALGORITHM + "_" + str(
                    RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(
                    RANDOM_SEED) + ".csv", "w") as file:
                # RESULTS_FOLDER + "fewestWrong_UNet_" + str(LAYERS) + "Layers_" + str(RANDOM_SEED) + ".csv", "w") as file:
                for wrongId in fewestWrong:
                    file.write(wrongId + "\n")

            with open(RESULTS_FOLDER + "results_county" + str(county) + "_" + ALGORITHM + "_" + str(
                    RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(
                    RANDOM_SEED) + ".csv", "w") as file:
                # RESULTS_FOLDER + "results_UNet_" + str(LAYERS) + "Layers_" + str(RANDOM_SEED) + ".csv", "w") as file:
                file.write("Limit,TP,FP,FN,TN,Acc,Recall_P,Precision_P,Recall_N,Precision_N,BalancedAcc,F1,MCC\n")
                for result in results:
                    file.write(str(result[0]) + "," + ",".join(["{0:.6f}".format(num) for num in result[1:]]) + "\n")

            with open(RESULTS_FOLDER + "predictions_county" + str(county) + "_" + ALGORITHM + "_" + str(
                    RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" + str(PERCENT_ALL_FOR_TRAINING) + minMax + "_" + str(
                    RANDOM_SEED) + ".csv", "w") as file:
                file.write("Id,Actual,Prediction\n")
                for count in counts:
                    L = [count[2], str(count[1]), str(count[0])]  # "1" if count[0] > THRESHOLD else "0"]
                    file.write(",".join(L) + "\n")


def tune(splitImages, splitLabels, splitIds):
    # create the data pipeline
    x = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # create the network
    output, predict = createNetwork(x)

    # create the training process
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output))

    # keep track of how many steps we've taken for weight decay
    starter_learning_rate = LEARNING_RATE
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               MAX_STEPS * int(math.ceil(len(splitImages[0]) / BATCH_SIZE)),
                                               0.9, staircase=True)

    # create the optimizer
    trainer = tf.train.AdamOptimizer(learning_rate)
    train_step = trainer.minimize(loss, global_step=global_step)

    # create the saver
    # saver = tf.train.Saver()

    # create the model directory if needed
    # modelFolder = MODEL_FOLDER + str(RATIO) + "Ratio_" + str(LEARNING_RATE) + "LR_" \
    #               + str(RANDOM_SEED) + "TrainTest/"
    # if not os.path.exists(modelFolder):
    #     os.mkdir(modelFolder)

    # setup the GPU resources
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # run the training
        epoch = 0
        elapsedTime = 0
        bestValidLoss = 99999999999
        lastBest = 0
        while epoch < MAX_STEPS and epoch < lastBest + ADDITIONAL_STEPS:
            epoch += 1

            # get the training batch
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[0], splitLabels[0],
                                                                                  splitIds[0], BATCH_SIZE, nextIndex)
                # train on the batch
                startTime = time.perf_counter()
                sess.run(train_step, feed_dict={x: batchX, y: batchY})
                endTime = time.perf_counter()
                elapsedTime += endTime - startTime

            # calculate the validation loss
            valid_loss = 0
            nextIndex = 0
            while nextIndex >= 0:
                batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[1], splitLabels[1],
                                                                                  splitIds[1], BATCH_SIZE, nextIndex)
                valid_loss += sess.run(loss, feed_dict={x: batchX, y: batchY})

            # is this our best validation loss?
            if valid_loss < bestValidLoss:
                # update that we found a new best model
                bestValidLoss = valid_loss
                lastBest = epoch

                # save the model to file
                # save_path = saver.save(sess, modelFolder + MODEL_START + str(epoch) + MODEL_EXTENSION)
                # print("Model saved!", save_path)

                # make predictions for each test instance
                bestCounts = []
                nextIndex = 0
                while nextIndex >= 0:
                    batchX, batchY, batchIds, nextIndex = WindmillLearningUtils.batch(splitImages[2], splitLabels[2],
                                                                                      splitIds[2], BATCH_SIZE,
                                                                                      nextIndex)
                    p = sess.run(predict, feed_dict={x: batchX})

                    for i in range(len(batchY)):
                        actual = batchY[i][0]
                        predicted = p[i]

                        bestCounts.append((predicted, actual, batchIds[i]))

            if epoch % 10 == 0:
                trainMatrix, trainAcc, _ = WindmillLearningUtils.calcResults(x, predict, sess,
                                                                             splitImages[0], splitLabels[0],
                                                                             splitIds[0],
                                                                             BATCH_SIZE)

                validMatrix, validAcc, _ = WindmillLearningUtils.calcResults(x, predict, sess,
                                                                             splitImages[1], splitLabels[1],
                                                                             splitIds[1],
                                                                             BATCH_SIZE)

                print("Epoch", epoch, "Time:", elapsedTime / epoch)
                print("Train:", trainAcc, "Valid:", validAcc)

                print("TRAINING:")
                WindmillLearningUtils.printMatrix(trainMatrix)
                print("VALIDATION:")
                WindmillLearningUtils.printMatrix(validMatrix)

        # restore the network
        # lastCheckpoint = tf.train.get_checkpoint_state(modelFolder)
        # modelPath = lastCheckpoint.model_checkpoint_path
        # segmentation_saver.restore(sess, modelPath)

        # create the ROC data
        limits = [0.001 * i for i in range(1000)]
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

        with open(RESULTS_FOLDER + "bestBalanced_" + ALGORITHM + "TrainTest_" + str(RATIO) + "Ratio_" + str(
                LEARNING_RATE) + "LR_" + str(RANDOM_SEED) + ".csv", "w") as file:
            for wrongId in bestWrong:
                file.write(wrongId + "\n")

        with open(RESULTS_FOLDER + "fewestWrong_" + ALGORITHM + "TrainTest_" + str(RATIO) + "Ratio_" + str(
                LEARNING_RATE) + "LR_" + str(RANDOM_SEED) + ".csv", "w") as file:
            for wrongId in fewestWrong:
                file.write(wrongId + "\n")

        with open(RESULTS_FOLDER + "results_" + ALGORITHM + "TrainTest_" + str(RATIO) + "Ratio_" + str(
                LEARNING_RATE) + "LR_" + str(RANDOM_SEED) + ".csv", "w") as file:
            file.write("Limit,TP,FP,FN,TN,Acc,Recall_P,Precision_P,Recall_N,Precision_N,BalancedAcc,F1,MCC\n")
            for result in results:
                file.write(str(result[0]) + "," + ",".join(["{0:.6f}".format(num) for num in result[1:]]) + "\n")


def main():
    if len(sys.argv) <= 5:
        print("Usage: Windmill" + ALGORITHM + " <seed> <mode> <ratio> <trainPerc> <other>")
        print("<mode> can be either --train or --test or --tune")
        print("<ratio> is the desired ratio between notwindmill and windmill images")
        print(
            "<trainPerc> is the percentage of all windmills to use for training data (not including validation for --tune)")
        print("<other> is either max or min for --train and --test, else it is validation percent for --tune")
        sys.exit(-1)

    # get the random seed
    global RANDOM_SEED, RATIO, PERCENT_ALL_FOR_TRAINING, TRAIN_FROM_MAX_COUNTIES, TRAINING_PERCENT, VALIDATION_PERCENT, LEARNING_RATE, BATCH_SIZE
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

        # process the training data
        allIds = WindmillLearningUtils.readOnlyIds(TRAINING_IMAGE_FOLDER)
        trainingIds, counties = WindmillLearningUtils.splitTrainingIdsByCounties(allIds, PERCENT_ALL_FOR_TRAINING,
                                                                                 TRAIN_FROM_MAX_COUNTIES)

        ids = WindmillLearningUtils.sampleIdsImbalance(trainingIds, RATIO)
        images = WindmillLearningUtils.readImagesByIds(TRAINING_IMAGE_FOLDER, ids)
        labels = WindmillLearningUtils.readLabelsByIds(ids)
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

        # process the testing data
        ids = WindmillLearningUtils.readOnlyIds(TRAINING_IMAGE_FOLDER)
        # trainingCounties, testingCounties = WindmillLearningUtils.splitCounties(ids, PERCENT_ALL_FOR_TRAINING, TRAIN_FROM_MAX_COUNTIES)
        trainingCounties, testingCounties = WindmillLearningUtils.splitCounties(ids, 0.5,
                                                                                TRAIN_FROM_MAX_COUNTIES)  # TODO hardcoded for consistency when experimenting with smaller training percentages
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

        # process the training
        allIds = WindmillLearningUtils.readOnlyIds(TRAINING_IMAGE_FOLDER)
        ids = WindmillLearningUtils.sampleIdsImbalance(allIds, RATIO)
        images = WindmillLearningUtils.readImagesByIds(TRAINING_IMAGE_FOLDER, ids)
        labels = WindmillLearningUtils.readLabelsByIds(ids)
        splitImages, splitLabels, splitIds = WindmillLearningUtils.splitData(images, labels, ids,
                                                                             TRAINING_PERCENT, VALIDATION_PERCENT)

        tune(splitImages, splitLabels, splitIds)


main()

