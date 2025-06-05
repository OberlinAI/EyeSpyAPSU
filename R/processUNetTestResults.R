library(plyr)
library(ggplot2)
library(data.table)

# this function creates and returns a list of unique random seeds used in the experiments
parseSeeds <- function(layers) {
  seeds <- list()
  seedIndex <- 1
  
  pattern <- paste0(LOGFILE_PREFIX, ".*")
  files <- list.files(pattern = pattern)
  for (file in files) {
    nameSplit <- strsplit(file, "_")[[1]]
    seedLoc <- 8
    seed <- gsub(".csv", "", nameSplit[seedLoc])
    
    if (!(seed %in% seeds)) {
      seeds[[seedIndex]] <- seed
      seedIndex <- seedIndex + 1
    }
  }
  
  return (unlist(seeds))
}

# this function creates and returns a list of test counties (with endings of FIPS codes as unique IDs)
parseCounties <- function() {
  counties <- list()
  countiesIndex <- 1
  
  pattern <- paste0(LOGFILE_PREFIX, ".*")
  files <- list.files(pattern = pattern)
  for (file in files) {
    nameSplit <- strsplit(file, "_")[[1]]
    
    county <- gsub("county", "", nameSplit[2])
    
    if (!(county %in% counties)) {
      counties[[countiesIndex]] <- county
      countiesIndex <- countiesIndex + 1
    }
  }
  
  return (unlist(counties))
}

# this function parses all of the individual results per random seed
# and returns a DataFrame (in *wide* format) with one row per seed/hyperparameter combination
parseLogs <- function(counties, layers, ratio) {
  # read in all of the files
  frames <- list()
  count <- 1
  
  for (county in counties) {
    prefix <- paste0("results_county", county, "_.*_", layers, "_.*", ratio, "Ratio_.*min.*")
    files <- list.files(pattern = prefix)
    print(paste0("County: ", county, " Files: ", length(files)))
    
    for (file in files) {
      nameSplit <- strsplit(file, "_")[[1]]
      
      seedIndex <- 8
      seed <- gsub(".csv", "", nameSplit[seedIndex])
      
      LR <- gsub("LR", "", nameSplit[6])
      LR <- as.character(as.numeric(LR))
      trainPerc <- gsub("min", "", nameSplit[7])
      
      # read in this data
      data <- fread(file, sep=",", header=TRUE)
      if (max(data$MCC) == 0) {
        print(paste0(seed, " ", LR, " ", trainPerc))
        next
      }
      
      data$Seed <- seed
      data$LR <- LR
      data$TrainPerc <- trainPerc
      data$county <- county
      frames[[count]] <- data
      count <- count + 1
    }
  }
    
  # combine all the files into one
  merged <- rbindlist(frames)
  print(nrow(merged))
  
  # sum up the results across all counties
  summed <- data.table(merged)[, .(
    sum(TP),
    sum(FP),
    sum(FN),
    sum(TN),
    .N
  ), 
  by=.(LR, TrainPerc, Limit, Seed)]
  names(summed)[5:ncol(summed)] <- c("TP", "FP", "FN", "TN",  "N")
  
  summed$Acc <- (summed$TP + summed$TN) / (summed$TP + summed$FP + summed$FN + summed$TN)
  summed$Recall_P <- summed$TP / (summed$TP + summed$FN)
  summed$Precision_P <- summed$TP / (summed$TP + summed$FP)
  summed$Recall_N <- summed$TN / (summed$TN + summed$FP)
  summed$Precision_N <- summed$TN / (summed$TN + summed$FN)
  summed$BalancedAcc <- (summed$Recall_P + summed$Recall_N) / 2
  summed$F1 <- ifelse(summed$Recall_P + summed$Precision_P > 0, 2 * (summed$Recall_P * summed$Precision_P) / (summed$Recall_P + summed$Precision_P), 0)
  summed$MCC <- ifelse(summed$TP + summed$FP > 0 & summed$TP + summed$FN > 0 & summed$TN + summed$FP > 0 & summed$TN + summed$FN > 0,
                       (summed$TP * summed$TN - summed$FP * summed$FN) / sqrt((summed$TP + summed$FP) * (summed$TP + summed$FN) * (summed$TN + summed$FP) * (summed$TN + summed$FN)),
                       0)
  
  return(summed)
}

# this function averages all of results (across seeds) from the parseLogs function
# it produces a DataFrame summarizing the results for each hyperparameter combination
averageLogs <- function(summed) {
  # average the results across all seeds
  logs <- data.table(summed)[, .(
    mean(TP),
    mean(FP),
    mean(FN),
    mean(TN),
    mean(Acc),
    mean(Recall_P),
    mean(Precision_P),
    mean(Recall_N),
    mean(Precision_N),
    mean(BalancedAcc),
    mean(F1),
    mean(MCC),
    .N
  ),
  by=.(LR, TrainPerc, Limit)]
  names(logs)[4:ncol(logs)] <- c("TP", "FP", "FN", "TN",  "Acc", "Recall_P", "Precision_P", "Recall_N", "Precision_N", "BalancedAcc", "F1", "MCC", "N")

  return(logs)
}

# this function parses all of the individual results per random seed, then averages the results across seeds,
# and returns a DataFrame (in *long* format) with one row per hyperparameter combination and pixel threshold
# Note:  this is used for plotting the results with ggplot
parseLogsLong <- function(counties, layers, ratio) {
  for (county in counties) {
    prefix <- paste0("results_county", county, "_.*_", layers, "_.*", ratio, "Ratio_.*min.*")
    files <- list.files(pattern = prefix)
    
    # read in all of the files
    frames <- list()
    count <- 1
    
    for (file in files) {
      nameSplit <- strsplit(file, "_")[[1]]
      
      seedIndex <- 8
      seed <- gsub(".csv", "", nameSplit[seedIndex])
      
      LR <- gsub("LR", "", nameSplit[6])
      LR <- as.character(as.numeric(LR))
      trainPerc <- gsub("min", "", nameSplit[7])
      
      # read in this data
      data <- fread(file, sep=",", header=TRUE)
      if (max(data$MCC) == 0) {
        next
      }
      
      data$Seed <- seed
      data$LR <- LR
      data$TrainPerc <- trainPerc
      data$county <- county
      frames[[count]] <- data
      count <- count + 1
    }
  }
    
  # combine all the files into one
  merged <- rbindlist(frames)
  
  # average the results for each limit
  rateList <- list()
  trainPercList <- list()
  limitList <- list()
  measuresList <- list()
  averageList <- list()
  maxList <- list()
  minList <- list()
  layersList <- list()
  index <- 1
  
  for (rate in sort(unique(merged$LR))) {
    for (trainPerc in sort(unique(merged$TrainPerc))) {
      for (limit in sort(unique(merged$Limit))) {
        mergedSub <- merged[merged$LR == rate & merged$Limit == limit & merged$TrainPerc == trainPerc,]
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "TP"
        tp <- data.frame(data.table(mergedSub)[, sum(TP), by=c("LR", "TrainPerc", "Limit","Seed")])
        tp <- tp[order(tp$Seed),]
        tp <- tp$V1
        averageList[[index]] <- mean(tp)
        maxList[[index]] <- max(tp)
        minList[[index]] <- min(tp)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "FP"
        fp <- data.frame(data.table(mergedSub)[, sum(FP), by=c("LR", "TrainPerc", "Limit","Seed")])
        fp <- fp[order(fp$Seed),]
        fp <- fp$V1
        averageList[[index]] <- mean(fp)
        maxList[[index]] <- max(fp)
        minList[[index]] <- min(fp)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "FN"
        fn <- data.frame(data.table(mergedSub)[, sum(FN), by=c("LR", "TrainPerc", "Limit","Seed")])
        fn <- fn[order(fn$Seed),]
        fn <- fn$V1
        averageList[[index]] <- mean(fn)
        maxList[[index]] <- max(fp)
        minList[[index]] <- min(fp)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "TN"
        tn <- data.frame(data.table(mergedSub)[, sum(TN), by=c("LR", "TrainPerc", "Limit","Seed")])
        tn <- tn[order(tn$Seed),]
        tn <- tn$V1
        averageList[[index]] <- mean(tn)
        maxList[[index]] <- max(tn)
        minList[[index]] <- min(tn)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "Accuracy"
        acc <- (tp + tn) / (tp + fp + fn + tn)
        averageList[[index]] <- mean(acc)
        maxList[[index]] <- max(acc)
        minList[[index]] <- min(acc)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "Recall_P"
        recall_p <- tp / (tp + fn)
        averageList[[index]] <- mean(recall_p)
        maxList[[index]] <- max(recall_p)
        minList[[index]] <- min(recall_p)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "Precision_P"
        precision_p <- ifelse(tp + fp > 0, tp / (tp + fp), 0)
        averageList[[index]] <- mean(precision_p)
        maxList[[index]] <- max(precision_p)
        minList[[index]] <- min(precision_p)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "Recall_N"
        recall_n <- tn / (tn + fp)
        averageList[[index]] <- mean(recall_n)
        maxList[[index]] <- max(recall_n)
        minList[[index]] <- min(recall_n)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "Precision_N"
        precision_n <- ifelse(tn + fn > 0, tn / (tn + fn), 0)
        averageList[[index]] <- mean(precision_n)
        maxList[[index]] <- max(precision_n)
        minList[[index]] <- min(precision_n)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "BalancedAccuracy"
        balanced <- (recall_p + recall_n) / 2
        averageList[[index]] <- mean(balanced)
        maxList[[index]] <- max(balanced)
        minList[[index]] <- min(balanced)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "F1"
        f1 <- ifelse(recall_p + precision_p > 0, 2 * (recall_p * precision_p) / (recall_p + precision_p), 0)
        averageList[[index]] <- mean(f1)
        maxList[[index]] <- max(f1)
        minList[[index]] <- min(f1)
        index <- index + 1
        
        rateList[[index]] <- rate
        trainPercList[[index]] <- trainPerc
        limitList[[index]] <- limit
        measuresList[[index]] <- "MCC"
        mcc <- ifelse(tp + fp > 0 & tp + fn > 0 & tn + fp > 0 & tn + fn > 0, (tp * tn - fp * fn) / sqrt((tp + fp) * (tp  + fn) * (tn + fp) * (tn + fn)), 0)
        averageList[[index]] <- mean(mcc)
        maxList[[index]] <- max(mcc)
        minList[[index]] <- min(mcc)
        index <- index + 1
      }
    }
  }
  
  # create the data frame
  logs <- data.frame(unlist(rateList), unlist(trainPercList), unlist(limitList), unlist(measuresList), unlist(averageList), unlist(maxList), unlist(minList))
  names(logs) <- c("LR", "TrainPerc", "Limit", "Measure", "Average", "Max", "Min")
  
  return(logs)
}

# this function creates a line of each performance measure as a function of the pixel threshold used to decide 
# whether an image contains a windmill
makeLimitPlots <- function(logs, measures) {
  data = logs[logs$Measure %in% measures,]
  data$AverageScaled <- 0
  for (measure in measures) {
    data$AverageScaled[data$Measure == measure] <- data$Average[data$Measure == measure] / max(data$Average[data$Measure == measure])
  }
  
  # data$Layers <- as.factor(data$Layers)
  data$Measure <- as.factor(data$Measure)
  
  print(ggplot(data=data, aes(x = Limit, y = Average, linetype=Measure)) +
          geom_line() +
          scale_x_continuous(name="Pixels", breaks = seq(0, 10000, 1000)) +
          scale_y_continuous(name="Performance") +
          ggtitle(paste0("Average Performance per Pixel Threshold")) +
          theme_bw() +
          guides(fill=guide_legend(nrow=2, title_position="top")))
}


### The script starts here ###

# set some constants
DATAFOLDER <- "/mnt/windmills/results/UNet/"   # TODO: change this to your results directory
LAYERS <- 4                                        # TODO: change this to your number of hidden layers in the UNet CNN
UNDERSAMPLING_RATIO <- 1                           # TODO: change this to your undersampling ratio (number of negative instances per positive instance)
LOGFILE_PREFIX <- paste0("results_.*_UNet_", LAYERS, "_", UNDERSAMPLING_RATIO, "Ratio_.*min.*")

# parse the seeds and counties
setwd(DATAFOLDER)
seeds <- parseSeeds(LAYERS)
counties <- parseCounties()

# parse the log files and average across seeds
logs_unet <- parseLogs(counties, LAYERS, UNDERSAMPLING_RATIO)
logs_unet_averages <- averageLogs(logs_unet)

# print the results
logs_unet_averages[logs_unet_averages$Limit == 400,]
logs_unet_averages[logs_unet_averages$Limit == 0,]

# plot the performances per pixel threshold
logsLong_unet_1ratio <- parseLogsLong(counties, LAYERS, UNDERSAMPLING_RATIO)
makeLimitPlots(logsLong_unet_1ratio, c("Recall_P", "Recall_N", "BalancedAccuracy", "MCC"))