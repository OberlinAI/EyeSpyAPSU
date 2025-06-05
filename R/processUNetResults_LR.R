library(data.table)
library(ggplot2)

# this function parses all of the individual results, averages across random seeds,
# and returns a DataFrame (in *wide* format) with one row per hyperparameter combination and pixel threshold 
# (where the threshold is used to determine whether an image contains a windmill)
parseLogs <- function(prefix, layers) {
  files <- list.files(pattern = paste0(prefix, ".*"))
  started <- FALSE
  
  # read in all of the files
  frames <- list()
  count <- 1
  
  for (file in files) {
    nameSplit <- strsplit(file, "_")[[1]]
    
    if (length(nameSplit) == 6) {
      seed <- gsub(".csv", "", nameSplit[6])
      LR <- gsub("LR", "", nameSplit[5])
      LR <- as.character(as.numeric(LR))
      
      # read in this data
      data <- fread(file, sep=",", header=TRUE)
      
      data$Layers <- layers
      data$Seed <- seed
      data$LR <- LR
      frames[[count]] <- data
      count <- count + 1
    }
  }
  
  # combine all the files into one
  merged <- rbindlist(frames)
  print(nrow(merged))
  
  logs <- data.table(merged)[, .(
    mean(Layers),
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
  by=.(LR, Limit)]
  names(logs)[3:ncol(logs)] <- c("Layers", "TP", "FP", "FN", "TN",  "Acc", "Recall_P", "Precision_P", "Recall_N", "Precision_N", "BalancedAcc", "F1", "MCC", "N")
  
  return(logs)
}

# this function parses all of the individual results, averages across random seeds,
# and returns a DataFrame (in *long* format) with one observation per hyperparameter combination and pixel threshold 
# (where the threshold is used to determine whether an image contains a windmill)
parseLogsLong <- function(prefix, layers) {
  files <- list.files(pattern = paste0(prefix, ".*"))
  started <- FALSE
  
  # read in all of the files
  frames <- list()
  count <- 1
  
  for (file in files) {
    nameSplit <- strsplit(file, "_")[[1]]
    
    if (length(nameSplit) == 6) {
      seed <- gsub(".csv", "", nameSplit[6])
      LR <- gsub("LR", "", nameSplit[5])
      LR <- as.character(as.numeric(LR))
      
      # read in this data
      data <- fread(file, sep=",", header=TRUE)
      
      data$Layers <- layers
      data$Seed <- seed
      data$LR <- LR
      frames[[count]] <- data
      count <- count + 1
    }
  }
  
  # combine all the files into one
  merged <- rbindlist(frames)
  print(nrow(merged))
  
  # average the results for each limit
  limitList <- list()
  measuresList <- list()
  averageList <- list()
  maxList <- list()
  minList <- list()
  layersList <- list()
  rateList <- list()
  index <- 1
  
  for (rate in sort(unique(merged$LR))) {
    for (limit in sort(unique(merged$Limit))) {
      mergedSub <- merged[merged$LR == rate & merged$Limit == limit,]
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "TP"
      averageList[[index]] <- mean(mergedSub$TP)
      maxList[[index]] <- max(mergedSub$TP)
      minList[[index]] <- min(mergedSub$TP)
      
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "FP"
      averageList[[index]] <- mean(mergedSub$FP)
      maxList[[index]] <- max(mergedSub$FP)
      minList[[index]] <- min(mergedSub$FP)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "FN"
      averageList[[index]] <- mean(mergedSub$FN)
      maxList[[index]] <- max(mergedSub$FN)
      minList[[index]] <- min(mergedSub$FN)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "TN"
      averageList[[index]] <- mean(mergedSub$TN)
      maxList[[index]] <- max(mergedSub$TN)
      minList[[index]] <- min(mergedSub$TN)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "Accuracy"
      averageList[[index]] <- mean(mergedSub$Acc)
      maxList[[index]] <- max(mergedSub$Acc)
      minList[[index]] <- min(mergedSub$Acc)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "Recall_P"
      averageList[[index]] <- mean(mergedSub$Recall_P)
      maxList[[index]] <- max(mergedSub$Recall_P)
      minList[[index]] <- min(mergedSub$Recall_P)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "Precision_P"
      averageList[[index]] <- mean(mergedSub$Precision_P)
      maxList[[index]] <- max(mergedSub$Precision_P)
      minList[[index]] <- min(mergedSub$Precision_P)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "Recall_N"
      averageList[[index]] <- mean(mergedSub$Recall_N)
      maxList[[index]] <- max(mergedSub$Recall_N)
      minList[[index]] <- min(mergedSub$Recall_N)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "Precision_N"
      averageList[[index]] <- mean(mergedSub$Precision_N)
      maxList[[index]] <- max(mergedSub$Precision_N)
      minList[[index]] <- min(mergedSub$Precision_N)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "BalancedAccuracy"
      averageList[[index]] <- mean(mergedSub$BalancedAcc)
      maxList[[index]] <- max(mergedSub$BalancedAcc)
      minList[[index]] <- min(mergedSub$BalancedAcc)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "F1"
      averageList[[index]] <- mean(mergedSub$F1)
      maxList[[index]] <- max(mergedSub$F1)
      minList[[index]] <- min(mergedSub$F1)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      layersList[[index]] <- mean(mergedSub$Layers)
      measuresList[[index]] <- "MCC"
      averageList[[index]] <- mean(mergedSub$MCC)
      maxList[[index]] <- max(mergedSub$MCC)
      minList[[index]] <- min(mergedSub$MCC)
      index <- index + 1
    }
  }
  
  # create the data frame
  logs <- data.frame(unlist(rateList), unlist(layersList), unlist(limitList), unlist(measuresList), unlist(averageList), unlist(maxList), unlist(minList))
  names(logs) <- c("LR", "Layers", "Limit", "Measure", "Average", "Max", "Min")
  
  return(logs)
}

# this function creates the box plots used to evaluate the hyperparameter combinations (in Appendix C)
makeLRBarPlots <- function(logs2, logs3, logs4, logs5, measures, limit, zoom) {
  # combine all of the results in one DataFrame
  logsList <- list()
  logsList[[1]] <- logs2
  logsList[[2]] <- logs3
  logsList[[3]] <- logs4
  logsList[[4]] <- logs5
  logs <- rbindlist(logsList)
  
  # subset based on the performance measure and pixel threshold
  subdata <- logs[logs$Measure %in% measures & logs$Limit == limit,]
  
  # prepare formatting for display
  subdata$Layers <- as.factor(subdata$Layers)
  subdata$Measure <- as.factor(subdata$Measure)
  subdata$LR <- as.factor(as.numeric(as.character(subdata$LR)))
  subdata$L <- as.factor(subdata$Limit)
  
  # determine whether we are zooming in on the results or not
  lower_y <- ifelse(zoom, 0.9, 0.0)
  y_gap <- ifelse(zoom, 0.02, 0.25)
  
  print(ggplot(data=subdata, aes(x = LR, y = Average, fill=Layers)) +
          geom_bar(stat="identity", position=position_dodge()) +
          coord_cartesian(ylim=c(lower_y, 1.0)) +
          scale_y_continuous(breaks = seq(lower_y, 1.0, y_gap)) +
          ggtitle("U-Net Grid Search Hyperparameter Tuning") +
          xlab("Learning Rate") + 
          ylab("Balanced Accuracy") + 
          theme_bw() +
          scale_fill_grey(start = .8, end=0) +
          theme(legend.position="top", legend.margin=margin(t=0, unit="cm"), plot.title=element_text(hjust=0.5), 
                plot.subtitle=element_text(hjust=0.5), text=element_text(size=20), axis.text.x=element_text(size=11)) +
          guides(fill=guide_legend(nrow=2, title_position="top")))
}

### The script starts here ###

# set some constants
DATAFOLDER <- "/mnt/windmills/results/UNet/"  # TODO: change this to your results directory
UNDERSAMPLING_RATIO <- 1

# go to the correct folder
setwd(DATAFOLDER)

# parse the 2 hidden layer results
NUM_LAYERS = 2
LOGFILE_PREFIX <- paste0("results_UNetTrainTest_", NUM_LAYERS, "Layers_", UNDERSAMPLING_RATIO, "Ratio")
logs_unet2_1ratio <- parseLogs(LOGFILE_PREFIX, NUM_LAYERS)
logsLong_unet2_1ratio <- parseLogsLong(LOGFILE_PREFIX, NUM_LAYERS)

# parse the 3 hidden layer results
NUM_LAYERS = 3
LOGFILE_PREFIX <- paste0("results_UNetTrainTest_", NUM_LAYERS, "Layers_", UNDERSAMPLING_RATIO, "Ratio")
logs_unet3_1ratio <- parseLogs(LOGFILE_PREFIX, NUM_LAYERS)
logsLong_unet3_1ratio <- parseLogsLong(LOGFILE_PREFIX, NUM_LAYERS)

# parse the 4 hidden layer results
NUM_LAYERS = 4
LOGFILE_PREFIX <- paste0("results_UNetTrainTest_", NUM_LAYERS, "Layers_", UNDERSAMPLING_RATIO, "Ratio")
logs_unet4_1ratio <- parseLogs(LOGFILE_PREFIX, NUM_LAYERS)
logsLong_unet4_1ratio <- parseLogsLong(LOGFILE_PREFIX, NUM_LAYERS)

# parse the 5 hidden layer results
NUM_LAYERS = 5
LOGFILE_PREFIX <- paste0("results_UNetTrainTest_", NUM_LAYERS, "Layers_", UNDERSAMPLING_RATIO, "Ratio")
logs_unet5_1ratio <- parseLogs(LOGFILE_PREFIX, NUM_LAYERS)
logsLong_unet5_1ratio <- parseLogsLong(LOGFILE_PREFIX, NUM_LAYERS)

# print the best Balanced Accuracy performance for each number of hidden layers (using the best pixel threshold)
logs_unet2_1ratio[logs_unet2_1ratio$BalancedAcc == max(logs_unet2_1ratio$BalancedAcc),]
logs_unet3_1ratio[logs_unet3_1ratio$BalancedAcc == max(logs_unet3_1ratio$BalancedAcc),]
logs_unet4_1ratio[logs_unet4_1ratio$BalancedAcc == max(logs_unet4_1ratio$BalancedAcc),]
logs_unet5_1ratio[logs_unet5_1ratio$BalancedAcc == max(logs_unet5_1ratio$BalancedAcc),]

# create the box plots (from Appendix C)
makeLRBarPlots(logsLong_unet2_1ratio, logsLong_unet3_1ratio, logsLong_unet4_1ratio, logsLong_unet5_1ratio, c("BalancedAccuracy"), 400, FALSE)
makeLRBarPlots(logsLong_unet2_1ratio, logsLong_unet3_1ratio, logsLong_unet4_1ratio, logsLong_unet5_1ratio, c("BalancedAccuracy"), 400, TRUE)