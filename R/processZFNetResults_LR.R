library(data.table)
library(ggplot2)

# this function parses all of the individual results, averages across random seeds,
# and returns a DataFrame (in *wide* format) with one row per hyperparameter combination and probability threshold 
# (where the threshold called Limit is used to determine whether an image contains a windmill)
parseLogs <- function(prefix, undersampling_ratio) {
  files <- list.files(pattern = paste0(prefix, ".*"))
  started <- FALSE
  
  # read in all of the files
  frames <- list()
  count <- 1
  
  for (file in files) {
    nameSplit <- strsplit(file, "_")[[1]]
    
    if (length(nameSplit) == 5) {
      seed <- gsub(".csv", "", nameSplit[5])
      LR <- gsub("LR", "", nameSplit[4])
      LR <- as.character(as.numeric(LR))
      
      # read in this data
      data <- fread(file, sep=",", header=TRUE)
      
      data$Ratio <- undersampling_ratio
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
    mean(Ratio),
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
  names(logs)[3:ncol(logs)] <- c("Ratio", "TP", "FP", "FN", "TN",  "Acc", "Recall_P", "Precision_P", "Recall_N", "Precision_N", "BalancedAcc", "F1", "MCC", "N")
  
  return(logs)
}

# this function parses all of the individual results, averages across random seeds,
# and returns a DataFrame (in *long* format) with one observation per hyperparameter combination and probability threshold 
# (where the threshold called Limit is used to determine whether an image contains a windmill)
parseLogsLong <- function(prefix, undersampling_ratio) {
  files <- list.files(pattern = paste0(prefix, ".*"))
  started <- FALSE
  
  # read in all of the files
  frames <- list()
  count <- 1
  
  for (file in files) {
    nameSplit <- strsplit(file, "_")[[1]]
    # print(nameSplit)
    
    if (length(nameSplit) == 5) {
      seed <- gsub(".csv", "", nameSplit[5])
      LR <- gsub("LR", "", nameSplit[4])
      LR <- as.character(as.numeric(LR))
      
      # read in this data
      data <- fread(file, sep=",", header=TRUE)
      
      data$Ratio <- undersampling_ratio
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
  ratioList <- list()
  rateList <- list()
  index <- 1
  
  for (rate in sort(unique(merged$LR))) {
    for (limit in sort(unique(merged$Limit))) {
      mergedSub <- merged[merged$LR == rate & merged$Limit == limit,]
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "TP"
      averageList[[index]] <- mean(mergedSub$TP)
      maxList[[index]] <- max(mergedSub$TP)
      minList[[index]] <- min(mergedSub$TP)
      
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "FP"
      averageList[[index]] <- mean(mergedSub$FP)
      maxList[[index]] <- max(mergedSub$FP)
      minList[[index]] <- min(mergedSub$FP)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "FN"
      averageList[[index]] <- mean(mergedSub$FN)
      maxList[[index]] <- max(mergedSub$FN)
      minList[[index]] <- min(mergedSub$FN)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "TN"
      averageList[[index]] <- mean(mergedSub$TN)
      maxList[[index]] <- max(mergedSub$TN)
      minList[[index]] <- min(mergedSub$TN)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "Accuracy"
      averageList[[index]] <- mean(mergedSub$Acc)
      maxList[[index]] <- max(mergedSub$Acc)
      minList[[index]] <- min(mergedSub$Acc)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "Recall_P"
      averageList[[index]] <- mean(mergedSub$Recall_P)
      maxList[[index]] <- max(mergedSub$Recall_P)
      minList[[index]] <- min(mergedSub$Recall_P)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "Precision_P"
      averageList[[index]] <- mean(mergedSub$Precision_P)
      maxList[[index]] <- max(mergedSub$Precision_P)
      minList[[index]] <- min(mergedSub$Precision_P)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "Recall_N"
      averageList[[index]] <- mean(mergedSub$Recall_N)
      maxList[[index]] <- max(mergedSub$Recall_N)
      minList[[index]] <- min(mergedSub$Recall_N)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "Precision_N"
      averageList[[index]] <- mean(mergedSub$Precision_N)
      maxList[[index]] <- max(mergedSub$Precision_N)
      minList[[index]] <- min(mergedSub$Precision_N)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "BalancedAccuracy"
      averageList[[index]] <- mean(mergedSub$BalancedAcc)
      maxList[[index]] <- max(mergedSub$BalancedAcc)
      minList[[index]] <- min(mergedSub$BalancedAcc)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "F1"
      averageList[[index]] <- mean(mergedSub$F1)
      maxList[[index]] <- max(mergedSub$F1)
      minList[[index]] <- min(mergedSub$F1)
      index <- index + 1
      
      rateList[[index]] <- rate
      limitList[[index]] <- limit
      ratioList[[index]] <- mean(mergedSub$Ratio)
      measuresList[[index]] <- "MCC"
      averageList[[index]] <- mean(mergedSub$MCC)
      maxList[[index]] <- max(mergedSub$MCC)
      minList[[index]] <- min(mergedSub$MCC)
      index <- index + 1
    }
  }
  
  # create the data frame
  logs <- data.frame(unlist(rateList), unlist(ratioList), unlist(limitList), unlist(measuresList), unlist(averageList), unlist(maxList), unlist(minList))
  names(logs) <- c("LR", "Ratio", "Limit", "Measure", "Average", "Max", "Min")
  
  return(logs)
}

# this function creates the box plots used to evaluate the hyperparameter combinations (in Appendix C)
makeLRBarPlots <- function(logs1, logs5, logs9, measures, zoom) {
  # combine all of the results in one DataFrame
  logsList <- list()
  logsList[[1]] <- logs1[logs1$Measure %in% measures & logs1$Limit == 0.5,]
  logsList[[2]] <- logs5[logs5$Measure %in% measures & logs5$Limit == 0.167,]
  logsList[[3]] <- logs9[logs9$Measure %in% measures & logs9$Limit == 0.1,]
  subdata <- rbindlist(logsList)
  
  # prepare formatting for display
  subdata$Ratio <- as.factor(subdata$Ratio)
  subdata$Measure <- as.factor(subdata$Measure)
  subdata$LR <- as.factor(as.numeric(as.character(subdata$LR)))
  subdata$L <- as.factor(subdata$Limit)
  
  # determine whether we are zooming in on the results or not
  lower_y <- ifelse(zoom, 0.9, 0.0)
  y_gap <- ifelse(zoom, 0.02, 0.25)
  
  print(ggplot(data=subdata, aes(x = LR, y = Average, fill=Ratio)) +
          geom_bar(stat="identity", position=position_dodge()) +
          coord_cartesian(ylim=c(lower_y, 1.0)) +
          scale_y_continuous(breaks = seq(lower_y, 1.0, y_gap)) +
          ggtitle("ZFNet Grid Search Hyperparameter Tuning") +
          xlab("Learning Rate") + 
          ylab("Balanced Accuracy") + 
          theme_bw() +
          scale_fill_grey(start = .8, end=0) +
          theme(legend.position="top", legend.margin=margin(t=0, unit="cm"), plot.title=element_text(hjust=0.5), 
                plot.subtitle=element_text(hjust=0.5), text=element_text(size=20), axis.text.x=element_text(size=11)) +
          guides(fill=guide_legend(nrow=2, title_position="top")))
}

### The script starts here ###
DATAFOLDER <- "/mnt/windmills/results/ZFNet/"  # TODO: change this to your results directory

# go to the correct folder
setwd(DATAFOLDER)

# get the undersampling ratio 1:1 results
UNDERSAMPLING_RATIO = 1
LOGFILE_PREFIX <- paste0("results_ZFNetTrainTest_", UNDERSAMPLING_RATIO, "Ratio")
logs_zfnet_1ratio <- parseLogs(LOGFILE_PREFIX, UNDERSAMPLING_RATIO)
logsLong_zfnet_1ratio <- parseLogsLong(LOGFILE_PREFIX, UNDERSAMPLING_RATIO)

# get the undersampling ratio 1:5 (windmill:nonwindmill) results
UNDERSAMPLING_RATIO = 5
LOGFILE_PREFIX <- paste0("results_ZFNetTrainTest_", UNDERSAMPLING_RATIO, "Ratio")
logs_zfnet_5ratio <- parseLogs(LOGFILE_PREFIX, UNDERSAMPLING_RATIO)
logsLong_zfnet_5ratio <- parseLogsLong(LOGFILE_PREFIX, UNDERSAMPLING_RATIO)

# get the undersampling ratio 1:9 (windmill:nonwindmill) results
UNDERSAMPLING_RATIO = 9
LOGFILE_PREFIX <- paste0("results_ZFNetTrainTest_", UNDERSAMPLING_RATIO, "Ratio")
logs_zfnet_9ratio <- parseLogs(LOGFILE_PREFIX, UNDERSAMPLING_RATIO)
logsLong_zfnet_9ratio <- parseLogsLong(LOGFILE_PREFIX, UNDERSAMPLING_RATIO)

# get the appropriate thresholds for each undersampling ratio
logs_zfnet_1ratio_thresholded = logs_zfnet_1ratio[logs_zfnet_1ratio$Limit == 0.5,]
logs_zfnet_5ratio_thresholded = logs_zfnet_5ratio[logs_zfnet_5ratio$Limit == 0.167,]
logs_zfnet_9ratio_thresholded = logs_zfnet_9ratio[logs_zfnet_9ratio$Limit == 0.1,]

# print the best Balanced Accuracy performance for each undersampling ratio
logs_zfnet_1ratio_thresholded[logs_zfnet_1ratio_thresholded$BalancedAcc == max(logs_zfnet_1ratio_thresholded$BalancedAcc),]
logs_zfnet_5ratio_thresholded[logs_zfnet_5ratio_thresholded$BalancedAcc == max(logs_zfnet_5ratio_thresholded$BalancedAcc),]
logs_zfnet_9ratio_thresholded[logs_zfnet_9ratio_thresholded$BalancedAcc == max(logs_zfnet_9ratio_thresholded$BalancedAcc),]

# create the box plots (from Appendix C)
makeLRBarPlots(logsLong_zfnet_1ratio, logsLong_zfnet_5ratio, logsLong_zfnet_9ratio, c("BalancedAccuracy"), FALSE)
makeLRBarPlots(logsLong_zfnet_1ratio, logsLong_zfnet_5ratio, logsLong_zfnet_9ratio, c("BalancedAccuracy"), TRUE)