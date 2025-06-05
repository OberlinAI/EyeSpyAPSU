# create a data set with the results (one observation per seed * model combination)
# NOTE: the other datasets are created by the processUNetTestResults.R script (and similar for VGG and ZFNet)
windmill_results <- rbind(logs_zfnet_test[logs_zfnet_test$Limit == 0.1,], logs_vgg_test[logs_vgg_test$Limit == 0.1,])
windmill_results <- rbind(windmill_results, logs_unet[logs_unet$Limit == 400,])
windmill_results$Method <- "ZFNet"
windmill_results$Method[31:60] <- "VGG-16"
windmill_results$Method[61:90] <- "U-Net"
windmill_results <- windmill_results[,c(18,1:17)]

# make sure the TP and FN values are normally distributed so we can use the t-test
shapiro.test(windmill_results$TP[windmill_results$Method == "ZFNet"])
shapiro.test(windmill_results$TP[windmill_results$Method == "VGG-16"])
shapiro.test(windmill_results$TP[windmill_results$Method == "U-Net"])

shapiro.test(windmill_results$FN[windmill_results$Method == "ZFNet"])
shapiro.test(windmill_results$FN[windmill_results$Method == "VGG-16"])
shapiro.test(windmill_results$FN[windmill_results$Method == "U-Net"])

shapiro.test(windmill_results$FP[windmill_results$Method == "ZFNet"])
shapiro.test(windmill_results$FP[windmill_results$Method == "VGG-16"])
shapiro.test(windmill_results$FP[windmill_results$Method == "U-Net"])

# perform pairwise t-tests between U-Net and ZFNet
t.test(windmill_results$TP[windmill_results$Method == "U-Net"], windmill_results$TP[windmill_results$Method == "ZFNet"])
t.test(windmill_results$FN[windmill_results$Method == "U-Net"], windmill_results$FN[windmill_results$Method == "ZFNet"])
wilcox.test(windmill_results$FP[windmill_results$Method == "U-Net"], windmill_results$FP[windmill_results$Method == "ZFNet"])

# perform pairwise t-tests between U-Net and VGG-16
t.test(windmill_results$TP[windmill_results$Method == "U-Net"], windmill_results$TP[windmill_results$Method == "VGG-16"])
t.test(windmill_results$FN[windmill_results$Method == "U-Net"], windmill_results$FN[windmill_results$Method == "VGG-16"])
wilcox.test(windmill_results$FP[windmill_results$Method == "U-Net"], windmill_results$FP[windmill_results$Method == "VGG-16"])
