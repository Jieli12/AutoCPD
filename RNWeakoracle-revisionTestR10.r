#
# Author         : Jie Li, Department of Statistics, London School of Economics.
# Date           : 2023-05-18 10:42:14
# Last Revision  : 2023-05-28 23:12:35
# Last Author    : Jie Li
# File Path      : /AI-assisstedChangePointDetection/R/RNWeakoracle-revisionTestR10.r
# Description    :
#
#
#  LR-test in R
# Table 1 of main text, double-checked.
#
#
#
#
# Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
# All Rights Reserved.
library(RcppCNPy)
library(changepoint)
library(not)
library(caret)
if (Sys.info()[1] == "Darwin") {
    home_path <- "~/Documents/"
} else if (Sys.info()[1] == "Windows") {
    home_path <- "~/GitHub/"
}
project_name <- "AI-assisstedChangePointDetection"
project_path <- paste0(home_path, project_name)
script_path <- paste0(project_path, "/R")
setwd(script_path)
source("func.r", chdir = TRUE)
data_x <- npyLoad("../datasets/BICRevision/RNWeak21R10data_x_test.npy")
data_y <- npyLoad("../datasets/BICRevision/RNWeak21R10data_y_test.npy", type = "integer")
n <- dim(data_x)[1]
pred_y <- rep(0, n)
# for change in variance
for (i in 1:n) {
    print(i)
    x <- data_x[i, ]
    y <- data_y[i]
    if (y == 0 || y == 1) {
        ans <- not(x, contrast = "pcwsConstMean")
        cp_temp <- features(ans)[4]
        if (cp_temp == "NA") {
            pred_y[i] <- 0
        } else if (length(cp_temp) == 1) {
            pred_y[i] <- 1
        }
    }
    # for change in var
    if (y == 2) {
        ans <- cpt.var(x, penalty = "SIC", pen.value = 0.01, method = "AMOC")
        cp_temp <- cpts(ans)
        if (length(cp_temp) == 1) {
            pred_y[i] <- 2
        }
    }
    if (y == 3 || y == 4) {
        ans <- not(x, contrast = "pcwsLinContMean")
        cp_temp <- features(ans)[4]
        if (cp_temp == "NA") {
            pred_y[i] <- 3
        } else if (length(cp_temp) == 1) {
            pred_y[i] <- 4
        }
    }
}



new_pred1 <- adjust_filtered(data_x)
pred_y_bic <- new_pred1[, 10]
c1 <- sum(data_y == 0 & pred_y_bic == 0) / sum(data_y == 0)
c2 <- sum(data_y == 1 & pred_y_bic == 1) / sum(data_y == 1)
c3 <- sum(data_y == 2 & pred_y_bic == 2) / sum(data_y == 2)
c4 <- sum(data_y == 3 & pred_y_bic == 3) / sum(data_y == 3)
c5 <- sum(data_y == 4 & pred_y_bic == 4) / sum(data_y == 4)
print("LR-adaptive")
print(c(c1, c2, c3, c4, c5))
print(sum(data_y == pred_y_bic) / n)

c1 <- sum(data_y == 0 & pred_y == 0) / sum(data_y == 0)
c2 <- sum(data_y == 1 & pred_y == 1) / sum(data_y == 1)
c3 <- sum(data_y == 2 & pred_y == 2) / sum(data_y == 2)
c4 <- sum(data_y == 3 & pred_y == 3) / sum(data_y == 3)
c5 <- sum(data_y == 4 & pred_y == 4) / sum(data_y == 4)
print("LR-oracle")
print(c(c1, c2, c3, c4, c5))
print(sum(data_y == pred_y) / n)
print("ResNet")
c(416 / sum(data_y == 0), 445 / sum(data_y == 1), 437 / sum(data_y == 2), 436 / sum(data_y == 3), 431 / sum(data_y == 0))
print((416 + 445 + 437 + 436 + 431) / n)
npySave("../datasets/BICRevision/RNWeak21R10Result.npy", cbind(data_y, pred_y, pred_y_bic))
