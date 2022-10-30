#
# Author         : Jie Li, Department of Statistics, London School of Economics.
# Date           : 2022-10-30 15:15:44
# Last Revision  : 2022-10-30 15:19:51
# Last Author    : Jie Li
# File Path      : /AutoCPD/Code/BIC-Weak-rep30.r
# Description    :
#
#
#
#
#
#
#
#
# Copyright (c) 2022 by Jie Li, j.li196@lse.ac.uk
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
project_name <- "AutoCPD"
project_path <- paste0(home_path, project_name)
script_path <- paste0(project_path, "/Code")
setwd(script_path)
source("func.r", chdir = TRUE)
n <- 1000
r <- 30
x_test <- npyLoad("../datasets/BIC/x_test_rweak_rep30.npy")
y_test <- npyLoad("../datasets/BIC/y_test_rweak_rep30.npy", type = "integer")
acc <- rep(0, r)
for (i in 1:r) {
    print(i)
    ind_temp <- ((i - 1) * n + 1):(i * n)
    new_pred <- adjust_filtered(x_test[ind_temp, ])
    y_pred_bic <- new_pred[, 10]
    y_test_true <- y_test[ind_temp]
    result <- confusionMatrix(as.factor(y_pred_bic), as.factor(y_test_true))
    acc[i] <- result$overall[1]
}
npySave("../datasets/BIC/y_pred_bic_rweak_rep30.npy", acc)
