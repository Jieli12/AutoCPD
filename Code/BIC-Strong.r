#
# Author         : Jie Li, Department of Statistics, London School of Economics.
# Date           : 2022-10-29 22:14:03
# Last Revision  : 2022-10-29 23:11:45
# Last Author    : Jie Li
# File Path      : /AutoCPD/Code/BIC-Strong.r
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
#
library(RcppCNPy)
library(changepoint)
library(not)
library(caret)
source("func.r", chdir = TRUE)
if (Sys.info()[1] == "Darwin") {
    home_path <- "~/Documents/"
} else if (Sys.info()[1] == "Windows") {
    home_path <- "~/GitHub/"
}
project_name <- "AutoCPD"
project_path <- paste0(home_path, project_name)
script_path <- paste0(project_path, "/Code")

setwd(script_path)

x_test <- npyLoad("../datasets/BIC/x_test_rstrong.npy")
y_test <- npyLoad("../datasets/BIC/y_test_rstrong.npy", type = "integer")

n <- length(y_test)
new_pred1 <- adjust_filtered(x_test[1:1000, ])
new_pred2 <- adjust_filtered(x_test[1001:n, ])
new_pred <- rbind(new_pred1, new_pred2)
y_pred_bic <- new_pred[, 10]
npySave("../datasets/BIC/y_pred_bic_rstrong.npy", y_pred_bic)
