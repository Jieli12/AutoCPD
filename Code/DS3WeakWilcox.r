#
# Author         : Jie Li, Department of Statistics, London School of Economics.
# Date           : 2023-05-08 20:16:33
# Last Revision  : 2023-05-08 20:17:33
# Last Author    : Jie Li
# File Path      : /AI-assisstedChangePointDetection/R/DS3WeakWilcox.r
# Description    :
#
#
#
#
#
#
#
#
# Copyright (c) 2023 by Jie Li, j.li196@lse.ac.uk
# All Rights Reserved.


library(RcppCNPy)
library(robts)
library(foreach)
library(doParallel)
registerDoParallel(cores = 4)
if (Sys.info()[1] == "Darwin") {
    home_path <- "~/Documents/"
} else if (Sys.info()[1] == "Windows") {
    home_path <- "~/GitHub/"
}
project_name <- "AutoCPD"
project_path <- paste0(home_path, project_name)
script_path <- paste0(project_path, "/Code")
setwd(script_path)
nn <- c(300, 400, 500, 600)
rmse <- rep(0, 4)
for (i in 1:4) {
    n <- nn[i]
    data_name <- paste("../datasets/RWilcoxon/DS3Weak", "n", as.character(n), ".npy", sep = "")
    cpt_name <- paste("../datasets/RWilcoxon/DS3Weakcpt", "n", as.character(n), ".npy", sep = "")
    data <- npyLoad(data_name)
    cpt <- npyLoad(cpt_name, type = "integer")

    dist <- foreach(i = 1:3000, .combine = cbind) %dopar% {
        cpt_i <- changerob(data[i, ], property = "location", test = "Wilcoxon", borderN = 0, plot = FALSE)
        cpt_i$estimate - cpt[i]
    }
    rmse[i] <- sqrt(mean(dist^2))
}
npySave("../datasets/RWilcoxon/DS3WeakRMSE.npy", rmse)
