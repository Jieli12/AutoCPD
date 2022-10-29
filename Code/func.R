#
# Author         : Jie Li, Department of Statistics, London School of Economics.
# Date           : 2022-09-22 21:54:36
# Last Revision  : 2022-10-29 23:08:34
# Last Author    : Jie Li
# File Path      : /AutoCPD/Code/func.R
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

compute_mse <- function(x, cp = 0, flag = "mean") {
    n <- length(x)
    if (flag == "mean") {
        if (cp == 0) {
            mu <- mean(x)
            mse <- mean((x - mu)^2)
        } else {
            x1 <- x[1:cp]
            x2 <- x[(cp + 1):n]
            s1 <- sum((x1 - mean(x1))^2)
            s2 <- sum((x2 - mean(x2))^2)
            mse <- (s1 + s2) / n
        }
    } else if (flag == "var") {
        y <- x^2
        if (cp == 0) {
            mu <- mean(y)
            mse <- sqrt(mean((y - mu)^2))
        } else {
            y1 <- y[1:cp]
            y2 <- y[(cp + 1):n]
            mse <- sqrt((sum((y1 - mean(y1))^2) + sum((y2 - mean(y2))^2)) / n)
        }
    } else if (flag == "slope") {
        if (cp == 0) {
            index <- 1:n
            lm.result <- lm(x ~ index)
            mse <- mean(lm.result$residuals^2)
        } else {
            x1 <- x[1:cp]
            x2 <- x[(cp + 1):n]
            index1 <- 1:cp
            index2 <- (cp + 1):n
            lm.result1 <- lm(x1 ~ index1)
            lm.result2 <- lm(x2 ~ index2)
            mse <- (sum(lm.result1$residuals^2) + sum(lm.result2$residuals^2)) / n
        }
    }

    return(mse)
}

adjust_filtered <- function(mat) {
    n <- nrow(mat)
    cp_mean_not <- rep(0, n)
    cp_var <- rep(0, n)
    cp_slope_not <- rep(0, n)
    cp_loc_mean <- rep(0, n)
    cp_loc_var <- rep(0, n)
    cp_loc_slope <- rep(0, n)
    bic_M <- rep(0, n)
    bic_V <- rep(0, n)
    bic_S <- rep(0, n)
    label <- rep(0, n)
    for (i in 1:n) {
        print(i)
        x <- mat[i, ]

        # for change in mean
        ans <- not(x, contrast = "pcwsConstMean")
        cp_temp <- features(ans)[4]
        if (cp_temp == "NA") {
            cp_mean_not[i] <- 0
        } else if (length(cp_temp) == 1) {
            cp_mean_not[i] <- 1
            cp_loc_mean[i] <- cp_temp$cpt[1]
        }

        # for change in var
        ans <- cpt.var(x, penalty = "SIC", pen.value = 0.01, method = "AMOC")
        cp_temp <- cpts(ans)
        if (length(cp_temp) == 1 & cp_temp[1] > 10 & cp_temp[1] < length(x) - 10) {
            cp_var[i] <- 2
            cp_loc_var[i] <- cp_temp
        }

        # for change in slope
        ans <- not(x, contrast = "pcwsLinContMean")
        cp_temp <- features(ans)[4]
        if (cp_temp == "NA") {
            cp_slope_not[i] <- 3
        } else if (length(cp_temp) == 1) {
            cp_slope_not[i] <- 4
            cp_loc_slope[i] <- cp_temp$cpt[1]
        }

        # compute bic
        bic_mean <- compute_bic(x, cp = cp_loc_mean[i], flag = "mean")
        bic_var <- compute_bic(x, cp = cp_loc_var[i], flag = "var")
        bic_slope <- compute_bic(x, cp = cp_loc_slope[i], flag = "slope")
        min_bic <- min(c(bic_mean, bic_slope, bic_var))
        if (bic_mean == min_bic) {
            label[i] <- cp_mean_not[i]
        } else if (bic_var == min_bic) {
            label[i] <- cp_var[i]
        } else if (bic_slope == min_bic) {
            label[i] <- cp_slope_not[i]
        }
        bic_M[i] <- bic_mean
        bic_V[i] <- bic_var
        bic_S[i] <- bic_slope
    }
    result <- cbind(cp_mean_not, cp_loc_mean, bic_M, cp_var, cp_loc_var, bic_V, cp_slope_not, cp_loc_slope, bic_S, label)
    return(result)
}

compute_bic <- function(x, cp = 0, flag = "mean") {
    n <- length(x)
    if (flag == "mean") {
        k <- 2
        if (cp == 0) {
            mu <- mean(x)
            sigma <- sd(x)
            bic <- -2 * sum(dnorm(x, mu, sigma, log = TRUE)) + log(n) * k
        } else {
            x1 <- x[1:cp]
            x2 <- x[(cp + 1):n]
            mu1 <- mean(x1)
            sigma <- sd(x)
            mu2 <- mean(x2)
            bic1 <- -2 * sum(dnorm(x1, mu1, sigma, log = TRUE))
            bic2 <- -2 * sum(dnorm(x2, mu2, sigma, log = TRUE))
            bic <- bic1 + bic2 + log(n) * (k + 1)
        }
    } else if (flag == "var") {
        k <- 2
        if (cp == 0) {
            mu <- mean(x)
            sigma <- sd(x)
            bic <- -2 * sum(dnorm(x, mu, sigma, log = TRUE)) + log(n) * k
        } else {
            x1 <- x[1:cp]
            x2 <- x[(cp + 1):n]
            mu <- mean(x)
            sigma1 <- sd(x1)
            sigma2 <- sd(x2)
            bic1 <- -2 * sum(dnorm(x1, mu, sigma1, log = TRUE))
            bic2 <- -2 * sum(dnorm(x2, mu, sigma2, log = TRUE))
            bic <- bic1 + bic2 + log(n) * (k + 1)
        }
    } else if (flag == "slope") {
        k <- 3
        if (cp == 0) {
            index <- 1:n
            lm.result <- lm(x ~ index)
            sigma <- sd(lm.result$residuals)
            alpha <- lm.result$coefficients[1]
            beta <- lm.result$coefficients[2]
            mu <- alpha + beta * index
            bic <- -2 * sum(dnorm(x, mu, sigma, log = TRUE)) + log(n) * k
        } else {
            x1 <- x[1:cp]
            x2 <- x[(cp + 1):n]
            index1 <- 1:cp
            index2 <- (cp + 1):n
            lm.result1 <- lm(x1 ~ index1)
            lm.result2 <- lm(x2 ~ index2)
            sigma <- sd(c(lm.result1$residuals, lm.result1$residuals))
            alpha1 <- lm.result1$coefficients[1]
            beta1 <- lm.result1$coefficients[2]
            mu1 <- alpha1 + beta1 * index1
            alpha2 <- lm.result2$coefficients[1]
            beta2 <- lm.result2$coefficients[2]
            mu2 <- alpha2 + beta2 * index2
            bic1 <- -2 * sum(dnorm(x1, mu1, sigma, log = TRUE))
            bic2 <- -2 * sum(dnorm(x2, mu2, sigma, log = TRUE))
            bic <- bic1 + bic2 + log(n) * (k + 2)
        }
    }

    return(bic)
}

