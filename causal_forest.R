library(grf)

data <- read.csv("ACIC 2019\\TestDatasets_lowD\\testdataset7.csv")

tau.forest <- causal_forest(data[-c(1, 2)], data$Y, data$A)

tau.hat.oob <- predict(tau.forest)
hist(tau.hat.oob$predictions)

average_treatment_effect(tau.forest, target.sample = "all")