
# Library -----------------------------------------------------------------
library(tidyverse)
library(rpart)
library(Metrics)
library(tree)
library(caTools)
library(dplyr)
library(readr)
library(randomForest)
library(BART)
library(tidymodels)
library(xgboost)


# Load Data / Feature Engineering -----------------------------------------
#rm(list = ls()) # environment reset button
austin_data <- austinhouses #read_csv("Documents/MSBA/Summer/Stats/Homeworks/Homework 1/austinhouses.csv")
austin_holdout <- austinhouses_holdout #read_csv("Documents/MSBA/Summer/Stats/Homeworks/Prediction Contest/austinhouses_holdout.csv")
austin_data <- austin_data %>%
  mutate(
    age = latest_saleyear - yearBuilt,
    bath_per_bed = ifelse(numOfBedrooms == 0, NA, numOfBathrooms / numOfBedrooms),
    lot_category = cut(lotSizeSqFt,
                       breaks = c(0, 5000, 10000, 20000, Inf),
                       labels = c("small", "medium", "large", "very_large"),
                       right = FALSE),
    log_sqft = log(livingAreaSqFt),
    log_lot = log(lotSizeSqFt + 1),
    logPrice = log(latestPrice),
    sqft_ratio = livingAreaSqFt / (lotSizeSqFt + 1),
    bath_bed_weighted = numOfBathrooms + 0.5 * numOfBedrooms,
    zipcode = as.factor(zipcode),
    homeType = as.factor(homeType),
    hasGarage = as.factor(hasGarage),
    hasSpa = as.factor(hasSpa),
    hasView = as.factor(hasView),
    hasAssociation = as.factor(hasAssociation),
    age_sqft_interaction = age * livingAreaSqFt,
    bath_sqft_ratio = numOfBathrooms / livingAreaSqFt,
  )
zip_counts <- austin_data %>%
  count(zipcode, name = "zip_listing_count")
austin_data <- austin_data %>%
  left_join(zip_counts, by = "zipcode")
austin_subset <- dplyr::select(austin_data, -streetAddress, -description, -homeType, -latest_saledate)
austin_subset <- na.omit(austin_subset)


# Cross Validation for each model
set.seed(200)
cv_folds <- vfold_cv(austin_subset, v = 5)


# Regression Tree ---------------------------------------------------------
cv_results_tree <- map_dfr(cv_folds$splits, function(split) {
  train_data <- analysis(split)
  test_data  <- assessment(split)
  tree_model <- rpart(logPrice ~ latitude + longitude + hasAssociation +
                        livingAreaSqFt + numOfBathrooms + numOfBedrooms + age +
                        lot_category + log_sqft + bath_bed_weighted + zipcode + log_lot +
                        age_sqft_interaction + bath_sqft_ratio, data = train_data)
  pred_log <- predict(tree_model, newdata = test_data)
  pred_price <- exp(pred_log)
  test_mse_tree <- mean((test_data$latestPrice - pred_price)^2)
  test_rmse_tree <- sqrt(test_mse_tree)
  best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
  pruned <- rpart::prune(tree_model, cp = best_cp)
  pred_log_pruned <- predict(pruned, newdata = test_data)
  pred_price_pruned <- exp(pred_log_pruned)
  test_mse_tree_pruned <- mean((test_data$latestPrice - pred_price_pruned)^2)
  test_rmse_tree_pruned <- sqrt(test_mse_tree_pruned)
  tibble(mse_tree = test_mse_tree, rmse_tree = test_rmse_tree,
         mse_tree_pruned = test_mse_tree_pruned, rmse_tree_pruned = test_rmse_tree_pruned)
})
cat("Regression Tree CV MSE (avg):", round(mean(cv_results_tree$mse_tree), 2), "\n")
cat("Regression Tree CV RMSE (avg):", round(mean(cv_results_tree$rmse_tree), 2), "\n")
cat("Regression Tree Pruned CV MSE (avg):", round(mean(cv_results_tree$mse_tree_pruned), 2), "\n")
cat("Regression Tree Pruned CV RMSE (avg):", round(mean(cv_results_tree$rmse_tree_pruned), 2), "\n")


# Bagging Model -----------------------------------------------------------
cv_results_bagging <- map_dfr(cv_folds$splits, function(split) {
  train_data <- analysis(split)
  test_data  <- assessment(split)
  bagging_model <- randomForest(
    logPrice ~ latitude + longitude + hasAssociation +
      livingAreaSqFt + numOfBathrooms + numOfBedrooms + age +
      lot_category + log_sqft + bath_bed_weighted + zipcode + log_lot +
      age_sqft_interaction + bath_sqft_ratio,
    data = train_data,
    mtry = 14,
    importance = TRUE
  )
  log_preds_bagging <- predict(bagging_model, newdata = test_data)
  price_preds_bagging <- exp(log_preds_bagging)
  actual_prices <- test_data$latestPrice
  mse_bagging <- mean((actual_prices - price_preds_bagging)^2)
  rmse_bagging <- sqrt(mse_bagging)
  tibble(mse = mse_bagging, rmse = rmse_bagging)
})

cat("Bagging CV MSE (avg):", round(mean(cv_results_bagging$mse), 2), "\n")
cat("Bagging CV RMSE (avg):", round(mean(cv_results_bagging$rmse), 2), "\n")

# Bagging has best RMSE. Pull out model.
bagging_model <- randomForest(
  logPrice ~ latitude + longitude + hasAssociation +
    livingAreaSqFt + numOfBathrooms + numOfBedrooms + age +
    lot_category + log_sqft + bath_bed_weighted + zipcode + log_lot +
    age_sqft_interaction + bath_sqft_ratio,
  data = austin_subset,  # or your full training data without splitting
  mtry = 14,
  importance = TRUE
)




# Random Forest -----------------------------------------------------------
cv_results_rf <- map_dfr(cv_folds$splits, function(split) {
  train_data <- analysis(split)
  test_data  <- assessment(split)
  rf_model <- randomForest(
    logPrice ~ latitude + longitude + hasAssociation +
      livingAreaSqFt + numOfBathrooms + numOfBedrooms+age+
      lot_category+log_sqft+bath_bed_weighted+zipcode+log_lot+
      age_sqft_interaction+bath_sqft_ratio,
    data = train_data,
    importance = TRUE
  )
  log_preds_rf <- predict(rf_model, newdata = test_data)
  price_preds_rf <- exp(log_preds_rf)
  actual_prices <- test_data$latestPrice
  mse_rf <- mean((actual_prices - price_preds_rf)^2)
  rmse_rf <- sqrt(mse_rf)
  tibble(mse = mse_rf, rmse = rmse_rf)
})
cat("Random Forest CV MSE (avg):", round(mean(cv_results_rf$mse), 2), "\n")
cat("Random Forest CV RMSE (avg):", round(mean(cv_results_rf$rmse), 2), "\n")


# BART --------------------------------------------------------------------
set.seed(200)
cv_results_bart <- map_dfr(cv_folds$splits, function(split) {
  train_data <- analysis(split)
  test_data  <- assessment(split)
  x_train_bart <- train_data %>% select(latitude, longitude, hasAssociation,
                                        livingAreaSqFt, numOfBathrooms, numOfBedrooms, age,
                                        lot_category, log_sqft, bath_bed_weighted, zipcode, log_lot,
                                        age_sqft_interaction, bath_sqft_ratio)
  x_test_bart  <- test_data %>% select(latitude, longitude, hasAssociation,
                                       livingAreaSqFt, numOfBathrooms, numOfBedrooms, age,
                                       lot_category, log_sqft, bath_bed_weighted, zipcode, log_lot,
                                       age_sqft_interaction, bath_sqft_ratio)
  x_train_bart <- data.matrix(x_train_bart)
  x_test_bart  <- data.matrix(x_test_bart)
  y_train_bart <- train_data$logPrice
  bart_model <- gbart(
    x.train = x_train_bart,
    y.train = y_train_bart,
    x.test = x_test_bart
  )
  log_preds_bart <- bart_model$yhat.test.mean
  price_preds_bart <- exp(log_preds_bart)
  actual_prices <- test_data$latestPrice
  mse_bart <- mean((price_preds_bart - actual_prices)^2)
  rmse_bart <- sqrt(mse_bart)
  tibble(mse = mse_bart, rmse = rmse_bart)
})
cat("BART CV MSE (avg):", round(mean(cv_results_bart$mse), 2), "\n")
cat("BART CV RMSE (avg):", round(mean(cv_results_bart$rmse), 2), "\n")


# XgBoost -----------------------------------------------------------------
set.seed(200)
cv_results_xgb <- map_dfr(cv_folds$splits, function(split) {
  train_data <- analysis(split)
  test_data  <- assessment(split)
  full_fold <- rbind(train_data, test_data)
  x_full <- model.matrix(logPrice ~ latitude + longitude + hasAssociation +
                           livingAreaSqFt + numOfBathrooms + numOfBedrooms+age+
                           lot_category+log_sqft+bath_bed_weighted+zipcode+log_lot+
                           age_sqft_interaction+bath_sqft_ratio, data = full_fold)[, -1]
  n_train <- nrow(train_data)
  x_train <- x_full[1:n_train, ]
  x_test  <- x_full[(n_train + 1):nrow(x_full), ]
  y_train <- train_data$logPrice
  xgb_model <- xgboost(
    data = x_train,
    label = y_train,
    objective = "reg:squarederror",
    nrounds = 300,
    eta = 0.1,
    max_depth = 6,
    verbose = 0
  )
  log_preds_xgb <- predict(xgb_model, newdata = x_test)
  price_preds_xgb <- exp(log_preds_xgb)
  actual_prices <- test_data$latestPrice
  rmse_xgb <- sqrt(mean((actual_prices - price_preds_xgb)^2))
  mse_xgb <- mean((actual_prices - price_preds_xgb)^2)
  tibble(mse = mse_xgb, rmse = rmse_xgb)
})
cat("XGBoost CV MSE (avg):", round(mean(cv_results_xgb$mse), 2), "\n")
cat("XGBoost CV RMSE (avg):", round(mean(cv_results_xgb$rmse), 2), "\n")


# Preprocess austin_holdout for prediction --------------------------------
austin_holdout <- austin_holdout %>%
  mutate(
    age = latest_saleyear - yearBuilt,
    bath_per_bed = ifelse(numOfBedrooms == 0, NA, numOfBathrooms / numOfBedrooms),
    lot_category = cut(lotSizeSqFt,
                       breaks = c(0, 5000, 10000, 20000, Inf),
                       labels = c("small", "medium", "large", "very_large"),
                       right = FALSE),
    log_sqft = log(livingAreaSqFt),
    log_lot = log(lotSizeSqFt + 1),
    sqft_ratio = livingAreaSqFt / (lotSizeSqFt + 1),
    bath_bed_weighted = numOfBathrooms + 0.5 * numOfBedrooms,
    zipcode = as.factor(zipcode),
    homeType = as.factor(homeType),
    hasGarage = as.factor(hasGarage),
    hasSpa = as.factor(hasSpa),
    hasView = as.factor(hasView),
    hasAssociation = as.factor(hasAssociation),
    age_sqft_interaction = age * livingAreaSqFt,
    bath_sqft_ratio = numOfBathrooms / livingAreaSqFt,
  )

zip_counts <- austin_holdout %>%
  count(zipcode, name = "zip_listing_count")
austin_holdout <- austin_holdout %>%
  left_join(zip_counts, by = "zipcode")
austin_holdout_subset <- dplyr::select(austin_holdout, -streetAddress, -description, -homeType)
austin_holdout_subset <- select(austin_holdout_subset, latitude, longitude, hasAssociation, livingAreaSqFt, numOfBathrooms, numOfBedrooms, age, lot_category, log_sqft, bath_bed_weighted, zipcode, log_lot, age_sqft_interaction, bath_sqft_ratio)

# for factors not seen in training data, replace with most common
factor_vars <- c("zipcode", "hasAssociation", "lot_category")
for (var in factor_vars) {
  train_levels <- levels(austin_subset[[var]])
  austin_holdout_subset[[var]] <- factor(austin_holdout_subset[[var]], levels = train_levels)
  most_common_level <- names(sort(table(austin_subset[[var]]), decreasing = TRUE))[1]
  austin_holdout_subset[[var]][is.na(austin_holdout_subset[[var]])] <- most_common_level
}
log_preds_holdout <- predict(bagging_model, newdata = austin_holdout_subset)
price_preds_holdout <- exp(log_preds_holdout)
output <- data.frame(predicted_price = price_preds_holdout)
write.csv(output, "Documents/MSBA/Summer/Stats/Homeworks/Prediction Contest/predicted_austinhouses_holdout.csv", row.names = FALSE)

