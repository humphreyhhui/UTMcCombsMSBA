- Predicted Austin housing prices by engineering 14 features (age, sqft_ratio, bath_bed_weighted, lot_category, interaction terms) with log transformations on price and square footage to reduce skewness, treating zipcode and amenities as categorical factors, and handling unseen holdout zipcodes by replacing with the most common training zipcode.

- Compared five models using 5-fold cross-validation on log-transformed prices: Bagging, Random Forest, XGBoost, BART, and Pruned Regression Tree. Selected Bagging as final model for its lowest RMSE and resistance to overfitting through ensemble averaging while maintaining low bias.

- Tech Stack: R (tidyverse, tidymodels, rpart, randomForest, BART, xgboost)
