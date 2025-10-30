- Built fraud detection models on imbalanced Kaggle dataset (555K transactions, 99.6% non-fraud) by resampling to 78.6% non-fraud for training, engineering features including time-of-day indicators, distance calculations, target-encoded merchants/categories, and removing highly correlated predictors (lat/long) identified through correlation heatmap analysis.

- Compared four classification models on 80/20 train-test split: Classification Tree achieved best performance (93% recall, 90% precision, 96% accuracy) using entropy criterion and max depth of 10, with transaction amount dominating feature importance (72%) followed by gas (6%) and groceries (2%) categories, outperforming Logistic Regression, Naïve Bayes, and KNN.

- Tech Stack: pandas, matplotlib, scikit-learn
