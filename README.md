# Machine Learning models in predicting ATP tennis matches

## Project Overview
This repository contains the code and methodology for predicting the outcomes of professional men's ATP tennis matches. The primary goal of this project was to evaluate the predictability of tennis matches across different tournament tiers (Grand Slams, Masters 1000, ATP 250/500, Davis Cup) using Machine Learning algorithms.

## Data Engineering and Preprocessing
To ensure the models learn meaningful patterns rather than biases, the dataset was transformed using a Player 1 vs Player 2 format. Before this change the data was set in Winner vs Loser schema, which could cause problems for prediction models.
A randomized boolean mask was used to swap the order of players in 50% of the matches. This resulted in a balanced binary target variable.
All data was sorted by date to respect the time-series nature of sports events.

## Modeling Strategy
Two classification algorithms were used to capture both linear and non-linear relationships:

1. **Logistic Regression with L1:**
   * Handled through a `Scikit-Learn Pipeline` with `StandardScaler`.
   * L1 regularization acted as a built-in feature selector, shrinking the coefficients of correlated or noisy statistics to `0`.
2. **Random Forest Classifier:**
   * Used to capture complex, non-linear interactions between player statistics.

## Validation and evaluation
A major focus of this project was preventing data Leakage. Without it the prediction models could have learned on future matches and predicting the outcome on past matches which is not logical and not correct to do in sports.

* **Train/Test Split:** The chronological dataset was split into an 80% training set and a 20% test set.
* **Cross-Validation:** Hyperparameter tuning (`GridSearchCV`) was performed on the training set using `TimeSeriesSplit`. This ensured that the models were validated on expanding chronological windows, never looking into the future as mentioned before.

## Statistical Testing and Results Reliability
To prove that the results were stable and not a product of random chance, statistical techniques were applied to the final predictions on the 20% test set:

### 1. Bootstrapping 
Instead of relying on a single point-estimate for accuracy, a bootstrap method was applied to the model's predictions. This generated 95% Confidence Intervals for both Accuracy and ROC AUC metrics.

### 2. Permutation Testing
In the analysis also a permutation test was conducted to determine if the predictability of matches statistically differs between tournament tiers. 
Predictions and true labels from pairs of tournament tiers were pooled and randomly shuffled.
To account for the Multiple Comparisons Problem, raw p-values were adjusted using the **Holm correction** ($\alpha = 0.05$).

## Feature Importance
* **Logistic Regression:** Evaluated using the absolute values of standardized $\beta$ coefficients.
* **Random Forest:** Evaluated using permutation feature importance.


## Author
Adrian Janas

