# Machine Learning models in predicting ATP tennis matches

## Project Overview
This repository contains the code and methodology for predicting the outcomes of professional men's ATP tennis matches. The primary goal of this project was to evaluate the predictability of tennis matches across different tournament tiers such as Grand Slams marked as G, Masters 1000 - M, ATP 250/500 - A, Davis Cup - D, Olimpics - O and Final ATP - F using Machine Learning algorithms and compare it with a baseline which is simply picking the higher in ranking player in a specific match.

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

# Results and step by step analysis

1) Initially, the datasets were merged and checked for any missing values in the variables. When match statistics had missing values, the observations were removed, and for others like age or height, the median was imputed.
2) A boolean mask was applied to randomly divide the players into Player 1 and Player 2.
3) A rolling window was created to iterate through the grouped players and extract average match statistics for the 10 previous matches.
4) Variables representing the differences between players and the players' handedness were added.
5) A graph was created showing the number of matches for specific tournament tiers.
  * <img width="572" height="463" alt="obrazek1" src="https://github.com/user-attachments/assets/a208b798-62a9-4f35-8a8b-e533d80b6696" />
6) A graph was created illustrating the normality of the ranking difference variable.
  * <img width="2589" height="1590" alt="hist_rank_diff" src="https://github.com/user-attachments/assets/7eaa39b1-17fd-4ae3-9ccb-55ad68922679" />
7) The baseline was calculated for the tournament tiers.
  * <img width="450" height="182" alt="Zrzut ekranu 2026-07-29 130514" src="https://github.com/user-attachments/assets/46fa4608-3403-4d62-bbc1-2aa1f4ce60ec" />
8) Variables were selected for model training: Player 1 and Player 2 rankings, difference in: ranking, ranking points, age, height, left-handedness, Player 1 and Player 2 handedness, as well as average match statistics, and `GridSearchCV` was used to select the best parameters for the models.
9) After applying the bootstrap method, accuracy statistics for the models were calculated.
  * <img width="705" height="202" alt="2" src="https://github.com/user-attachments/assets/18941374-febe-455b-ae93-7cda552ae856" />
10) A decision was made to remove tiers F and O from the analysis due to a small number of observations.
11) The significance of differences between the tiers was compared using the permutation test method, and the Holm correction was applied to prevent the multiple comparisons problem.
  * <img width="717" height="160" alt="3" src="https://github.com/user-attachments/assets/6ec3d7b4-4c73-4c6b-8c4e-cdeb212b0c4e" />
12) Feature importance was checked for both models - Orange for Logistic Regression and blue for Random Forests.
  * <img width="2789" height="1452" alt="333" src="https://github.com/user-attachments/assets/71af3eaf-952d-480b-bfa5-58a1facf0187" />
  * <img width="2765" height="1452" alt="123" src="https://github.com/user-attachments/assets/1c74b017-1a97-4557-8b4e-c97354dc0b14" />
13) Finally, the model results were compared with the baseline threshold.
  * <img width="600" height="202" alt="555" src="https://github.com/user-attachments/assets/c672e0fd-4dbd-415d-a3ba-6d0b2ad8d2b8" />
  * <img width="656" height="202" alt="444" src="https://github.com/user-attachments/assets/e11ca20e-3db4-4f0f-963b-d6d08fb2866d" />


## Data Source 
The historical tennis match data used in this project was sourced from the publicly available repository maintained by **Jeff Sackmann**.

The original is mentioned in homepage: [JeffSackmann](https://www.jeffsackmann.com/).
The dataset seems to be removed from Jeff Sackmann's github.

*Note: For the purpose of this analysis, the raw CSV files were downloaded, merged, and heavily preprocessed to create the final modeling dataset.*


## Author
Adrian Janas

