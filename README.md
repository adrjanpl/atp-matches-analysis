# ATP Tennis Match Prediction: A Machine Learning Approach 

## Project Overview
This repository contains the code and methodology for predicting the outcomes of professional men's ATP tennis matches. The primary goal of this project was to evaluate the predictability of tennis matches across different tournament tiers (Grand Slams, Masters 1000, ATP 250/500, Davis Cup) using Machine Learning algorithms.

## Data Engineering and Preprocessing
To ensure the models learn meaningful patterns rather than biases, the dataset was transformed using a Player 1 vs Player 2 format. Before this change the data was set in Winner vs Loser schema, which could cause problems for prediction models.
A randomized boolean mask was used to swap the order of players in 50% of the matches. This resulted in a balanced binary target variable.
All data was sorted by date to respect the time-series nature of sports events.

## 🧠 Modeling Strategy
Two classification algorithms were used to capture both linear and non-linear relationships:

1. **Logistic Regression with L1:**
   * Handled through a `Scikit-Learn Pipeline` with `StandardScaler`.
   * L1 regularization acted as a built-in feature selector, shrinking the coefficients of correlated or noisy statistics to `0`.
2. **Random Forest Classifier:**
   * Used to capture complex, non-linear interactions between player statistics.

## Rigorous Validation & Evaluation (Zero Data Leakage)
A major focus of this project was preventing data Leakage. Without it the prediction models could have learned on future matches and predicting the outcome on past matches which is not logical and not correct to do in sports.

* **Train/Test Split:** The chronological dataset was split into an 80% training set and a 20% test set.
* **Nested Cross-Validation:** Hyperparameter tuning (`GridSearchCV`) was performed exclusively on the training set using `TimeSeriesSplit` (5 splits). This ensured that the models were validated on expanding chronological windows, never looking into the future.

## 📊 Statistical Testing & Results Reliability
To prove that the results were stable and not a product of random chance, advanced statistical techniques were applied to the final predictions on the 20% Test Set:

### 1. Bootstrapping (95% Confidence Intervals)
Instead of relying on a single point-estimate for accuracy, a **Bootstrap method (1000 iterations with replacement)** was applied to the model's predictions. This generated 95% Confidence Intervals for both Accuracy and ROC AUC metrics, proving the stability of the models across varying test subsets.

### 2. Permutation Testing (Tournament Tier Comparison)
To determine if the predictability of matches statistically differs between tournament tiers (e.g., Are Grand Slams more predictable than ATP 250s?), a **Permutation Test** (1000 iterations) was conducted. 
* Predictions and true labels from pairs of tournament tiers were pooled and randomly shuffled.
* To account for the Multiple Comparisons Problem, raw p-values were adjusted using the **Holm correction** ($\alpha = 0.05$).

## 💡 Feature Importance
* **Logistic Regression:** Evaluated using the absolute values of standardized $\beta$ coefficients.
* **Random Forest:** Evaluated using **Permutation Feature Importance** (measuring the direct drop in ROC AUC when a feature is shuffled).

*(Optional: Add your Heatmap images here)*
> `![Feature Importance Heatmap](link_to_your_image.png)`

## 🛠️ Technologies & Libraries Used
* **Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (`LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, `TimeSeriesSplit`, `Pipeline`)
* **Statistical Testing:** `statsmodels` (Holm correction), Custom Permutation/Bootstrap scripts.
* **Visualization:** `matplotlib`, `seaborn`

## 👨‍💻 Author
[Twoje Imię i Nazwisko]
Connect with me on [LinkedIn](Twój_Link_Do_Profilu).
