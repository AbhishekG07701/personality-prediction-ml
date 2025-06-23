# 🧠 Personality Prediction Using Machine Learning

## 🔍 Project Overview

This project explores the relationship between behavioral traits and personality types using machine learning. The goal was to predict whether a person is an **Introvert or Extrovert** based on social activity, time spent alone, and other personal habits.

## 📊 Dataset Information

**Source:** [Kaggle - Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data)

**Columns:**
- `time_spent_alone`: Hours spent alone daily (0–11)
- `stage_fear`: Whether the person has stage fear (Yes/No)
- `social_event_attendance`: Weekly attendance at social events (0–10)
- `going_outside`: Weekly outdoor frequency (0–7)
- `drained_after_socializing`: Whether they feel tired after socializing (Yes/No)
- `friends_circle_size`: Number of close friends (0–15)
- `post_frequency`: Frequency of posting on social media (0–10)
- `personality`: **Target column** – Introvert or Extrovert

## 🔧 Process Overview

1. **Data Cleaning** – Removed duplicates, handled missing values  
2. **EDA** – Visualized how personality influences behavior  
3. **Feature Engineering** – Created new features:
   - `total_social_activity`
   - `social_comfort_index`
   - `alone_to_social_ratio`
   - `social_fatigue_ratio`
   - `online_vs_offline_ratio`  
4. **Model Building** – Compared:
   - Logistic Regression
   - KNN
   - Random Forest
   - CatBoost, XGBoost, LGBM
   - AdaBoost (Final Model)
5. **Hyperparameter Optimization** – Used **Optuna** to fine-tune the top models  
6. **Final Pipeline** – Created using AdaBoost with engineered features and preprocessing

## 📌 Key Questions Explored

- Do introverts spend more time alone than extroverts?
- How does social fatigue differ between personality types?
- What role does social media play in defining personality?
- Are extroverts more active socially, both online and offline?
- Can we quantify social overload and comfort?

## 📈 Visualizations

- Countplots and boxplots for personality vs. traits
- Correlation heatmap of numerical features
- Pair plots across variables
- Annotated bar charts for categorical comparisons

## 📊 Model Performance (Final AdaBoost)

| Metric     | Score        |
|------------|--------------|
| Accuracy   | 93.33%       |
| Precision  | 92.44%       |
| Recall     | 92.98%       |
| F1-Score   | 92.71%       |
| ROC AUC    | 96.19%       |

## 🛡️ Preventing Overfitting

- Used **Stratified K-Fold Cross Validation**
- Applied **SelectFromModel** for feature selection
- Balanced categorical encodings with KNN and SimpleImputer
- Optimized hyperparameters using **Optuna**

## 🧰 Tools & Technologies Used

- Python, Pandas, Seaborn, Matplotlib  
- Scikit-learn, Optuna, CatBoost, LightGBM, XGBoost  
- Pipeline, Feature Engineering, Label Encoding, KNNImputer  
- Stratified K-Fold, Classification Metrics

## 📌 Project Structure
- 📂 **Data/:** Contains the dataset file
- 📂 **Artifacts/:** Contains the final trained model
- 📜 **personality_detection.ipynb:** Jupyter notebooks for analysis, visualization and predictive modeling
- 📜 **README.md:** Project details and documentation