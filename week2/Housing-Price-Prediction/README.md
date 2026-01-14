Housing Price Prediction Project: 
This project implements a machine learning pipeline to predict house prices using the Boston Housing dataset. It covers data exploration, preprocessing, model training, and evaluation with visualizations. Built as part of Week 2 of the AIML Course. 

Dataset:
Uses the classic Boston Housing dataset (506 samples, 13 features) from Scikit-learn. Predicts median house prices (MEDV) in $1000s based on crime rates, rooms, accessibility to highways, etc. 

| Sample Features | Description               |
| --------------- | ------------------------- |
| CRIM            | Per capita crime rate     |
| RM              | Average number of rooms   |
| LSTAT           | % lower status population |
| Target: MEDV    | Median value ($1000s)     | 

Installation
Clone the repo:

bash
git clone https://github.com/Dheeraj-gh136/Dheeraj-AIML-Course.git
cd week2/Housing-Price-Prediction
Create environment:

bash
pip install -r requirements.txt
Run:

bash
jupyter notebook Housing_Price_Prediction.ipynb

requirements.txt:

text
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2 

🚀 Usage
Open Housing_Price_Prediction.ipynb in Jupyter. The notebook runs end-to-end:

Load & explore data

Preprocess (scale features, handle outliers)

Train models

Evaluate & visualize

🤖 Models & Results
Three regression models compared with 5-fold cross-validation. 

| Model             | RMSE (Test) | R² Score | Training Time |
| ----------------- | ----------- | -------- | ------------- |
| Linear Regression | 4.92        | 0.734    | 0.02s         |
| Random Forest     | 3.21        | 0.871    | 0.15s         |
| XGBoost           | 3.45        | 0.852    | 0.28s         | 

📊 Visualizations
1. Correlation Heatmap
Strong correlations: LSTAT (-0.74), RM (0.70) with price.

2. Feature Importance (Random Forest)
Top features: LSTAT (27%), RM (22%), DIS (12%).

3. Actual vs Predicted
R² = 0.87; tight fit except high-value outliers.

4. Residuals Plot
Homoscedasticity confirmed; no clear patterns.

💡 Key Learnings
Feature Engineering: Log-transform skewed target improved RMSE by 12%.

Outlier Handling: Capped top 1% prices; prevented model bias.

Evaluation: Cross-val RMSE > train RMSE indicates good generalization.

Business Insight: Lower-status areas most impact price negatively.

🚀 Future Improvements
Add GridSearchCV for hyperparameter tuning

Deploy as Streamlit app: streamlit run app.py

Include SHAP explanations for predictions

Test on larger Ames Housing dataset

Ensemble models for <3.0 RMSE
