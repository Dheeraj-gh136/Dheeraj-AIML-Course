# IRIS-Flower-classification

<img src="https://miro.medium.com/max/875/1*7bnLKsChXq94QjtAiRn40w.png">

## Project Overview

This project demonstrates a machine learning approach to classify iris flowers into three species: Iris-setosa, Iris-versicolor, and Iris-virginica, based on four features: sepal length, sepal width, petal length, and petal width. The goal is to build a model that can predict the species of an iris flower given its measurements.

## Dataset

The dataset used is the classic Iris dataset, which contains 150 samples of iris flowers, each with the following attributes:

- Sepal length (cm)  
- Sepal width (cm)  
- Petal length (cm)  
- Petal width (cm)  
- Species (target variable: setosa, versicolor, virginica)

The dataset is either loaded directly from `iris.csv` or from `sklearn.datasets.load_iris()`.

## Project Steps

1. **Data Loading and Exploration**  
   - Loaded the dataset into a Pandas DataFrame.  
   - Performed exploratory data analysis (EDA) including `.head()`, `.info()`, `.describe()`.  
   - Visualized the data with pairplots, histograms, and a correlation heatmap.

2. **Data Preprocessing**  
   - Checked for missing values and duplicates.  
   - Encoded the target variable if needed for machine learning models.

3. **Model Selection and Training**  
   - Split the dataset into training and testing sets.  
   - Trained multiple classifiers such as:
     - Logistic Regression  
     - Decision Tree  
     - K-Nearest Neighbors (KNN)  
     - Support Vector Machine (SVM)

4. **Model Evaluation**  
   - Evaluated models using:
     - Accuracy score  
     - Confusion matrix  
     - Classification report (precision, recall, F1-score)  
   - Selected the best-performing model based on evaluation metrics.

5. **Final Predictions**  
   - Demonstrated predictions on new sample inputs to classify iris species.

## Results

- The models achieved high accuracy (typically above 95%).  
- The confusion matrix and classification report indicate that the models can reliably distinguish between the three iris species.  
- Example prediction:
- Input: [5.1, 3.5, 1.4, 0.2]
Predicted species: setosa


## Technologies and Libraries

- Python 3.x  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn (sklearn)  

## How to Run the Project

1. Clone the repository:

git clone <repository-url>

2. Ensure required libraries are installed:

pip install -r requirements.txt 


3. Open the Jupyter Notebook (`iris_classification.ipynb`) or use Google Colab.

4. Run all cells to reproduce data exploration, model training, evaluation, and predictions.

## Notes

- The notebook is structured to first explore and visualize the dataset, then train and evaluate multiple models, and finally demonstrate predictions.  
- The target column should not be included when calculating correlations or feature analysis plots. Only numeric feature columns are used for such analyses.

Thank for viewing!

