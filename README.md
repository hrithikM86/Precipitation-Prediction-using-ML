# Precipitation Prediction using ML

## Project Overview
This project aims to predict precipitation using machine learning techniques. By leveraging various machine learning algorithms, we will build a model that can accurately classify whether precipitation will occur based on historical weather data.

## Table of Contents
- [Walkthrough](#walkthrough)
  - [Week 1: Setting up the Project](#week-1-setting-up-the-project)
  - [Week 2: Data Importing and Exploration](#week-2-data-importing-and-exploration)
  - [Week 3: Handling Class Imbalance and Missing Values](#week-3-handling-class-imbalance-and-missing-values)
  - [Standardizing Data and Feature Selection](#standardizing-data-and-feature-selection)
  - [Final Phase: Training Models](#final-phase-training-models)
- [Chi-Square Technique](#chi-square-technique)
- [ROC/AUC](#rocauc)
- [License](#license)

## Walkthrough

### Week 1: Setting up the Project
To get started, install Anaconda and Jupyter Notebook on your device. Follow these video tutorials for assistance:
- [Setting Up Anaconda and Jupyter](https://www.youtube.com/watch?v=uOwCiZKj2rg&feature=youtu.be)
- [Python Installation](https://www.youtube.com/watch?v=eWRfhZUzrAc)
- [Pandas Tutorial](https://www.youtube.com/watch?v=vmEHCJofslg)
- [Matplotlib Tutorial](https://www.youtube.com/watch?v=yZTBMMdPOww&feature=youtu.be)
- [Seaborn Tutorial](https://www.youtube.com/watch?v=6GUZXDef2U0)
- [Scikit-learn Tutorial](https://www.youtube.com/watch?v=pqNCD_5r0IU)

### Week 2: Data Importing and Exploration
- **Dataset**: [Download Dataset](https://drive.google.com/file/d/1xaspu6UgMI0mBZsOmkiVMIkBQP8V1Jvg/view)
- Use the Pandas framework to import the data and perform exploratory data analysis.
- In the dataset, the precipitation column will be the target feature. Replace values greater than 0 with 1 (indicating precipitation will occur) and values equal to 0 with 0 (indicating no precipitation).

### Week 3: Handling Class Imbalance and Missing Values
- Visualize the imbalance in the dataset using Matplotlib/Seaborn.
- Overbalance the minority class using `sklearn.utils.resample`.
- Check for null values; drop features with excessive nulls, and replace remaining nulls with the mode.

### Standardizing Data and Feature Selection
- Feature selection will be performed using the Chi-square test.
- Normalize the data using Pandas. More information can be found [here](https://www.geeksforgeeks.org/data-normalization-with-pandas/).

### Final Phase: Training Models
- Split the data into training and testing datasets.
- Use classifiers such as Logistic Regression, Decision Tree, and Neural Networks on the training dataset.
- Evaluate the models based on accuracy, precision, recall, F1 score, and ROC_AUC on the test dataset. Visualize results using confusion matrices and model comparison plots.

## Chi-Square Technique
1. The goal is to identify features that aren’t useful for prediction.
2. A higher Chi-Square value indicates a feature is more dependent on the target variable.
3. A higher p-value suggests a feature is independent of the target variable and not suitable for model training.
4. Implementation example:
   ```python
   from sklearn.feature_selection import chi2
   chi_scores = chi2(X, y)
   p_values = pd.Series(chi_scores[1], index=X.columns)
   p_values.sort_values(ascending=False, inplace=True)
   p_values.plot.bar()
