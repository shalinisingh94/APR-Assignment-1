# APR-Assignment-1
# Diabetes Prediction Model 🩺

This repository contains a Jupyter Notebook (`Diabetes.ipynb`) that demonstrates a complete workflow for building and evaluating a machine learning model to predict diabetes. The project uses a dataset with various health-related metrics to predict the likelihood of a positive diabetes outcome.

---

## 💻 Project Overview

The notebook follows a standard data science pipeline:
1.  **Data Loading & Initial Exploration**: The dataset is loaded and its basic properties, such as shape, data types, and descriptive statistics, are examined.
2.  **Data Cleaning & Preprocessing**: Missing or zero values in key columns like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are imputed. The imputation strategy is to replace these zeros with the median value of the respective column, separated by `Outcome` (diabetes positive/negative).
3.  **Data Visualization**: Various plots, including countplots, histograms, distribution plots, and a pairplot, are used to visualize the data and understand the relationships between different features.
4.  **Feature Scaling**: The dataset is scaled using `MinMaxScaler` to normalize the values between 0 and 1, which is a common practice before training machine learning models.
5.  **Model Training & Evaluation**: The data is split into training and testing sets. Two classification models, **Logistic Regression** and **Gaussian Naive Bayes**, are trained on the data. Their performance is then evaluated using accuracy scores and a confusion matrix.

---

## 📂 Repository Contents

* `diabetes.csv`: The primary dataset used for this project.
* `Diabetes.ipynb`: A Jupyter Notebook containing all the code for the data analysis, preprocessing, visualization, and model building.
* `README.md`: This file, providing an overview of the project.

---

## 📊 Key Findings

* [cite_start]The dataset contains 768 rows and 9 columns[cite: 14, 19].
* [cite_start]The `Outcome` column, which indicates the presence (1) or absence (0) of diabetes, shows that approximately **34.9%** of the individuals have diabetes[cite: 63].
* [cite_start]Data imputation was necessary for columns like `Glucose`, `BloodPressure`, and `Insulin`, which had a significant number of zero values[cite: 282, 284, 286, 288, 290].
* After imputation and scaling, the models were trained and tested.
* [cite_start]The **Logistic Regression** model achieved an accuracy of approximately **74.68%**[cite: 377].
* [cite_start]The **Naive Bayes** model achieved an accuracy of approximately **72.73%**[cite: 378].

---

## 🤝 How to Run the Code

1.  Clone this repository: `git clone [repository URL]`
2.  Ensure you have the required libraries installed:
    * `pandas`
    * `numpy`
    * `matplotlib`
    * `seaborn`
    * `scikit-learn`
3.  Open the `Diabetes.ipynb` notebook in a Jupyter environment (like Jupyter Notebook, JupyterLab, or Google Colab).
4.  Run the cells sequentially to execute the entire analysis pipeline.
