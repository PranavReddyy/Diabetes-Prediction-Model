# Diabetes Prediction Model

## Overview

This project focuses on developing a machine learning model to predict diabetes based on diagnostic measurements. It uses a dataset of 768 female patients, with 268 having diabetes and 500 not having diabetes. Exploratory data analysis (EDA) is performed to understand the relationships between different features and the outcome variable. The data is then standardized using `MinMaxScaler` before being used to train a predictive model. Several classification algorithms are explored, and their performance is evaluated to select the best model. The following models are tested: Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Decision Tree Classifier, and Random Forest Classifier.

## Data Analysis

### 1. Data Loading

The project begins by loading the diabetes dataset from a CSV file into a pandas DataFrame.


The dataset includes the following columns:

-   `Pregnancies`: Number of pregnancies
-   `Glucose`: Glucose level
-   `BloodPressure`: Blood pressure level
-   `SkinThickness`: Skin thickness
-   `Insulin`: Insulin level
-   `BMI`: Body mass index
-   `DiabetesPedigreeFunction`: Diabetes pedigree function
-   `Age`: Age
-   `Outcome`: Whether the patient has diabetes (1) or not (0)

The DataFrame `diabetes_data` initially contains 768 rows and 9 columns, displaying the diagnostic measurements and diabetes outcome for each patient.

### 2. Data Description

The project provides a statistical description of the dataset using the `describe()` method.


This provides the count, mean, standard deviation, min, max, and quartile values for each feature.

### 3. Correlation Analysis

The project analyzes the correlation between different features using a heatmap.


Observations:

*   Insulin is highly correlated with Glucose, BMI, and Age. As the values of glucose, BMI, and Age increase, the insulin is also increasing.
*   SkinThickness is highly correlated with BMI.

### 4. Checking Data Balance

The project checks if the data is balanced or imbalanced using a countplot.


Observations:

*   A total of 768 women were registered in the database. 268 women had diabetes, while 500 women did not have diabetes.
*   The dataset is biased towards non-diabetic people. The number of non-diabetic people is almost twice the number of diabetic patients.

### 5. Scatter Matrix

A pair-plot is used to visualize the relationships between different features and the outcome.


Observations:

*   The median BMI does not significantly change as the number of pregnancies increases.
*   Those who tested positive for diabetes had higher BMIs than those who did not.

### 6. Pedigree Function vs Diabetes

A boxplot is used to visualize the relationship between the diabetes pedigree function and the outcome.


Observations:

*   Those who tested positive have a higher median and more high outliers, showing that the pedigree function accurately helps estimate the test results for diabetes.
*   The genetic component is likely to contribute more to the emergence of diabetes.

### 7. Pregnancy vs Diabetes

A boxplot is used to visualize the relationship between the number of pregnancies and the outcome.


Observations:

*   The average number of pregnancies is higher in diabetic as compared to non-diabetic women.

### 8. Prevalence of Diabetes vs BMI

The prevalence of diabetes and its relation to BMI is analyzed.


Observations:

*   BMI shows a significant association with the occurrence of diabetes.
*   Women who tested positive have higher BMIs.

### 9. Age vs Diabetes

A boxplot is used to visualize the relationship between age and the outcome.


Observations:

*   A significant relation can be seen between the age distribution and occurrence of diabetes. Women at age group > 31 years were at higher risk of getting diabetes in comparison to the younger age group.

### 10. Data Standardization (MinMaxScaler)

The data is standardized using `MinMaxScaler` to scale features to a range between 0 and 1.


The scaled data is then printed to show the standardized values for each feature.

## Model Training and Evaluation

### 1. Data Splitting

The standardized data is split into training and testing sets. The data is split into 80% training and 20% testing sets with a `random_state=42` for reproducibility.


### 2. Model Selection and Training

Several classification models are trained on the training data, including:

*   Logistic Regression
*   K-Nearest Neighbors (KNN)
*   Support Vector Classifier (SVC)
*   Decision Tree Classifier
*   Random Forest Classifier


### 3. Model Evaluation

The trained models are evaluated on the test set, and their performance is assessed using metrics such as accuracy and classification report.


### 4. Results and Comparison

Based on the accuracy scores and classification reports, the models can be compared. The classification reports provide insights into precision, recall, and F1-score for each class.

The following results were obtained after applying the models:

*   **Logistic Regression:**
    *   Accuracy: 0.7792
    *   Precision: 0.73
    *   Recall: 0.60
    *   F1-score: 0.66
*   **K-Nearest Neighbors:**
    *   Accuracy: 0.7403
    *   Precision: 0.64
    *   Recall: 0.64
    *   F1-score: 0.64
*   **Support Vector Machine:**
    *   Accuracy: 0.7597
    *   Precision: 0.70
    *   Recall: 0.58
    *   F1-score: 0.63
*   **Decision Tree:**
    *   Accuracy: 0.7208
    *   Precision: 0.60
    *   Recall: 0.64
    *   F1-score: 0.62
*   **Random Forest:**
    *   Accuracy: 0.7532
    *   Precision: 0.65
    *   Recall: 0.67
    *   F1-score: 0.66

## Conclusion

This project provides a comprehensive exploratory data analysis, preprocessing, and model building pipeline for a diabetes prediction model. By understanding the relationships between features, standardizing the data, and training and evaluating multiple models, this analysis helps in building an effective predictive model. Further improvements can be made by:

*   Addressing data imbalance through techniques like oversampling or undersampling.
*   Fine-tuning model hyperparameters using cross-validation, such as GridSearchCV or RandomizedSearchCV.
*   Exploring more advanced machine learning algorithms.
*   Using an ensemble method to increase overall accuracy of the model.

