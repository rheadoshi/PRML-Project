# PRML-Project

Stroke is the second leading cause of death globally, accounting for approximately 11% of total deaths. Stroke is a condition that occurs when the blood supply to the brain is interrupted or reduced due to a blockage (ischemic stroke) or rupture of a blood vessel (hemorrhagic stroke). Without blood, the brain will not get oxygen and nutrients, so cells in some areas of the brain will die. This condition causes parts of the body controlled by the damaged area of the brain to not function properly. Early prediction of stroke occurrence is crucial for preventive healthcare. Understanding and predicting stroke risk can aid in preventive measures and early intervention, potentially saving lives. This project presents an analysis of a dataset from Kaggle aimed at predicting stroke occurrence based on factors such as gender, age, lifestyle choices (smoking status or residence or type of job), diseases such as hypertension and heart diseases.

## Understanding the given dataset 

Dataset: The dataset contains 5110 rows and 11 columns, with one additional target column 'stroke'.

### Description of the features

[Insert description of features here]

### Data Visualisation

![Visualization 1](https://example.com/visualization1.png)
![Visualization 2](https://example.com/visualization2.png)

# Data Pre-Processing and Preparation

We first filled missing values of the bmi feature. We implemented the Synthetically Minority Over-Sampling Technique. We did this to remove the imbalance between the classes. So we generated more data points of the minority class (Stroke=1). Then next, we removed outliers from the columns, avg_glucose_level and bmi using the IQR method. We then finally split our dataset into training and testing sets.

# Models

We trained our data on several machine learning models including Decision Tree, Random Forest, Naive Bayes, ANN, etc.

# Evaluation Metrics

We evaluated the performance of our models using various evaluation metrics such as accuracy, precision, recall, and F1-score.

# Results

[Insert discussion of results and findings here]

# Conclusion

[Insert conclusion and final remarks here]
