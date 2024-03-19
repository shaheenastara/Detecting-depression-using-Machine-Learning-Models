
# Project Title - Detecting depression in individuals based on demographics using Machine Learning model
# Objective 
Depression is a prevalent mental health disorder that affects millions of people worldwide. Early detection and intervention are crucial for effective treatment and management of depression. This project aims to develop a machine learning model to predict the likelihood of depression based on demographics of people in a rural area. By leveraging demographic factors such as Age, Gender, Education level, Income level, Employment, Marital status, the model will provide valuable insights into the risk factors associated with depression. Additionally, the project seeks to explore the potential of ML techniques in mental health assessment and contribute to the development of scalable and accessible solutions for mental health screening and monitoring.
# Dataset
The project will utilize a dataset containing demographic information (e.g., age, gender, income, education level) and depression status. The dataset has been taken from Kaggle.
# Methodology
Data Collection: Demographic and depression status data has been collected from Kaggle for this project
Data Preprocessing: Cleanse the dataset, handle missing values, encode categorical variables, and perform feature engineering if necessary.
Exploratory Data Analysis (EDA): Analyze the distribution of demographic variables and their relationship with depression status. Identify correlations and patterns in the data.
Feature Selection: Select relevant demographic features that contribute to predicting depression using techniques such as statistical tests or feature importance analysis.
Model Development: Train and evaluate machine learning models (e.g., logistic regression, decision trees, random forests, SVM) using the selected demographic features. Experiment with different algorithms and hyperparameters to optimize performance.
Model Evaluation: Assess the model's performance using appropriate evaluation metrics (e.g., accuracy, precision, recall, ROC-AUC). Validate the model using cross-validation to ensure robustness.
Interpretation: Interpret the model results to understand the relationship between demographic factors and depression identification. Identify key demographic predictors of depression and their relative importance.
# Model Summary
KNN,SVC and Random Forest Classidier Model is used for Depression Detection. The objective of this model is to predict the likelihood of depression based on demographic conditions. The model utilizes demographic features such as age, gender, income, expense level, education level, and marital status to predict depression status.
The target variable is depression status, which is binary - depressed or non-depressed. The model is based on SVC, a method for binary classification. It models the probability of depression as a function of the demographic features
The model was trained on a dataset containing demographic information and depression status. Features were standardized, and hyperparameters were optimized using grid search with cross-validation.
The model's performance was evaluated using accuracy, precision, recall, and ROC-AUC
# Language used - install python 3.6.8
# Libraries used
numpy : it is used for numerical operations in Python.
pandas : for data manipulation and analysis.
StandardScaler: For data preprocessing
sklearn.decomposition PCA: Import the PCA class from scikit-learn for performing Principal Component Analysis.
matplotlib: Import the matplotlib library for creating plots and visualizations.
seaborn: Import the seaborn library for improved data visualization.
train_test_split: module from sklearn to split the dataset into training and testing sets.
GrisdSearchCV: module from sklearn for hyperparameter tuning
accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc: Import metrics for evaluating the classifier's performance
# Data Preprocessing
### Dropping rows with null values
### Scaling using StandardScaler
### One Hot Encoding
# Model used
### K-Nearest Neighbours model is used for classification.Hyperparameter tuning is done for the parameters ‘p’, ‘weights’, ‘n_neighbors’.Cross Validation is done across five folds.The dataset is split in the train:test ratio of 75:25 and setting the random seed at 100.Parameters used to evaluate performance are Accuracy, Confusion matrix, Precision , Recall and ROC-AUC Curve
### SVC model is used for this binary  classification.Hyperparameter tuning is done for the parameters ‘c’ and ‘gamma’.Kernel used is ‘linear’.Cross Validation is done across five folds.The dataset is split in the train:test ratio of 75:25 and setting the random seed at 100.SVC is used along with Principal Component Analysis as well as Feature Selection (SelectFromModel) to reduce dimensionality for ease of convergence.Parameters used to evaluate performance are Accuracy, Confusion matrix, Precision , Recall and ROC-AUC Curve
### Random Forest is an ensemble method. It is used for this binary classification task. The complexity of Random Forest model is higher than KNN and SVC.Hyperparameters are specified during model initialisation. n_estimators=7,  random_state=100, min_samples_split=2, min_samples_leaf=4Cross Validation is done across five folds. The dataset is split in the train:test ratio of 75:25 and setting the random seed at 100.Random Forest Classifier is used along with Principal Component Analysis as well as Feature Selection (SelectFromModel) to reduce dimensionality for ease of convergence. Parameters used to evaluate performance are Accuracy, Confusion matrix, Precision , Recall and ROC-AUC Curve

# Performance evaluation
###  K-Nearest Neighbours is achieving accuracy above 80% for both training and testing data but isn't able to classify positives well. The complexity of KNN depends on the number of training instances and ,here, the dataset is smaller so it is giving better accuracy than other models. However, the AUC value shows that the model is not doing good at classifying instances¶
### Support Vector Classifier(SVC) is underfitting which implies model is too simple to capture the underlying patters. Also, the model is showing inconsistent result even after setting a fixed random seed
### To increase model complexity, Random Forest Classifier is used with Principal Component Analysis(PCA) as well as Feature Importance but the models are showing only moderately good accuracy. More training data can help cover a wider range of scenarios and variations present in the features, and, hence increase accuracy. This allows the model to learn more robust patterns and generalizable representations of the data, reducing the risk of memorizing specific instances, and leading to improved performance on unseen data and overcome overfitting.
### In the field of medical science, accuracy alone is not the best metric for evaluating depression detection models because the dataset might be imbalanced (i.e., more non-depressed individuals than depressed ones), leading to a high accuracy even if the model performs poorly on detecting depression cases. Instead, metrics like sensitivity (true positive rate), specificity (true negative rate), precision (positive predictive value) are often used to evaluate the performance of binary classification models used in depression detection.
### For a machine learning model to detect depression, both precision and recall are vital. But their importance vary depending on the application. In this medical context, misdiagnosing individuals as depressed (False Positive) will lead to unnecessary treatment while on the other hand, failing to correctly classify individuals as depressed will aggravate their mental health condition. Therefore, a balance between precision and recall is important but ,in this scenario, prioritizing recall to minimize False Negatives is more critical

# Future Aspects
### Furthermore, depression detection is a complex and multifaceted task that cannot be based only on demographics. It requires structured data (e.g., demographic information, questionnaire responses) but also unstructured data (e.g., text from social media posts, audio recordings of speech). The living conditions alone cannot be an indicator of depression. We would also need an insight into the regular habits eg. hobbies, social media etc. Detection of depression will be, comparatively, more accurate if both types of data are combined. If the input contains different types of data then Deep Learning models need to be used to evaluate.
###  In the future, I would like to build a model to analyse and detect depression, and build a function in a smart watch or an application in the phone which gives positive/motivational reminders to people to help people deal with depression, along with medical guidance, and guide them to a positive lifestyle


# Reference
### kaggle.com
### cloudxlab.com

![image](https://github.com/shaheenastara/Detecting-Depression-using-ML-Model/assets/139441568/0a0fa3af-b262-4338-9cdc-59c0ae4ef11a)
