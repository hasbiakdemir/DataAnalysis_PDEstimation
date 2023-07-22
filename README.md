## Data Analysis: Probability of Default Estimation

In this notebook, I aimed to predict whether a loan was going to be charged off or fully paid with two tree based Machine Learning (ML) model, namely Random Forest and XGBoost. I used Lending Club dataset which is a big dataset. Therefore, since some of the implementations such as oversampling using SMOTE AdaSyn or Recursive Feature Elimination with GridSearchCV was computationally expensive, you will see a lot of commented sections. However, my goal here is to show you my thought process while doing data analysis. Logistic Regression as a base model was implemented by my teammate so it is not uploaded.

#### Data Description and Data Preprocessing




I referenced these two repositories below.

1. The functions for calculating missing values and plotting the distributions of variables in the EDA part are beautifully done.
https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans/blob/master/LC_Loan_full.ipynb

2. Imputing missing values with Mice Forest algorithm. It works for both numerical and categorical features.
https://github.com/AnotherSamWilson/miceforest/tree/master/miceforest
