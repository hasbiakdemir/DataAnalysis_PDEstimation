# Data Analysis: Probability of Default Estimation

In this notebook, I aimed to predict whether a loan was going to be charged off or fully paid with two tree-based Machine Learning (ML) models, namely Random Forest and XGBoost. I used the Lending Club dataset, which is a big dataset. Therefore, since some of the implementations, such as oversampling using SMOTE AdaSyn or Recursive Feature Elimination with GridSearchCV, were computationally expensive, you will see a lot of commented sections. However, my goal here is to show you my thought process while doing data analysis. Logistic Regression as a base model was implemented by my teammate, so it was not uploaded.

## Data Description and Data Preprocessing
### Data Description

The Lending Club data is a complex dataset, containing over 2.260.700 accepted loan records with 151 columns of data. The loans are issued on the peer-to-peer lending platform. This platform allows small businesses and individuals to borrow money directly from investors, instead of using intermediaries. The data ranges from 2007 to 2018.

The records in the data represent a single loan holder with individual characteristics, which will serve as my explanatory variables in the credit default prediction. Our target variable, after data processing, can be divided into two categories:
- Assigned `0` for loan statuses: `Fully Paid` and `Does not meet the credit policy. Status: Fully Paid`.
- Assigned `1` for the loan statuses: `Charged Off` and `Does not meet the credit policy. Status: Charged Off`.

### Data Preprocessing
#### Missing Features
The first step is to remove columns with too many missing values. The dataset has 151 columns, however, some columns have over 90% missing values which is not possible to extract valuable information. Therefore, I decided to drop columns with more than 50% missing data. This cutoff has deleted 58 columns and left 93 columns with a maximum percentage of missing values of 13.12%, which it would make sense to apply imputation to fill the missing values.

#### Outcome Bias and Information Reduction
Several features were not known at the origin of the loan, and therefore, can not be used in default prediction because of the `Outcome Bias`. These feature will be dropped to eliminate the bias in the model, allowing to the creation of a model that is not influenced by factors. Using these features lead to data leakage because they convey information about the ability of the loan holder to pay back the loan after the loan has been funded. Furthermore, the ID feature is not important and the categorical variables that have too many unique values (`Emp Title`) have also been dropped. After this process, there are 31 features left. These features will form the basis for our feature selection process.

#### Imputing Missing Values
I tried to employed Factor Analysis of Mixed Data (FAMD) to impute missing values for both categorical and numerical variables. However, due to the dataset’s size and FAMD's memory need, I could not use it. As a result, an alternative approach using MissForest was explored, which leverages the Random Forest model for imputing missing values but it was computataionally exhaustive.

To overcome these challenges, the MICE Forest algorithm coupled with LightGBM estimator was adopted for missing data imputation. MICE Forest enables the use of diverse predictive models as estimators such as LightGBM. It works for both numerical and categorical variables and it takes significantly less amount of time to be completed, compared to the methods above.

## Exploratory Data Analysis

In the next step, I went over the features that will be based for feature elimination. When I look at the 31 features, I see that some of the features do not contain distinctive information. For example, `Grade` feature is eliminated because it is embedded in `Sub-grade`. `Policy Code` feature only contains the value of 1, meaning that there is no information to be extracted from the feature. Also, only the features that are unknown before lending out the loans should be passed to the models. Therefore, `Issue_d` from our analysis. The other variables that are excluded from the analysis because of their high unique values are as follows: `Title`, `URL`, and `Zip Code`. Finally, there are 25 features left.

### Outliers

I plot the histograms (countplots for categorical), box-plots, distributions (barplots for categorical) of Fully Paid and Charged-Off loans. If there is an outlier and skewness in the distribution, I applied either squareroot or log transformation to mitigate the effects of the outliers. Moreover, I grouped the values of some of the categorical variables to obtain more generalised outcomes.

### Statistical Hypothesis Testing
For numerical features, I utilised Kolmogorov-Smirnov test (KS Test) to see whether the features’ distributions based on the target feature (`loan_status`) are statistically different from each other. For categorical variables, Chi-Squared test (CS Test) is applied to check whether the observed frequencies are statistically different from what would be expected by chance and whether there is a significant association between the categorical and the target features.

### Weight of Evidence (WOE) and Information Value (IV)
To check the importance of a categorical variable, I also calculated information value (IV) which if it is higher than 0.4, it indicates that the categorical variable is a good feature. WOE tells the predictive power of an features in relation to the target. IV helps to rank features based on their importance. I did not use IV as an elimination criteria. I utilised it to see which features might be important.

## Feature Selection



## References

1. The functions for calculating missing values and plotting the distributions of variables in the EDA part are beautifully done.
https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans/blob/master/LC_Loan_full.ipynb

2. Imputing missing values with MICE Forest algorithm. It works for both numerical and categorical features.
https://github.com/AnotherSamWilson/miceforest/tree/master/miceforest
