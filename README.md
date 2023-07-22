# Data Analysis: Probability of Default Estimation

In this notebook, I aimed to predict whether a loan was going to be charged off or fully paid with two tree-based Machine Learning (ML) models, namely `Random Forest` and `XGBoost`. I used the Lending Club dataset, which is a big dataset. Therefore, since some of the implementations, such as oversampling using `SMOTE AdaSyn` or `Recursive Feature Elimination with GridSearchCV (RFECV)`, were computationally expensive, you will see a lot of commented sections. However, my goal here is to show you my thought process while doing data analysis. Logistic Regression as a base model was implemented by my teammate, so it was not uploaded.

## Data Description and Data Preprocessing
### Data Description

The Lending Club data is a highly imbalanced and complex dataset, containing over 2.260.700 accepted loan records with 151 columns of data. The loans are issued on the peer-to-peer lending platform. This platform allows small businesses and individuals to borrow money directly from investors, instead of using intermediaries. The data ranges from 2007 to 2018.

The records in the data represent a single loan holder with individual characteristics, which will serve as my explanatory variables in the credit default prediction. The target variable, after data processing, can be divided into two categories:
- Assigned `0` for loan statuses: `Fully Paid` and `Does not meet the credit policy. Status: Fully Paid`.
- Assigned `1` for the loan statuses: `Charged Off` and `Does not meet the credit policy. Status: Charged Off`.

### Data Preprocessing
#### Missing Features

The first step is to remove columns with too many missing values. The dataset has 151 columns, however, some columns have over 90% missing values which is not possible to extract valuable information. Therefore, I decided to drop columns with more than 50% missing data. This cutoff has deleted 58 columns and left 93 columns with a maximum percentage of missing values of 13.12%, which it would make sense to apply imputation to fill the missing values.

#### Outcome Bias and Information Reduction

Several features were not known at the origin of the loan, and therefore, can not be used in default prediction because of the `Outcome Bias`. These feature will be dropped to eliminate the bias in the model, allowing to the creation of a model that is not influenced by factors. Using these features lead to data leakage because they convey information about the ability of the loan holder to pay back the loan after the loan has been funded. Furthermore, the ID feature is not important and the categorical variables that have too many unique values (`Emp Title`) have also been dropped. After this process, there are 31 features left. These features will form the basis for my feature selection process.

#### Imputing Missing Values

I tried to employed Factor Analysis of Mixed Data (FAMD) to impute missing values for both categorical and numerical variables. However, due to the dataset’s size and FAMD's memory need, I could not use it. As a result, an alternative approach using MissForest was explored, which leverages the Random Forest model for imputing missing values but it was computataionally exhaustive.

To overcome these challenges, the MICE Forest algorithm coupled with LightGBM estimator was adopted for missing data imputation. MICE Forest enables the use of diverse predictive models as estimators such as LightGBM. It works for both numerical and categorical variables and it takes significantly less amount of time to be completed, compared to the methods above.

## Exploratory Data Analysis

In the next step, I went over the features that will be based for feature elimination. When I look at the 31 features, I see that some of the features do not contain distinctive information. For example, `Grade` feature is eliminated because it is embedded in `Sub-grade`. `Policy Code` feature only contains the value of 1, meaning that there is no information to be extracted from the feature. Also, only the features that are unknown before lending out the loans should be passed to the models. Therefore, `Issue_d` from my analysis. The other variables that are excluded from the analysis because of their high unique values are as follows: `Title`, `URL`, and `Zip Code`. Finally, there are 25 features left.

### Outliers

I plot the histograms (countplots for categorical), boxplots, and distributions (barplots for categorical) of Fully Paid and Charged-Off loans. If there is an outlier and skewness in the distribution, I applied either squareroot or log transformation to mitigate the effects of the outliers. Moreover, I grouped the values of some of the categorical variables to obtain more generalised outcomes.

### Statistical Hypothesis Testing

For numerical features, I utilised Kolmogorov-Smirnov test (KS Test) to see whether the features’ distributions based on the target feature (`loan_status`) are statistically different from each other. For categorical variables, Chi-Squared test (CS Test) is applied to check whether the observed frequencies are statistically different from what would be expected by chance and whether there is a significant association between the categorical and the target features.

### Weight of Evidence (WOE) and Information Value (IV)

To check the importance of a categorical variable, I also calculated information value (IV) which if it is higher than 0.4, it indicates that the categorical variable is a good feature. WOE tells the predictive power of an features in relation to the target. IV helps to rank features based on their importance. I did not use IV as an elimination criteria. I utilised it to see which features might be important.

## Standardisation, Encoding, and Correlation

After looking at the variables, the next steps were the encoding and standardisation of the features in the dataset. ML algorithms work best with standardisation. Standardization is a method where the observations are centred around the mean with one unit standard deviation, meaning that the mean of the feature becomes zero, and the distribution has a unit standard deviation. Therefore, I have used standardisation for numerical variables and one-hot encoding and label encoding for the features that have ordinal relationships such as `Sub-Grade` for categorical variables. After making these transformations, there were over 1.3 million observations with 98 features. When I look at the correlation map, I see that `installment` and `loan_amnt`, `total_acc` and `open_acc`, `mo_sin_old_rev_tl_op` and `earliest_cr_line` is highly correlated. Therefore, `installment`, `total_acc`, and `mo_sin_old_rev_tl_op` is decided to be dropped.

## Splitting the Dataset, Feature Selection, and Deciding Oversampling vs. Undersampling
### Splitting the Dataset

I split my dataset into three parts by selecting random observations: train, validation, and test: 60% for training, and 20% for validation and test data for each. 

### Feature Selection
After looking at correlations, I used recursive feature elimination with cross-validation to select features. RFECV is a feature selection technique that combines two popular methods (recursive feature elimination and cross-validation), to automatically identify the most relevant features for a given model.

The RFECV method works by recursively eliminating less important features and assessing the model’s performance using cross-validation. The initial step involves ranking all input features based on their importance. Then, it iteratively eliminates the least significant features until arriving at the desired number of remaining features. During each iteration, the model undergoes training and evaluation using cross-validation to estimate its performance accurately.

This process helps identify critical features that contribute significantly to predictive power while reducing dimensionality in high-dimensional datasets. As an estimator, we utilised LightGBM - a gradient-boosting framework known for handling large-scale datasets efficiently and accurately.

### Deciding Oversampling vs. Undersampling

Since this is an imbalanced dataset, after splitting my dataset, I decided to try taking the hardest route and oversampling my data with the AdaSyn method. `Adaptive Synthetic Sampling (AdaSyn)` is a data augmentation technique that addresses the issue of class imbalance in machine learning. Class imbalance occurs when one class has significantly more or fewer instances than another, leading to biased model performance and my data has this problem. The AdaSyn approach generates synthetic samples for the minority class by examining the density distribution of each feature and generating new examples along directions where data points are scarce.

I also utilised the GridSearchCV method to tune AdaSyn’s n_neighbors parameter with Stratified CV which is a technique in machine learning to evaluate the performance of classification models on imbalanced datasets. However, since the dataset got even bigger, the RFECV algorithm to select important features took a lot of time. Furthermore, I tried to train my first ML model using the linear SVM algorithm but it was computationally exhaustive to train around 1.3 million observations with 48 features.

After my initial attempt at oversampling, I decided to move forward with undersampling. I utilised Imblearn’s `RandomUnderSampler` class to create a balanced training dataset. It undersamples the majority class by randomly picking samples with or without replacement. I used the default sampling strategy. After this step, I repeat the previous RFECV step to select the best features. 61 features are selected by the model training with around 315K observations.

## Models
### Random Forest

Random Forest is a widely used ensemble learning method. It combines the predictions of multiple decision trees to create a robust and accurate model for tasks like classification or regression. One of the advantages of the Random Forest is that it has a good ability to handle high dimensional data and does not exacerbate bias since it belongs to the bagging family. However, Random Forest model is prone to overfitting and after I initailly trained my model, I achieved high score of accuracy on the train set but very low accuracy on the test set. This says that the model is overfitted.

In order to overcome the overfitting problem, I employed additional hyperparameters to tune called `max_depth` and `n_estimators`. The max_depth parameter controls the maximum allowed depth for each decision tree within the Random Forest. By limiting this parameter, we can prevent the trees from becoming excessively intricate and prone to overfitting. Similarly, n_estimators can help mitigate overfitting. In the Random Forest, each estimator is trained independently on a random subset of the data, and their predictions are combined to make the final prediction. By increasing the number of estimators, we introduce more randomness and diversity into the model. I ran GridSearchCV with Stratifed CV to tune this parameters to obtain the best values for the parameters to get rid of the overfitting problem.

### XGBoost

XGBoost, is an algorithm for optimizing gradient boosting that excels at efficiency and scalability. It solves various classification, regression, and ranking problems. It is an effective algorithm since it can handle big datasets, handle missing values, and capture complex relationships. I chose this model because unlike Random Forest, it is a boosting algorithm and it would be interesting to compare the results of both models on the same problem.

## Explainability

I looked at both models' explainability using `SHAP`. The features that explain and influence the decision of both models are `Sub-Grade`, `int_rate`, and `term`, and `loan_amnt` which makes sense. Because the grade of the loan, interest rate, term, and the loan amount are usually the most important factors when we decide whether we should not give loans.

## Conclusion Model Results

We see that all of the models that I built have similar Accuracy and AUC Score. However, our problem is predicting the status of the loan. Therefore, Precision should be considered as the most important performance metric. Because in the prediction of default or fraud, identifying the defaults correctly is essential while minimizing false positive. Therefore, precision serves as a measure of our models’ effectiveness in accurately identifying true charged off status among all the observations that they classify as positive.

In conclusion, the model into production with the highest precision score and F1-Score which is Logistic Regression should be implemented since the other performance metrics are similar to each other. After implementing the Logistic Regression model as the champion model, I should go back to the beginning and improve the prediction power of my challenger models as they consist of greater potential than logistic regression.

| Model                   | Accuracy | AUC    | Precision | F1     |
| ----------------------- |:--------:|:------:|:---------:|:------:|
| Logistic Regression     | 0.6581   | 0.6539 | 0.8823    | 0.7557 |
| Random Forest           | 0.6324   | 0.6532 | 0.3104    | 0.4278 |
| XGBoost                 | 0.6468   | 0.6568 | 0.3194    | 0.4344 |

## Final Remarks

In the project, I tried to implement my data analysis knowledge on a business problem which is one of the most important issues in finance, especially for Banks. My aim was not to build the best model in this project. The main goal for me is to try as many data analysis techniques as possible and look for areas for development.

## References

1. The functions for calculating missing values and plotting the distributions of variables in the EDA part are beautifully done.
https://github.com/yanxiali/Predicting-Default-Clients-of-Lending-Club-Loans/blob/master/LC_Loan_full.ipynb

2. Imputing missing values with MICE Forest algorithm. It works for both numerical and categorical features.
https://github.com/AnotherSamWilson/miceforest/tree/master/miceforest
