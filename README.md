# A-semantic-Measure-of-Online-Review-Helpfulness-using-AI

This study used Amazon.com appliance reviews to explore new ways of predicting how helpful an online review is, going beyond traditional user vote-based methods.

## Review Helpfulness Prediction: Data Pipeline and Modeling

### 1. Input Features
   
Feature Name:	Description

    Number of Helpfulness Evaluations:	Count of helpfulness votes received by the review
    
    Number of Stars:	Star rating given in the review
    
    Number of Words in the Product Description:	Word count of the product description
    
    Number of Words in the Review:	Word count of the review
    
    Readability/ARI:	Automated Readability Index score of the review
    
    Posting Order:	Order in which the review was posted
    
    Sentiment Score Deviation:	Deviation in sentiment score of the review
    
    Information Entropy Increment:	Increase in information entropy, indicating novelty or informativeness

### 2. Log Transformation

To normalize skewed distributions and reduce variance, the following columns were log-transformed:

  1. Number of Helpfulness Evaluations
  
  2. Number of Words in the Review
  
  3. Number of Words in the Product Description
  
  4. Posting Order

  5. Target Column: Review Helpfulness

### 3. Handling Class Imbalance with CTGAN

To address the imbalance in the target classes:

Step-by-Step Process:

  1. Train-Test Split:
  
      Split ratio: 80% training / 20% testing
      
      Stratified by the target variable to preserve class distribution
  
  2. Class Balancing Strategy:
  
      Identify the majority class (Zero Inputs)
      
      Separate the minority classes and group them to ensure each group has ≥ 300 samples
      
      Use CTGAN (Conditional Tabular GAN) to generate synthetic samples for each group
      
  3. CTGAN Training and Sampling:
  
    For each minority group:
    
      a. Ensure all numeric features are clipped to avoid negative values
      
      b. Train CTGAN on the group
      
      c. Sample synthetic data equal to the size of the majority class
      
      d. Post-process synthetic data to ensure validity
      
      e. Concatenate all synthetic samples with the original training set
  
  4. Final Training Dataset:
  
      Contains original majority and minority samples, along with synthetic minority samples
      
      Randomly shuffled for training consistency
  
  5. Distribution Check:
  
      Print original vs. augmented class distributions for validation

### 4. Regression Modeling
    Multiple regression algorithms were applied to predict the helpfulness score:
    
    Model Name	Notes
    
    RandomForestRegressor:	Ensemble of decision trees using bagging
    
    DecisionTreeRegressor:	Simple tree-based model
    
    BayesianRidge	Linear: regression with Bayesian regularization
    
    KNeighborsRegressor:	Based on k-nearest neighbors
    
    XGBRegressor:	Gradient boosting using XGBoost
    
    LGBMRegressor:	Gradient boosting using LightGBM
    
    CatBoostRegressor:	Gradient boosting with categorical support
    
    LinearRegression:	Standard least squares linear model
    
    ### Best Performing Model
    
    Model: LGBMRegressor
    
    ### Performance:
    
    R² Score: 0.90

Indicates that the model explains 90% of the variance in the target variable.

**All the codes are private.**
