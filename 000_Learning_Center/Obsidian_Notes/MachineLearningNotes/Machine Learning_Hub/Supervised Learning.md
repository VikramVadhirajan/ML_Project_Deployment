# Supervised Learning Models

Supervised learning algorithms learn a mapping from **input features (X)** to **target labels (y)** using labeled training data.

The goal is to learn a function:

f(X) → y

that can generalize to unseen data.

Supervised learning problems are mainly divided into:

- [[Regression]] → predict continuous values
    
- [[Classification]] → predict discrete classes
    

---

# Linear Models

Linear models assume a **linear relationship between features and target**.

- [[Linear Regression]]
    
- [[Ridge Regression]]
    
- [[Lasso Regression]]
    
- [[Elastic Net]]
    
- [[Logistic Regression]]
    

Advantages:

- simple
    
- interpretable
    
- fast training
    

---

# Distance-Based Models

These models rely on **distance or similarity between data points**.

- [[K Nearest Neighbors]]
    

They classify or predict based on **neighboring samples**.

---

# Probabilistic Models

These models rely on **probability theory and Bayes theorem**.

- [[Naive Bayes]]
    
- [[Gaussian Naive Bayes]]
    
- [[Multinomial Naive Bayes]]
    
- [[Bernoulli Naive Bayes]]
    

Often used in **text classification**.

---

# Tree-Based Models

Tree-based models split the dataset based on feature values.

- [[Decision Trees]]
    
- [[Random Forest]]
    
- [[Extra Trees]]
    

Advantages:

- interpretable
    
- handle nonlinear relationships
    
- robust to outliers
    

---

# Ensemble Methods

Ensemble methods combine multiple models to improve performance.

- [[Bagging]]
    
- [[Random Forest]]
    
- [[Gradient Boosting]]
    
- [[AdaBoost]]
    
- [[XGBoost]]
    
- [[LightGBM]]
    
- [[CatBoost]]
    

These models dominate many **Kaggle competitions**.

---

# Margin-Based Models

These models focus on **maximizing the margin between classes**.

- [[Support Vector Machines]]
    
- [[Support Vector Regression]]
    

---

# Neural Network Models

These models use layered neural structures to learn complex relationships.

- [[Neural Networks]]
    
- [[Deep Neural Networks]]
    
- [[CNN]]
    
- [[RNN]]
    

---

# Important Concepts

These concepts are critical when working with supervised learning models:

- [[Model Evaluation]]
    
- [[Overfitting]]
    
- [[Underfitting]]
    
- [[Bias Variance Tradeoff]]
    
- [[Cross Validation]]
    
- [[Regularization]]
    

---

# Typical Supervised Learning Workflow

1. [[Data Cleaning]]
    
2. [[Missing Data Handling]]
    
3. [[Outlier Treatment]]
    
4. [[Encoding Categorical Variables]]
    
5. [[Feature Scaling]]
    
6. Train supervised model
    
7. [[Model Evaluation]]