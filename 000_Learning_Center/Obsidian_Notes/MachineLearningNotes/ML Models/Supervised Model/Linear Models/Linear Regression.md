
![[Pasted image 20260416181924.png]]

# Linear Regression

Linear Regression is a supervised learning algorithm used to model the relationship between **input variables and a continuous output variable**.

The model assumes the target variable is a **linear combination of the input features**.

---

## Mathematical Model

Simple linear regression:

y = β0 + β1x + ε

Where:

- y → target variable
    
- x → input feature
    
- β0 → intercept
    
- β1 → coefficient
    
- ε → random error
    

---

## Multiple Linear Regression

When multiple features exist:

y = β0 + β1x1 + β2x2 + ... + βnxn + ε

Matrix form:

y = Xβ + ε

---

# Hyper Parameters

(_*_, _fit_intercept=True_, _copy_X=True_, _tol=1e-06_, _n_jobs=None_, _positive=False_)

---

## Model Prediction

Predicted output:

ŷ = β0 + Σ βixi

---

## Training Objective

The goal is to estimate parameters β that minimize prediction error.

See: [[Loss Functions]]

---

## Training Methods

- [[Normal Equation]]
    
- [[Gradient Descent]]
    

---

## Residuals

Residuals represent the difference between predicted and actual values.

Residual = y − ŷ

See: [[Residual Analysis]]

---

## Assumptions

Linear regression relies on several assumptions:

- Linearity
    
- Independence of observations
    
- Homoscedasticity
    
- Normal distribution of errors
    
- No strong [[Multicollinearity]]

---
# Python Documentation

https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html