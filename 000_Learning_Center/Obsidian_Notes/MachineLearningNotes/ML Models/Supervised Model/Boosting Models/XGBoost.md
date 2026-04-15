# XGBoost

## Definition

XGBoost (Extreme Gradient Boosting) is an optimized implementation of [[Gradient Boosting]] designed for **speed and performance**.

It includes additional **regularization and system optimizations**.

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

XGBoost improves gradient boosting by:

- regularization
    
- parallelization
    
- tree pruning
    

---

## Mathematical Formulation

Objective function:

Obj = Loss + Regularization

Regularization helps control model complexity.

---

## Training Process

1. Train trees sequentially
    
2. Compute gradients of loss
    
3. Fit trees to gradients
    
4. Apply regularization
    

---

## Important Hyperparameters

n_estimators  
learning_rate  
max_depth  
subsample

---

## Advantages

- very high performance
    
- handles missing values
    
- efficient training
    

---

## Limitations

- complex tuning
    
- less interpretable
    

---

## Applications

- Kaggle competitions
    
- recommendation systems
    
- fraud detection
    

---

## Related Concepts

[[Gradient Boosting]]  
[[LightGBM]]  
[[CatBoost]]