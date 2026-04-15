# CatBoost

## Definition

CatBoost is a gradient boosting algorithm developed by Yandex that **handles categorical features efficiently**.

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

CatBoost introduces:

- ordered boosting
    
- efficient categorical encoding
    

This reduces prediction shift and improves performance.

---

## Training Process

1. Encode categorical features internally
    
2. Train boosting trees sequentially
    
3. Optimize loss function
    

---

## Important Hyperparameters

iterations  
learning_rate  
depth

---

## Advantages

- excellent handling of categorical variables
    
- minimal preprocessing required
    
- strong performance
    

---

## Limitations

- slower than LightGBM in some cases
    
- less widely used than XGBoost
    

---

## Applications

- recommendation systems
    
- classification problems with categorical data
    

---

## Related Concepts

[[Gradient Boosting]]  
[[Encoding Categorical Variables]]  
[[XGBoost]]