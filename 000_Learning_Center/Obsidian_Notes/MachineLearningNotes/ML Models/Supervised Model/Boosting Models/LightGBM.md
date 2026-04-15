# LightGBM

## Definition

LightGBM is a gradient boosting framework developed by Microsoft designed for **high efficiency and scalability**.

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

LightGBM uses:

- leaf-wise tree growth
    
- histogram-based training
    

These methods improve training speed and accuracy.

---

## Training Process

1. Build trees sequentially
    
2. Expand leaf with maximum loss reduction
    
3. Continue until stopping criteria
    

---

## Important Hyperparameters

num_leaves  
learning_rate  
max_depth  
n_estimators

---

## Advantages

- very fast training
    
- handles large datasets
    
- efficient memory usage
    

---

## Limitations

- can overfit small datasets
    
- sensitive to parameter tuning
    

---

## Applications

- ranking systems
    
- large-scale ML problems
    

---

## Related Concepts

[[Gradient Boosting]]  
[[XGBoost]]  
[[CatBoost]]