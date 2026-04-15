# Cross Validation

Cross Validation is a technique used to **evaluate model performance by splitting the dataset into multiple training and validation subsets**.

It provides a **more reliable estimate of model performance**.

---

# K-Fold Cross Validation

Most common approach.

Process:

1. Split dataset into K folds
    
2. Train model on K−1 folds
    
3. Validate on remaining fold
    
4. Repeat K times
    
5. Average performance
    

---

# Advantages

- reduces evaluation bias
    
- uses entire dataset for training and validation
    

---

# Limitations

- computationally expensive
    

---

# Related Concepts

[[Model Evaluation]]  
[[Model Tuning]]  
[[Grid Search]]