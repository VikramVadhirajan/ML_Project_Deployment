# Model Evaluation

Model Evaluation refers to the process of **assessing how well a machine learning model performs on unseen data**.

The goal is to determine whether a trained model **generalizes well beyond the training dataset**.

Evaluation is typically performed using a **separate test dataset** or validation techniques such as [[Cross Validation]].

---

# Why Model Evaluation is Important

Model evaluation helps to:

- measure predictive performance
    
- compare different models
    
- detect [[Overfitting]] or [[Underfitting]]
    
- select the best model for deployment
    

Without proper evaluation, a model may perform well on training data but fail on new data.

---

# Dataset Splitting

Before evaluation, data is typically divided into:

**Training Set**

Used to train the model.

**Validation Set**

Used for tuning model parameters.

**Test Set**

Used to evaluate final model performance.

See: [[Train Test Split]]

---

# Types of Evaluation Metrics

Evaluation metrics depend on the **type of machine learning task**.

---

# Regression Metrics

Used when predicting **continuous values**.

Common regression metrics:

- [[Mean Squared Error]]
    
- [[Root Mean Squared Error]]
    
- [[Mean Absolute Error]]
    
- [[R Squared]]
    

These metrics measure how close predictions are to actual values.

---

# Classification Metrics

Used when predicting **categorical classes**.

Common classification metrics:

- [[Accuracy]]
    
- [[Precision]]
    
- [[Recall]]
    
- [[F1 Score]]
    
- [[Confusion Matrix]]
    
- [[ROC Curve]]
    
- [[AUC]]
    

These metrics evaluate how well a model distinguishes between classes.

---

# Cross Validation

Cross validation improves reliability of evaluation by **splitting data into multiple training and validation sets**.

Common technique:

[[K Fold Cross Validation]]

This helps reduce variance caused by a single train-test split.

---

# Bias–Variance Considerations

Evaluation helps detect problems such as:

High training accuracy + low test accuracy → [[Overfitting]]

Low training accuracy + low test accuracy → [[Underfitting]]

Understanding these patterns helps improve model design.

---

# Model Comparison

Multiple models can be compared using evaluation metrics.

Example workflow:

1. Train several models
    
2. Evaluate each model on validation data
    
3. Compare performance metrics
    
4. Select best-performing model
    

---

# Practical Evaluation Pipeline

Typical workflow:

1. Split dataset
    
2. Train model
    
3. Generate predictions
    
4. Compute evaluation metrics
    
5. Compare models
    
6. Select best model
    

---

# Related Concepts

[[Cross Validation]]  
[[Train Test Split]]  
[[Overfitting]]  
[[Underfitting]]  
[[Bias Variance Tradeoff]]  
[[Mean Squared Error]]  
[[Accuracy]]  
[[Precision]]  
[[Recall]]  
[[F1 Score]]