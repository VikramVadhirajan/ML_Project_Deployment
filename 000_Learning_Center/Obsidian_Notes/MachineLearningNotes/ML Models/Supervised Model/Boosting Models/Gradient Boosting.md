# Gradient Boosting

## Definition

Gradient Boosting is an ensemble method that builds models **sequentially**, where each new model attempts to correct errors made by previous models.

It is commonly implemented using **decision trees as weak learners**.

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

Instead of training models independently, gradient boosting trains models **in sequence**, focusing on correcting residual errors.

---

## Mathematical Formulation

The model is built as:

F(x) = Σ γₘ hₘ(x)

Where:

hₘ(x) = weak learner  
γₘ = learning rate

---

## Training Process

1. Train initial model
    
2. Compute residual errors
    
3. Train new model on residuals
    
4. Add new model to ensemble
    
5. Repeat for many iterations
    

---

## Important Hyperparameters

n_estimators → number of trees  
learning_rate → contribution of each tree  
max_depth → depth of trees

---

## Advantages

- high predictive accuracy
    
- handles complex patterns
    
- widely used in competitions
    

---

## Limitations

- slower training
    
- sensitive to hyperparameters
    
- prone to overfitting without regularization
    

---

## Applications

- ranking systems
    
- financial prediction
    
- fraud detection
    

---

## Related Concepts

[[Decision Trees]]  
[[Random Forest]]  
[[Ensemble Learning]]