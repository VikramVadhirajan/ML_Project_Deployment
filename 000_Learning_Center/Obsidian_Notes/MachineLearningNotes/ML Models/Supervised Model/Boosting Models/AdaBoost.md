# AdaBoost

## Definition

AdaBoost (Adaptive Boosting) is an ensemble algorithm that combines **multiple weak learners into a strong classifier**.

It focuses on **misclassified samples during training**.

---

## Problem Type

- Classification
    

---

## Core Idea

Each new model gives more importance to **previously misclassified data points**.

---

## Mathematical Formulation

Final model:

F(x) = sign( Σ αₘ hₘ(x) )

Where:

αₘ = weight of learner  
hₘ(x) = weak classifier

---

## Training Process

1. Assign equal weights to training samples
    
2. Train weak learner
    
3. Increase weight for misclassified samples
    
4. Train next learner
    
5. Combine predictions
    

---

## Important Hyperparameters

n_estimators → number of learners  
learning_rate → shrinkage factor

---

## Advantages

- improves weak learners
    
- relatively simple ensemble method
    

---

## Limitations

- sensitive to noisy data
    
- sensitive to outliers
    

---

## Applications

- face detection
    
- text classification
    

---

## Related Concepts

[[Gradient Boosting]]  
[[Ensemble Learning]]  
[[Decision Trees]]