# Decision Trees

## Definition

Decision Trees are supervised learning models that make predictions by **recursively splitting the dataset based on feature values**.

They create a tree-like structure where:

- internal nodes represent feature tests
    
- branches represent decisions
    
- leaf nodes represent predictions
    

Decision trees can be used for both [[Classification]] and [[Regression]].

---

## Problem Type

- Classification
    
- Regression
    

---

## Core Idea

The algorithm splits the dataset into subsets based on **feature values that maximize class separation**.

Each split aims to reduce **impurity in the dataset**.

---

## Mathematical Formulation

Common impurity measures:

Gini Impurity:

Gini = 1 − Σ pᵢ²

Entropy:

Entropy = − Σ pᵢ log₂(pᵢ)

Where pᵢ is the probability of class i.

---

## Training Process

1. Start with the entire dataset
    
2. Evaluate possible splits
    
3. Select the best split using impurity measures
    
4. Recursively split nodes
    
5. Stop when a stopping condition is met
    

---

## Important Hyperparameters

max_depth → maximum tree depth  
min_samples_split → minimum samples required to split  
min_samples_leaf → minimum samples per leaf

---

## Advantages

- easy to interpret
    
- handles nonlinear relationships
    
- works with numerical and categorical data
    

---

## Limitations

- prone to [[Overfitting]]
    
- unstable with small data changes
    

---

## Applications

- credit risk assessment
    
- medical diagnosis
    
- customer segmentation
    

---

## Related Concepts

[[Random Forest]]  
[[Gradient Boosting]]  
[[Model Evaluation]]  
[[Overfitting]]