
![[Pasted image 20260416194809.png]]

# Random Forest

Random Forest is an **ensemble learning algorithm** that builds multiple [[Decision Trees]] and combines their predictions to produce a final result.

It improves model performance by **reducing overfitting and variance** compared to a single decision tree.

Random Forest can be used for:

- Classification
    
- Regression
    

---

# Core Idea

Instead of training a single decision tree, Random Forest trains **many trees on different subsets of the data** and aggregates their predictions.

This process is based on the principle of **wisdom of the crowd**.

Each tree makes a prediction, and the final output is obtained by combining these predictions.

---

# How Random Forest Works

The algorithm follows these steps:

1. Draw multiple bootstrap samples from the training dataset. 
	1. creating multiple new dataset randomly from the existing dataset
	2. (Sample with replacement or consistent size or OOB- 36.8%)
    
2. Train a [[Decision Trees]] model on each sample (random selection of Feature).
    
3. At each split, select a **random subset of features**.
    
4. Grow many decision trees independently.
    
5. Combine predictions from all trees.
    

---

# Bootstrap Sampling

Random Forest uses **bootstrap sampling**, which means sampling data **with replacement**.

Example:

Dataset with 100 samples → each tree is trained on a random sample of those 100 observations.

This introduces diversity among trees.

---

# Feature Randomness

At each split in a tree, Random Forest considers **only a random subset of features** instead of all features.

This helps reduce **correlation between trees**, improving overall performance.

---

# Prediction Process

## Classification

Each tree votes for a class.

Final prediction = **majority vote**.

---

## Regression

Each tree outputs a numerical value.

Final prediction = **average of predictions**.

---

# Advantages

- reduces overfitting compared to a single tree
    
- handles high-dimensional data well
    
- robust to noise and outliers
    
- requires little feature scaling
    

---

# Limitations

- less interpretable than a single decision tree
    
- computationally expensive for large forests
    
- can require more memory
    

---

# Important Hyperparameters

Number of trees (`n_estimators`)

Controls how many trees are built.

More trees usually improve performance but increase computation.

---

Maximum tree depth (`max_depth`)

Limits how deep each tree can grow.

Helps control overfitting.

---

Maximum features (`max_features`)
(_n_estimators=100_, _*_, _criterion='gini'_, _max_depth=None_, _min_samples_split=2_, _min_samples_leaf=1_, _min_weight_fraction_leaf=0.0_, _max_features='sqrt'_, _max_leaf_nodes=None_, _min_impurity_decrease=0.0_, _bootstrap=True_, _oob_score=False_, _n_jobs=None_, _random_state=None_, _verbose=0_, _warm_start=False_, _class_weight=None_, _ccp_alpha=0.0_, _max_samples=None_, _monotonic_cst=None_)[[source]](https://github.com/scikit-learn/scikit-learn/blob/fe2edb3cd/sklearn/ensemble/_forest.py#L1174)
Number of features considered at each split.

Common values:

- sqrt(number_of_features) for classification
    
- log2(number_of_features)
    

---

# Feature Importance

Random Forest can estimate **feature importance** by measuring how much each feature reduces impurity across trees.

This helps identify **which features contribute most to predictions**.

---

# Out-of-Bag (OOB) Error

Because bootstrap sampling leaves some samples unused in each tree, those samples can be used as **validation data**.

This is called **Out-of-Bag error estimation**.

It provides an internal estimate of model performance without needing a separate validation set.

---

# Random Forest vs Decision Trees

Decision Trees:

- prone to [[Overfitting]]
    
- high variance
    

Random Forest:

- reduces variance
    
- more stable predictions
    

---

# Example (Python)

Using **scikit-learn**:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Related Concepts

[[Decision Trees]]  
[[Ensemble Learning]]  
[[Bagging]]  
[[Overfitting]]  
[[Model Evaluation]]  
[[Feature Importance]]