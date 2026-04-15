# Semi-Supervised Learning

Semi-Supervised Learning is a machine learning paradigm that uses **both labeled and unlabeled data** for training.

In many real-world problems, **labeled data is scarce and expensive**, while **unlabeled data is abundant and cheap**.  
Semi-supervised learning leverages the structure of unlabeled data to improve model performance.

It lies between:

- [[Supervised Learning]] (only labeled data)
    
- [[Unsupervised Learning]] (no labels)
    

---

# Motivation

Labeling data often requires **human experts**.

Examples:

Medical imaging → requires doctors  
Speech transcription → requires manual annotation  
Image labeling → requires human tagging

Because of this, datasets typically contain:

- a **small labeled dataset**
    
- a **large unlabeled dataset**
    

Semi-supervised learning uses both.

---

# Dataset Structure

Example dataset:

Labeled data:

|Input|Label|
|---|---|
|Image 1|Cat|
|Image 2|Dog|

Unlabeled data:

|Input|
|---|
|Image 3|
|Image 4|
|Image 5|

The model learns patterns from **both labeled and unlabeled samples**.

---

# Key Assumptions

Semi-supervised learning relies on certain assumptions about data distribution.

---

## Smoothness Assumption

Points that are **close in feature space should have similar labels**.

---

## Cluster Assumption

Data points tend to form clusters, and **points in the same cluster are likely to share a label**.

---

## Manifold Assumption

High-dimensional data often lies on a **lower-dimensional manifold**, which can be exploited during learning.

---

# Common Methods

## Self-Training

Steps:

1. Train model using labeled data
    
2. Predict labels for unlabeled data
    
3. Add confident predictions to labeled dataset
    
4. Retrain the model
    

This process repeats iteratively.

---

## Co-Training

Two models are trained using **different feature subsets**.

Each model labels unlabeled data for the other model.

---

## Label Propagation

Labels are spread across the dataset based on **similarity between data points**.

Often used with graph-based methods.

---

## Semi-Supervised Deep Learning

Modern approaches use deep learning techniques such as:

- pseudo-labeling
    
- consistency regularization
    
- generative models
    

These approaches are widely used in **computer vision and NLP**.

---

# Advantages

- reduces the need for large labeled datasets
    
- improves performance when labeled data is limited
    
- leverages abundant unlabeled data
    

---

# Limitations

- incorrect pseudo-labels may introduce noise
    
- performance depends on assumptions about data distribution
    
- implementation complexity
    

---

# Applications

Semi-supervised learning is used in:

- image recognition
    
- speech recognition
    
- medical image analysis
    
- web page classification
    
- recommendation systems
    

---

# Example (Python)

Using scikit-learn's Label Propagation:

```python
from sklearn.semi_supervised import LabelPropagation

model = LabelPropagation()
model.fit(X, y)
```

Here, `y` may contain **missing labels (-1)** for unlabeled data.

---

# Related Concepts

[[Supervised Learning]]  
[[Unsupervised Learning]]  
[[Self Training]]  
[[Label Propagation]]  
[[Deep Learning]]