# Principal Component Analysis (PCA)

## Definition

PCA is a dimensionality reduction technique that transforms features into **orthogonal components capturing maximum variance**.

---

## Core Idea

Instead of using original features, PCA creates new variables called **principal components**.

Each component explains a portion of variance in the data.

---

## Mathematical Concept

Principal components are **eigenvectors of the covariance matrix**.

Variance captured is given by **eigenvalues**.

---

## Training Process

1. Standardize features
    
2. Compute covariance matrix
    
3. Calculate eigenvectors
    
4. Project data onto principal components
    

---

## Advantages

- reduces dimensionality
    
- removes redundant features
    
- improves visualization
    

---

## Limitations

- components are difficult to interpret
    
- sensitive to feature scaling
    

---

## Applications

- data visualization
    
- noise reduction
    
- feature extraction
    

---

## Related Concepts

[[Dimensionality Reduction]]  
[[Linear Algebra]]  
[[Feature Scaling]]