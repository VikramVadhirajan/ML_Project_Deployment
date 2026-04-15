# K Means Clustering

## Definition

K-Means is a clustering algorithm that partitions data into **K clusters based on similarity**.

Each cluster is represented by its **centroid (mean of cluster points)**.

---

## Core Idea

The algorithm assigns each data point to the **nearest cluster centroid**.

Cluster centers are iteratively updated until convergence.

---

## Mathematical Formulation

Distance measure:

Euclidean distance

d(x, μ) = √Σ(xᵢ − μᵢ)²

Objective:

Minimize within-cluster variance.

---

## Training Process

1. Choose number of clusters K
    
2. Initialize cluster centroids
    
3. Assign points to nearest centroid
    
4. Recalculate centroids
    
5. Repeat until convergence
    

---

## Important Hyperparameters

K → number of clusters  
max_iterations → maximum iterations  
init → centroid initialization method

---

## Advantages

- simple and fast
    
- scalable for large datasets
    
- easy to implement
    

---

## Limitations

- requires predefined K
    
- sensitive to [[Outlier Treatment]]
    
- assumes spherical clusters
    

---

## Applications

- customer segmentation
    
- image compression
    
- document clustering
    

---

## Related Concepts

[[Clustering]]  
[[Unsupervised Learning]]  
[[Feature Scaling]]