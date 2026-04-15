# DBSCAN

## Definition

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups points based on **density of data points**.

---

## Core Idea

Clusters are defined as regions with **high point density separated by low-density areas**.

---

## Key Concepts

Core points → points with many neighbors

Border points → near core points

Noise points → outliers

---

## Important Hyperparameters

eps → neighborhood radius

min_samples → minimum points required for cluster

---

## Advantages

- detects arbitrary-shaped clusters
    
- identifies noise automatically
    
- no need to specify number of clusters
    

---

## Limitations

- sensitive to parameter selection
    
- struggles with varying densities
    

---

## Applications

- anomaly detection
    
- spatial data analysis
    

---

## Related Concepts

[[Clustering]]  
[[Outlier Detection]]