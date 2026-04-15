# Hierarchical Clustering

## Definition

Hierarchical clustering builds a **tree-like hierarchy of clusters** called a dendrogram.

---

## Core Idea

Clusters are formed by **recursively merging or splitting clusters**.

Two approaches:

Agglomerative → bottom-up  
Divisive → top-down

---

## Training Process

1. Treat each data point as a cluster
    
2. Compute distances between clusters
    
3. Merge closest clusters
    
4. Repeat until one cluster remains
    

---

## Linkage Methods

Single linkage  
Complete linkage  
Average linkage

---

## Advantages

- no need to specify number of clusters initially
    
- produces hierarchical structure
    

---

## Limitations

- computationally expensive
    
- sensitive to noise
    

---

## Applications

- biological taxonomy
    
- document organization
    

---

## Related Concepts

[[Clustering]]  
[[Dendrogram]]  
[[Distance Metrics]]