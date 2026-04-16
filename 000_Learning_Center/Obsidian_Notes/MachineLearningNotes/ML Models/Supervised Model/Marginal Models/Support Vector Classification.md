
![[Pasted image 20260416183030.png]]

# Support Vector Classification (SVC)

## Definition

Support Vector Classification is a classification algorithm based on [[Support Vector Machines]] that finds the **optimal hyperplane separating different classes with the maximum margin**.

---

## Problem Type

- Classification

---

## Model Visualization

![[svc_decision_boundary.png]]

The plot shows:

- Two different classes
- The **decision boundary (hyperplane)**
- **Support vectors** that determine the boundary

---

## Core Idea

SVC attempts to **find a decision boundary that maximizes the margin between two classes**.

The data points closest to the decision boundary are called **support vectors**, and they determine the position of the hyperplane.

---

## Mathematical Formulation

### Hyperplane Equation

The decision boundary is defined as:

$$
w \cdot x + b = 0
$$

Where:

- **w** → weight vector  
- **x** → input feature vector  
- **b** → bias (intercept)

---

### Classification Rule

$$
y =
\begin{cases}
+1 & \text{if } w \cdot x + b \ge 0 \\
-1 & \text{if } w \cdot x + b < 0
\end{cases}
$$

---

### Optimization Objective

SVM tries to **maximize the margin while minimizing classification errors**.

$$
\min \frac{1}{2} ||w||^2 + C \sum \xi_i
$$

Where:

- **||w||²** → controls margin width
- **C** → regularization parameter
- **ξᵢ** → slack variables allowing misclassification

---

### Margin Boundaries

Support vectors lie on these boundaries:

$$
w \cdot x + b = 1
$$

$$
w \cdot x + b = -1
$$

Margin width:

$$
\frac{2}{||w||}
$$

---

## Training Process

1. Identify the optimal hyperplane separating the classes
2. Maximize the margin between the classes
3. Allow some misclassification using slack variables
4. Use kernel functions to handle non-linear boundaries

---

## Important Hyperparameters



C → regularization parameter controlling margin vs misclassification  

kernel → linear, polynomial, RBF  

gamma → controls influence of individual training points  

degree → used for polynomial kernel

---

## Advantages

- Effective in high-dimensional spaces
- Works well when number of features > number of samples
- Can model nonlinear boundaries using kernels

---

## Limitations

- Computationally expensive for large datasets
- Sensitive to kernel and parameter selection
- Less interpretable compared to simpler models

---

## Applications

- text classification
- image classification
- bioinformatics classification
- spam detection

---

## Python Documentation

https://scikit-learn.org/stable/modules/svm.html#svm-classification

---


## Related Concepts

[[Support Vector Machines]]

[[Support Vector Regression]]

[[Kernel Trick]]

[[Margin Maximization]]