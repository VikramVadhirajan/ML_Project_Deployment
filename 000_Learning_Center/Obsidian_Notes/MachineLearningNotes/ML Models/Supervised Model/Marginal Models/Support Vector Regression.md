


# Support Vector Regression (SVR)

## Definition

Support Vector Regression is the regression counterpart of [[Support Vector Machines]] that predicts continuous values while maintaining an **epsilon margin (tube) around predictions**.

Only errors **outside this margin are penalized**, making the model robust to small deviations.

---

## Problem Type

- Regression

---

## Model Visualization

![[svr_epsilon_tube.png]]

The diagram shows:

- Regression function
- **Epsilon tube** around the prediction
- **Support vectors** lying outside the tube
- Only points outside the tube contribute to the loss

---

## Core Idea

Instead of minimizing prediction error directly, SVR tries to **fit a function that stays within an epsilon distance from the actual data points**.

Small errors inside the epsilon tube are ignored.

---

## Mathematical Formulation

### Regression Function

The predicted function is:

$$
f(x) = w \cdot x + b
$$

Where:

- **w** → weight vector  
- **x** → input feature vector  
- **b** → bias term

---

### Optimization Objective

SVR minimizes the following objective:

$$
\min \frac{1}{2} ||w||^2 + C \sum (\xi_i + \xi_i^*)
$$

Where:

- **||w||²** → controls model complexity  
- **C** → regularization parameter  
- **ξᵢ , ξᵢ\*** → slack variables for errors outside epsilon margin

---

### Epsilon-Insensitive Loss

Errors inside the epsilon tube are ignored.

Loss function:

$$
L =
\begin{cases}
0 & |y - f(x)| \le \epsilon \\
|y - f(x)| - \epsilon & \text{otherwise}
\end{cases}
$$

---

## Training Process

1. Map features into higher dimensional space
2. Define an **epsilon margin (tube)**
3. Fit regression function
4. Penalize points outside the epsilon tube
5. Use kernel trick for nonlinear relationships

---

## Important Hyperparameters

C → regularization parameter controlling error penalty  

epsilon → width of the tolerance tube  

kernel → linear, polynomial, RBF  

gamma → influence of training points (RBF kernel)

---

## Advantages

- Works well with high-dimensional data
- Robust to small noise due to epsilon margin
- Can model nonlinear regression with kernels

---

## Limitations

- Computationally expensive for large datasets
- Parameter tuning required
- Harder to interpret compared to linear regression

---

## Applications

- stock price prediction
- demand forecasting
- energy load prediction
- time series regression

---

## Related Concepts

[[Support Vector Machines]]

[[Support Vector Classification]]

[[Kernel Trick]]

[[Regression]]