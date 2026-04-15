# Logistic Regression

Logistic Regression is a **supervised learning algorithm used for binary classification problems**.  
It models the **probability that an input belongs to a particular class**.

Despite its name, logistic regression is **a classification algorithm**, not a regression algorithm.

---

# Problem Setup

In binary classification, the target variable takes two values:

y ∈ {0,1}

Examples:

- Spam vs Not Spam
    
- Fraud vs Legitimate Transaction
    
- Disease vs Healthy
    

The goal is to estimate:

P(y = 1 | x)

which represents the **probability that the input belongs to class 1**.

---

# Linear Model

Logistic regression begins with a **linear combination of features**:

z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

Where:

- β = model parameters
    
- x = input features
    
- z = linear score
    

However, this value can range from **−∞ to +∞**, which is not valid for probabilities.

Therefore we apply a transformation.

---

# Sigmoid Function

The sigmoid function converts the linear score into a **probability between 0 and 1**.

σ(z) = 1 / (1 + e⁻ᶻ)

Final prediction:

P(y = 1 | x) = σ(β₀ + β₁x₁ + ... + βₙxₙ)

Properties:

- Output range: (0,1)
    
- Smooth and differentiable
    
- Monotonic increasing
    

See: [[Activation Functions]]

---

# Decision Boundary

To convert probability into a class label:

ŷ = 1 if P(y=1|x) ≥ 0.5  
ŷ = 0 if P(y=1|x) < 0.5

The boundary where the model switches classes is called the **decision boundary**.

The decision boundary occurs when:

β₀ + β₁x₁ + ... + βₙxₙ = 0

This forms a **linear boundary** in feature space.

---

# Log-Odds (Logit Interpretation)

Logistic regression models the **log odds** of the positive class.

Odds:

P / (1 − P)

Log odds:

log(P / (1 − P)) = β₀ + β₁x₁ + ... + βₙxₙ

This is called the **logit function**.

Interpretation:

Each coefficient represents the **change in log-odds for a one-unit increase in the feature**.

---

# Training Objective

Logistic regression parameters are estimated using **Maximum Likelihood Estimation (MLE)**.

Instead of minimizing squared error, the model minimizes:

See: [[Cross Entropy Loss]]

---

# Optimization

The loss function for logistic regression **does not have a closed-form solution**.

Therefore parameters are learned using optimization algorithms such as:

- [[Gradient Descent]]
    
- [[Stochastic Gradient Descent]]
    

---

# Regularized Logistic Regression

To prevent overfitting, regularization can be added.

## L2 Regularization

See: [[Ridge Regression]]

Penalty:

λ Σ β²

---

## L1 Regularization

See: [[Lasso Regression]]

Penalty:

λ Σ |β|

---

# Geometric Interpretation

Logistic regression finds a **linear decision boundary** separating two classes.

Dimensions:

1 feature → point threshold  
2 features → line  
3 features → plane  
n features → hyperplane

---

# Multiclass Logistic Regression

Logistic regression can be extended to multiclass classification using:

### One-vs-Rest (OvR)

Train one classifier per class.

### Softmax Regression

See: [[Softmax Regression]]

---

# Advantages

- Simple and interpretable
    
- Probabilistic output
    
- Works well for linearly separable data
    
- Efficient to train
    

---

# Limitations

- Assumes linear decision boundary
    
- Struggles with complex nonlinear relationships
    
- Sensitive to multicollinearity
    

---

# Related Concepts

[[Supervised Learning]]  
[[Linear Models]]  
[[Sigmoid Function]]  
[[Cross Entropy Loss]]  
[[Gradient Descent]]  
[[Softmax Regression]]  
[[Regularization]]