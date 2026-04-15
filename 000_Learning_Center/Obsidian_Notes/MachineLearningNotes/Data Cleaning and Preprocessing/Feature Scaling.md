# Feature Scaling

Feature Scaling is the process of **transforming numerical features so they are on a similar scale**.

In many machine learning algorithms, features with larger numerical ranges can dominate those with smaller ranges. Scaling ensures that **all features contribute proportionally to the model**.

Feature scaling is an important step in the [[Data Preprocessing]] pipeline.

---

# Why Feature Scaling is Important

Consider the following features:

|Feature|Range|
|---|---|
|Age|18 – 60|
|Salary|30,000 – 200,000|

Without scaling, models may assign **more importance to Salary simply because its values are larger**.

Feature scaling helps:

- improve model convergence
    
- prevent dominance of large-scale features
    
- stabilize optimization algorithms
    

---

# Algorithms Sensitive to Feature Scaling

Some machine learning algorithms rely on **distance calculations or gradient-based optimization**, making scaling essential.

Sensitive models:

- [[K Nearest Neighbors]]
    
- [[Support Vector Machines]]
    
- [[Logistic Regression]]
    
- [[Linear Regression]]
    
- [[Neural Networks]]
    
- [[K Means]]
    

Less sensitive models:

- [[Decision Trees]]
    
- [[Random Forest]]
    
- [[Gradient Boosting]]
    

These models split data based on thresholds rather than distances.

---

# Common Feature Scaling Techniques

## Standardization (Z-score Scaling)

Standardization transforms features so they have:

- mean = 0
    
- standard deviation = 1
    

Formula:

z = (x − μ) / σ

Where:

- x = original value
    
- μ = mean of feature
    
- σ = standard deviation
    

Advantages:

- widely used
    
- works well with gradient-based algorithms
    

Used in:

- [[Logistic Regression]]
    
- [[Support Vector Machines]]
    
- [[Neural Networks]]
    

---

## Min-Max Scaling (Normalization)

Min-max scaling rescales values into a **fixed range**, usually between 0 and 1.

Formula:

x' = (x − min(x)) / (max(x) − min(x))

Advantages:

- preserves original distribution
    
- useful for neural networks
    

Limitations:

- sensitive to [[Outlier Treatment]]
    

---

## Robust Scaling

Robust scaling uses **median and interquartile range (IQR)** instead of mean and standard deviation.

Formula:

x' = (x − median) / IQR

Advantages:

- resistant to outliers
    
- stable for skewed distributions
    

Used when dataset contains extreme values.

---

# When to Apply Feature Scaling

Feature scaling should typically be applied:

- after [[Data Cleaning]]
    
- after [[Missing Data Handling]]
    
- after [[Encoding Categorical Variables]]
    
- before model training
    

---

# Example (Python)

Using **scikit-learn**:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Min-Max scaling:

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

# Feature Scaling vs Normalization

Feature scaling is a **general concept** that includes several techniques.

Normalization often refers specifically to **min-max scaling**.

---

# Important Considerations

Scaling parameters should be computed **only on the training dataset** and then applied to the test dataset.

Example workflow:

1. Fit scaler on training data
    
2. Transform training data
    
3. Transform test data using the same scaler
    

This prevents **[[Data Leakage]]**.

---

# Practical Workflow

1. Clean dataset ([[Data Cleaning]])
    
2. Handle missing values ([[Missing Data Handling]])
    
3. Encode categorical variables ([[Encoding Categorical Variables]])
    
4. Apply feature scaling
    
5. Train machine learning model
    

---

# Related Concepts

[[Data Preprocessing]]  
[[Data Cleaning]]  
[[Missing Data Handling]]  
[[Encoding Categorical Variables]]  
[[Outlier Treatment]]  
[[K Nearest Neighbors]]  
[[Support Vector Machines]]