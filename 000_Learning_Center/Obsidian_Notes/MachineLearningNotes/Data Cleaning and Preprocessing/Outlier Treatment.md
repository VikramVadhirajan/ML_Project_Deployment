# Outlier Treatment

Outlier Treatment refers to techniques used to **handle extreme values in a dataset that significantly differ from other observations**.

Outliers can arise due to **measurement errors, data entry mistakes, or genuine rare events**. If not handled properly, they may distort statistical analysis and negatively impact machine learning models.

---

# What is an Outlier

An outlier is a data point that lies **far away from the majority of observations in a dataset**.

Example:

Dataset (house prices):

200000  
210000  
220000  
230000  
5000000

The value **5,000,000** is an outlier.

---

# Causes of Outliers

## Data Entry Errors

Incorrect manual entry of data.

Example:  
Age recorded as **200 instead of 20**.

---

## Measurement Errors

Faulty sensors or instruments may produce incorrect values.

---

## Data Processing Errors

Errors during data transformation or merging datasets.

---

## Genuine Rare Events

Some outliers represent **valid but rare observations**.

Example:  
Very high income in a salary dataset.

---

# Why Outliers Matter

Outliers can affect:

- mean and variance
    
- model coefficients
    
- training stability
    
- prediction accuracy
    

Some algorithms are **highly sensitive to outliers**.

Sensitive models:

- [[Linear Regression]]
    
- [[Logistic Regression]]
    
- [[K Means]]
    

More robust models:

- [[Decision Trees]]
    
- [[Random Forest]]
    

---

# Detecting Outliers

## Z-Score Method

Measures how many standard deviations a data point is from the mean.

Formula:

Z = (x − μ) / σ

Common rule:

|Z| > 3 → potential outlier

---

## Interquartile Range (IQR) Method

Based on quartiles.

IQR = Q3 − Q1

Outlier thresholds:

Lower bound:

Q1 − 1.5 × IQR

Upper bound:

Q3 + 1.5 × IQR

Values outside this range are considered outliers.

---

## Visualization Methods

### Box Plot

Highlights extreme values using whiskers.

### Scatter Plot

Useful for detecting anomalies in relationships.

### Histogram

Helps identify unusual distributions.

---

# Outlier Treatment Methods

## Removing Outliers

Remove observations when:

- they are clearly errors
    
- they distort analysis significantly
    

Risk:

May remove important rare cases.

---

## Capping (Winsorization)

Replace extreme values with threshold limits.

Example:

Values above 95th percentile → replaced with 95th percentile.

Benefits:

Preserves dataset size while limiting extreme influence.

---

## Transformation

Apply transformations that reduce skewness.

Common transformations:

- Log transformation
    
- Square root transformation
    
- Box-Cox transformation
    

See: [[Feature Transformation]]

---

## Robust Models

Use algorithms that are less sensitive to outliers.

Examples:

- [[Decision Trees]]
    
- [[Random Forest]]
    

---

# When Not to Remove Outliers

Outliers should not be removed if they represent **real and important events**.

Examples:

- fraud detection
    
- rare diseases
    
- extreme weather events
    

In these cases, outliers may contain **valuable information**.

---

# Practical Workflow

1. Detect potential outliers
    
2. Investigate their cause
    
3. Decide whether they are errors or valid observations
    
4. Apply an appropriate treatment method
    

---

# Example (Python)

```python
import numpy as np

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_filtered = df[(df['salary'] >= lower) & (df['salary'] <= upper)]
```

---

# Related Concepts

[[Data Cleaning]]  
[[Linear Regression]]  
[[Decision Trees]]  
