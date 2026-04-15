# Missing Data Handling

Missing Data Handling refers to techniques used to **detect, analyze, and manage missing values in a dataset**.

Missing data occurs when **no value is stored for a variable in an observation**. If not handled properly, it can lead to **biased models, incorrect analysis, and reduced model performance**.

Handling missing data is an essential step in [[Data Cleaning]] and [[Data Preprocessing]].

---

# What is Missing Data

A missing value indicates that **data for a particular feature is unavailable**.

Example dataset:

|Age|Salary|Experience|
|---|---|---|
|25|50000|2|
|NA|62000|4|
|30|NA|5|

Here, Age and Salary contain missing values.

---

# Causes of Missing Data

## Data Collection Issues

Data may not be collected due to survey errors or incomplete forms.

Example:  
A respondent skips the income question.

---

## Data Entry Errors

Human mistakes during manual data entry.

---

## Sensor or Measurement Failure

Devices may fail to record values during data collection.

---

## Data Integration Problems

Missing values may appear when merging datasets from different sources.

---

# Types of Missing Data

Understanding the **mechanism of missingness** helps determine the correct treatment method.

---

## Missing Completely at Random (MCAR)

The probability of missing data is **independent of any variable in the dataset**.

Example:  
Random system failure causes some records to be lost.

Characteristics:

- Missingness is purely random
    
- Removing these rows does not introduce bias
    

---

## Missing at Random (MAR)

The missingness depends on **other observed variables but not the missing variable itself**.

Example:

Income may be missing more often for younger respondents.

Missingness depends on **Age**, not Income itself.

---

## Missing Not at Random (MNAR)

The missingness depends on **the value that is missing**.

Example:

High-income individuals may choose not to report their income.

This is the **most difficult case to handle**.

---

# Detecting Missing Data

Common techniques include:

- summary statistics
    
- null value counts
    
- visualizations
    

Example using Python:

```python
df.isnull().sum()
```

This shows the number of missing values in each column.

---

# Missing Data Handling Techniques

## 1. Deletion Methods

### Listwise Deletion

Remove rows that contain missing values.

Advantages:

- simple to implement
    

Disadvantages:

- loss of data
    
- may introduce bias if data is not MCAR
    

---

### Column Deletion

Remove features with too many missing values.

Common rule:

Remove columns with **more than 40–60% missing values**.

---

## 2. Simple Imputation

Replace missing values with statistical measures.

### Mean Imputation

Replace missing values with the **mean of the column**.

Example:

```python
df['age'].fillna(df['age'].mean())
```

Limitation:

Reduces variance in the data.

---

### Median Imputation

Replace missing values with the **median**.

Better for **skewed distributions**.

---

### Mode Imputation

Used for **categorical variables**.

Replace missing values with the most frequent category.

---

## 3. Forward and Backward Fill

Used for **time series data**.

Forward fill:

Replace missing value with the previous observation.

Backward fill:

Replace missing value with the next observation.

---

## 4. Model-Based Imputation

Predict missing values using machine learning models.

Example methods:

- K-Nearest Neighbors Imputation
    
- Regression Imputation
    

These methods use relationships between features to estimate missing values.

---

## 5. Indicator Method

Create a new binary feature indicating whether the value was missing.

Example:

Age_missing = 1 if age missing else 0

This allows models to **capture missingness as information**.

---

# Choosing the Right Method

The appropriate method depends on:

- amount of missing data
    
- type of variable
    
- missing data mechanism (MCAR, MAR, MNAR)
    
- downstream machine learning model
    

---

# Practical Workflow

1. Detect missing values
    
2. Analyze missingness patterns
    
3. Determine missing data mechanism
    
4. Choose an appropriate imputation strategy
    
5. Validate results
    

---

# Example (Python)

```python
import pandas as pd

df = pd.read_csv("data.csv")

# check missing values
df.isnull().sum()

# fill missing values
df['age'] = df['age'].fillna(df['age'].median())
```

---

# Related Concepts

[[Data Cleaning]]  
[[Outlier Treatment]]  
[[Feature Engineering]]  
[[Feature Scaling]]  
[[Encoding Categorical Variables]]  
[[KNN Imputation]]