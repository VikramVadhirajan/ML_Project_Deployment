# Data Cleaning

Data Cleaning is the process of **identifying and correcting errors, inconsistencies, and inaccuracies in a dataset** to improve data quality before analysis or model training.

Raw data collected from real-world sources is often **incomplete, noisy, or inconsistent**, which can negatively affect machine learning models.

Data cleaning ensures that the dataset is **reliable, consistent, and usable for analysis**.

---

# Importance of Data Cleaning

Machine learning models rely heavily on data quality.

Poor quality data can lead to:

- inaccurate predictions
    
- biased models
    
- unreliable insights
    
- training instability
    

In many real-world ML projects, **data cleaning takes the majority of development time**.

---

# Common Data Issues

## Missing Data

Missing values occur when no value is stored for a feature.

Example:

|Age|Salary|
|---|---|
|25|50000|
|NA|62000|

See: [[Missing Data Handling]]

---

## Duplicate Records

Duplicate rows may appear due to data entry errors or merging datasets.

Example:

|ID|Name|
|---|---|
|101|John|
|101|John|

Duplicates can bias model training.

---

## Inconsistent Data

Values may have inconsistent formats.

Example:

Country column:

USA  
U.S.A  
United States

These represent the same category but appear different to the model.

---

## Outliers

Outliers are **data points that significantly deviate from the rest of the data**.

Example:

Typical salary range:

40k – 120k

Outlier:

2,000,000

Outliers can distort statistical measures and model training.

See: [[Outlier Treatment]]

---

## Incorrect Data Types

Features may be stored in incorrect formats.

Example:

Age stored as a string instead of numeric.

---

# Common Data Cleaning Steps

## Remove Duplicates

Identify and remove duplicate rows to avoid biased analysis.

---

## Handle Missing Values

Strategies include:

- Removing rows with missing values
    
- Imputing values using statistics
    
- Using model-based imputation
    

See: [[Missing Data Handling]]

---

## Fix Data Types

Ensure each feature has the correct type.

Examples:

- numerical
    
- categorical
    
- datetime
    
- boolean
    

---

## Standardize Formats

Convert inconsistent values into a standard format.

Example:

Date formats:

01/02/2024  
2024-02-01

Standardization improves consistency.

---

## Handle Outliers

Possible approaches:

- remove extreme values
    
- cap values using percentile thresholds
    
- transform the feature
    

See: [[Outlier Treatment]]

---

# Data Validation

After cleaning, the dataset should be validated to ensure:

- correct data types
    
- no unexpected missing values
    
- consistent feature ranges
    

---

# Data Cleaning vs Data Preprocessing

Data Cleaning focuses on **fixing data quality issues**.

Data Preprocessing includes additional steps such as:

- [[Feature Scaling]]
    
- [[Encoding Categorical Variables]]
    
- [[Feature Engineering]]
    

---

# Practical Workflow

Typical data preparation pipeline:

1. Load raw dataset
    
2. Inspect dataset
    
3. Identify data issues
    
4. Clean dataset
    
5. Validate cleaned data
    
6. Proceed to preprocessing
    

---

# Tools for Data Cleaning

Common tools used in machine learning:

- Python (pandas)
    
- SQL
    
- OpenRefine
    
- Data validation libraries
    

Example using pandas:

```python
import pandas as pd

df = pd.read_csv("data.csv")

df = df.drop_duplicates()
df = df.dropna()
```

---

# Related Concepts

[[Data Preprocessing]]  
[[Missing Data Handling]]  
[[Outlier Treatment]]]  
[[Feature Engineering]]  
[[Feature Scaling]]  
[[Encoding Categorical Variables]]