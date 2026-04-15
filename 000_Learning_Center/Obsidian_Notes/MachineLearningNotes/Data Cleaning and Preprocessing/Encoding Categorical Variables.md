# Encoding Categorical Variables

Encoding Categorical Variables is the process of **converting categorical data into numerical form so that machine learning algorithms can process it**.

Most machine learning models operate on **numerical inputs**, therefore categorical features must be transformed before training.

Example categorical variable:

|Color|
|---|
|Red|
|Blue|
|Green|

These values must be converted into numbers.

Encoding is part of the **[[Data Preprocessing]] pipeline**.

---

# Types of Categorical Variables

Understanding the type of categorical variable helps determine the appropriate encoding method.

---

## Nominal Variables

Nominal variables represent categories **without any natural order**.

Examples:

- Color (Red, Blue, Green)
    
- Country
    
- Product Category
    

Order does **not matter**.

---

## Ordinal Variables

Ordinal variables have a **natural ordering among categories**.

Examples:

- Education level (High School, Bachelor, Master, PhD)
    
- Satisfaction level (Low, Medium, High)
    

Order **contains meaningful information**.

---

# Why Encoding is Necessary

Machine learning models interpret numbers mathematically.

If categorical variables remain as text, algorithms cannot compute relationships between variables.

Encoding transforms categories into **numerical representations that models can understand**.

---

# Encoding Methods

## Label Encoding

Label encoding assigns a **unique integer to each category**.

Example:

|Color|Encoded|
|---|---|
|Red|0|
|Blue|1|
|Green|2|

Advantages:

- Simple
    
- Memory efficient
    

Limitations:

Models may incorrectly assume **ordinal relationships between categories**.

Recommended for:

- ordinal variables
    

See: [[Ordinal Encoding]]

---

## One-Hot Encoding

One-hot encoding creates **binary columns for each category**.

Example:

|Color|Red|Blue|Green|
|---|---|---|---|
|Red|1|0|0|
|Blue|0|1|0|
|Green|0|0|1|

Advantages:

- Removes false ordinal relationships
    
- Widely used in machine learning
    

Limitations:

- increases dimensionality
    
- inefficient when many categories exist
    

---

## Binary Encoding

Binary encoding converts categories into **binary numbers**, reducing dimensionality compared to one-hot encoding.

Example:

Category index → binary representation.

Advantages:

- fewer columns than one-hot encoding
    
- useful for high-cardinality features
    

---

## Target Encoding

Target encoding replaces each category with the **mean value of the target variable for that category**.

Example:

|City|Average House Price|
|---|---|
|City A|250000|
|City B|300000|

Advantages:

- captures relationship between feature and target
    
- effective for high-cardinality variables
    

Limitations:

- risk of **data leakage**
    

---

# High Cardinality Features

High cardinality occurs when a categorical variable contains **many unique values**.

Examples:

- ZIP codes
    
- User IDs
    
- Product IDs
    

Problems:

- One-hot encoding creates too many columns.
    

Solutions:

- target encoding
    
- binary encoding
    
- embedding methods
    

---

# Dummy Variable Trap

In one-hot encoding, using **all categories may introduce multicollinearity**.

Example:

For three categories:

Red  
Blue  
Green

One column can be removed.

This is called **dropping the reference category**.

See: [[Multicollinearity]]

---

# Example (Python)

```python
import pandas as pd

df = pd.read_csv("data.csv")

# One-hot encoding
df = pd.get_dummies(df, columns=['color'])

# Label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['city'] = le.fit_transform(df['city'])
```

---

# Choosing an Encoding Method

General guidelines:

Nominal variables → One-hot encoding

Ordinal variables → Label encoding

High-cardinality variables → Target encoding or binary encoding

Choice may also depend on the machine learning algorithm.

Tree-based models often handle label encoding well.

---

# Related Concepts

[[Data Cleaning]]  
[[Data Preprocessing]]  
[[Feature Engineering]]  
[[Feature Scaling]]  
[[Multicollinearity]]  
[[Target Encoding]]