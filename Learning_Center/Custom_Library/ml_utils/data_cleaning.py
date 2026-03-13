import pickle
from sklearn.impute import KNNImputer, SimpleImputer


def find_duplicates(df):
    """Return duplicate rows"""
    return df[df.duplicated()]


def remove_duplicates(df):
    """Remove duplicate rows"""
    return df.drop_duplicates()

def missing_values(df):
    """Return missing value count per column"""
    return df.isnull().sum()


def drop_null_rows(df):
    return df.dropna()

def get_column_types(df):
    
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    
    categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    
    return numeric_cols, categorical_cols


def numerical_imputer(df,columns, n_neighbors=5):
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    
    df[columns] = knn_imputer.fit_transform(df[columns])
    with open("knn_imputer.pkl", "wb") as f:
        pickle.dump(knn_imputer, f)
    
    return df, knn_imputer

def categorical_imputer(df, columns, strategy='most_frequent'):
    
    categorical_imputer = SimpleImputer(strategy=strategy)
    
    df[columns] = categorical_imputer.fit_transform(df[columns])

    with open("categorical_imputer.pkl", "wb") as f:
        pickle.dump(categorical_imputer, f)
    
    return df, categorical_imputer



def drop_columns(df, columns):
    return df.drop(columns=columns, axis=1)


def clean_column_names(df):
    
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    
    return df

def get_column_types(df):
    
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    
    categorical_cols = df.select_dtypes(include=['object','category']).columns
    
    return numeric_cols, categorical_cols


def remove_constant_columns(df):
    
    return df.loc[:, df.nunique() > 1]