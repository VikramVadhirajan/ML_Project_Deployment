import pickle
from sklearn.impute import KNNImputer, SimpleImputer


def find_duplicates(df):
    """Return duplicate rows"""
    return df[df.duplicated()]


def remove_duplicates(df):
    """Remove duplicate rows"""
    return df.drop_duplicates()

def missing_values(df):
    """Return missing value count per column with percentage"""
    return (df.isnull().sum()/len(df)*100).round(2).sort_values(ascending=False)    


def drop_null_rows(df):
    return df.dropna()

def get_column_types(df):
    
    numeric_cols = (df
                    .select_dtypes(include=['int64','float64'])
                    .columns
                    .tolist())
    
    categorical_cols = (df
                        .select_dtypes(include=['object','category'])
                        .columns
                        .tolist())
    
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
    df_dropped= df.drop(columns=columns, axis=1)
    return df_dropped


def clean_column_names(df):
    
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    
    return df

def remove_constant_columns(df):
    
    return df.loc[:, df.nunique() > 1]


def unqiue_value_count(df,columns):
    for column in columns:
        print("The Column - {} has {}". format(column,df[column].unique()))


def clean_strings(df):
    import string
    import numpy as np
    symbol_List = list(string.punctuation)
    for i in df.columns:
        df[i]=df[i].apply(lambda x: np.nan if x in symbol_List else x)