from sklearn.preprocessing import LabelEncoder
import pickle

def label_encode(df, columns):
    encoders={}

    for column in columns:
        le = LabelEncoder()        
        df[column] = le.fit_transform(df[column])
        encoders[column] = le

    with open("encoder.pkl", "wb") as f:
        pickle.dump(encoders, f)
    
    return df, encoders


def target_label_encode(df, column):
    le = LabelEncoder()        
    df = le.fit_transform(df[column])
    with open("target_label_encode.pkl", "wb") as f:
        pickle.dump(le, f)
    
    return df, target_label_encode