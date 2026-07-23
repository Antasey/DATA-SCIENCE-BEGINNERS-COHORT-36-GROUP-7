import pandas as pd

TARGET = "Attrition"

def load_csv(path):
    return pd.read_csv(path)

def basic_report(df):
    print("shape:", df.shape)
    print("\nnulls:\n", df.isnull().sum()[df.isnull().sum() > 0])
    print("\ndupes:", df.duplicated().sum())
    print("\ndtypes:\n", df.dtypes)
