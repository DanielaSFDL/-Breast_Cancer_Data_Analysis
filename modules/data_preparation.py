

# Importing necessary libraries
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split

def set_data():
    # 1. Dataset Acquisition and Preparation
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    # 2. Data Cleaning
    # Check for missing values
    missing_data = X.isnull().sum()
    print(f"Missing values in each feature:\n{missing_data}")
    
    # Ensure there are no missing values
    if missing_data.sum() == 0:
        print("No missing values detected.")
    
    # Check for duplicates
    duplicates = X.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")
    
    # Drop duplicates if any
    if duplicates > 0:
        X.drop_duplicates(inplace=True)
        print("Duplicate rows dropped.")
    else:
        print("No duplicate rows detected.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = set_data()
    print("Data loaded, cleaned, and split successfully!")
