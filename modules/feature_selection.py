from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

def select_features(X_train, y_train, feature_names, k=10):
    # Feature selection using SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Get the selected feature names
    selected_features = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    
    return X_train_selected, selected_feature_names, scaler

if __name__ == "__main__":
    from modules.data_preparation import set_data

    # Load and prepare the data
    X_train, X_test, y_train, y_test = set_data()
    feature_names = X_train.columns.tolist()

    # Perform feature selection and scaling on the training set
    X_train_scaled, selected_features, scaler = select_features(X_train, y_train, feature_names, k=10)

    # Select corresponding features for X_test
    X_test_selected = X_test[selected_features]

    # Manually scale the X_test_selected data using the same scaler
    X_test_scaled = scaler.transform(X_test_selected)


    # Output the results
    print("Selected features:", selected_features)
    print(f"Shape of training features after scaling: {X_train_scaled.shape}")
    print(f"Shape of test features after selection: {X_test_selected.shape}")