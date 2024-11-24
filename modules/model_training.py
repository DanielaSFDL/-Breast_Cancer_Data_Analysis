from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def grid_search_ann(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlpclassifier', MLPClassifier(random_state=42))
    ])

    param_grid = {
        'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (200,)],
        'mlpclassifier__activation': ['relu', 'tanh'],
        'mlpclassifier__solver': ['adam', 'sgd'],
        'mlpclassifier__max_iter': [200, 300]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"])
    
    return accuracy, report

if __name__ == "__main__":
    from modules.data_preparation import set_data
    from modules.feature_selection import select_features

    # Load and prepare the dataset
    X_train, X_test, y_train, y_test = set_data()
    feature_names = X_train.columns.tolist()

    # Perform feature selection and scaling on the training set
    X_train_scaled, selected_features, scaler = select_features(X_train, y_train, feature_names, k=10)

    # Select corresponding features for X_test
    X_test_selected = X_test[selected_features]
     # Manually scale the X_test_selected data using the same scaler
    X_test_scaled = scaler.transform(X_test_selected)

    # Perform Grid Search for the best ANN model
    print("Starting Grid Search for ANN hyperparameters...")
    best_model = grid_search_ann(X_train_scaled, y_train)

    # Train and evaluate the best ANN model
    print("Training and evaluating the best ANN model...")
    accuracy, report = train_and_evaluate_model(best_model, X_train_scaled, X_test_scaled, y_train, y_test)
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
