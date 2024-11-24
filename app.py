import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from modules.data_preparation import set_data
from modules.feature_selection import select_features
from modules.model_training import grid_search_ann, train_and_evaluate_model

# Streamlit App
# Streamlit App UI setup
st.set_page_config(page_title="Breast Cancer Prediction and Analysis", layout="wide")

# Add a custom CSS style for the background
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://www.uab.edu/news/images/2018/Breast_cancer_Month.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Breast Cancer Prediction and Analysis")

# Load and preprocess data
st.write("### Dataset Overview")
X_train, X_test, y_train, y_test = set_data()
st.write("Training Data Sample:")
st.write(X_train.head())

# Feature selection
st.write("### Feature Selection")
X_train_selected, selected_features, scaler = select_features(X_train, y_train, X_train.columns.tolist(), k=10)
X_test_selected = X_test.loc[:, selected_features]
X_test_scaled = scaler.transform(X_test_selected)  # Manually scale the test data
st.write("Selected Features:")
st.write(selected_features)

# Feature importance visualization
st.write("### Feature Importance")
selector = SelectKBest(score_func=f_classif, k=len(X_train.columns))
selector.fit(X_train, y_train)
feature_scores = pd.DataFrame({
    "Feature": X_train.columns,
    "Score": selector.scores_
}).sort_values(by="Score", ascending=False).head(25)

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x="Score", y="Feature", data=feature_scores, ax=ax, palette="viridis")
ax.set_title("Top 25 Feature Importance (F-Score)")
ax.set_xlabel("F-Score")
st.pyplot(fig)

# Grid search and training
st.write("### Model Training")
st.write("Performing grid search for hyperparameter tuning...")
best_model = grid_search_ann(X_train_selected, y_train)

# Evaluate model
accuracy, report = train_and_evaluate_model(best_model, X_train_selected, X_test_scaled, y_train, y_test)
st.write(f"**Test Accuracy:** {accuracy:.2f}")
st.text("Classification Report:")
st.text(report)

# Create tabs for user interaction and visualization
tabs = st.tabs(["Prediction", "Model Evaluation", "Data Visualization"])

with tabs[0]:
    st.write("### Make a Prediction")
    input_data = []
    for feature in selected_features:
        value = st.number_input(f"Enter value for {feature}:", value=0.0)
        input_data.append(value)

    if st.button("Predict"):
        # Manually scale the input data
        input_scaled = scaler.transform([input_data])  # Use the scaler from feature selection
        prediction = best_model.predict(input_scaled)
        result = "Malignant" if prediction[0] == 1 else "Benign"
        st.markdown(f"### Prediction: **{result}**")

with tabs[1]:
    st.write("### Confusion Matrix")
    y_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Malignant"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### ROC Curve")
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

with tabs[2]:
    st.write("### Data Visualization")
    selected_feature_indices = [X_train.columns.get_loc(feature) for feature in selected_features]

    # Now use these indices to access the columns
    feature = st.selectbox("Select a feature for distribution visualization", X_train.columns[selected_feature_indices].tolist())
    fig, ax = plt.subplots()
    sns.histplot(X_train[feature], kde=True, ax=ax)
    st.pyplot(fig)

    st.write("#### Class Distribution")
    fig, ax = plt.subplots()
    y_train.value_counts().plot(kind="bar", color=["blue", "orange"], ax=ax)
    st.pyplot(fig)

    st.write("#### Feature Correlation Heatmap")
    corr = X_train.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)