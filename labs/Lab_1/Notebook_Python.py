## End-to-End Machine Learning Pipeline with Data Analysis
"""
This hands-on lab will guide you through creating an end-to-end machine learning pipeline in Python using Jupyter Notebook. We will cover the following steps:

1. Installing and importing necessary libraries.
2. Downloading and collecting a dataset.
3. Performing and Exploratory Data Analysis
4. Processing the data (cleaning and feature engineering).
5. Model Development:
   • Pipeline Setup: We defined a pipeline for data preprocessing, feature engineering, and model training.
   • Multiple Models: Logistic Regression, Random Forest, and SVM models were tested.
   • Hyperparameter Tuning: GridSearchCV was used to find the best model and hyperparameters.
   • Model Evaluation: The best model was evaluated on the test set, and its performance was analyzed.
   • Inference Pipeline: The best model was saved and loaded into an inference pipeline, which can be deployed via an API for real-time predictions.
"""

# ========================================
# 1. Installing the Libraries
# ========================================
# (If you've already installed these libraries, you can skip this step.)
# !pip install pandas numpy scikit-learn joblib flask matplotlib seaborn

# ========================================
# 2. Importing the Necessary Libraries
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from joblib import dump

# ========================================
# 3. Data Preprocessing
# A. Data Collection
# ========================================
def get_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer(as_frame=True)
    df = pd.concat([data['data'], data['target']], axis=1)
    return df

# Load the Breast Cancer Wisconsin dataset
data = get_data()

# Display the first few rows
print("Dataset shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# ========================================
# B. Exploratory Data Analysis (EDA)
# ========================================
"""
Let's perform some data analysis to understand the distribution of the features 
and their relationships with the target variable.
"""

# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x=data['target'])
plt.title('Distribution of Target Variable (Malignant vs Benign)')
plt.show()

# ========================================
# C. Mutual information and feature importance:
# ========================================
"""
- mutual_info_classif: This function calculates the mutual information between each feature and the target, 
  which helps in identifying how much information each feature contributes to predicting the target.
- Visualization: The code creates a bar plot to visualize the importance of each feature based on mutual information scores.
"""

# Split the data into features and target
X = data.drop("target", axis=1)
y = data["target"]

def Show_Feature_Score(X, y):
    # Compute the mutual information scores
    mi_scores = mutual_info_classif(X, y, random_state=42)

    # Create a DataFrame to display the scores
    mi_scores_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Information': mi_scores
    })

    # Sort the DataFrame by mutual information scores
    mi_scores_df = mi_scores_df.sort_values(by='Mutual Information', ascending=False)

    # Visualize the mutual information scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Mutual Information', y='Feature', data=mi_scores_df, palette="viridis")
    plt.title('Mutual Information Scores for Each Feature')
    plt.show()

Show_Feature_Score(X, y)

# ========================================
# D. Feature Engineering
# ========================================
def add_combined_feature(X):
    X = X.copy()  # Ensure we're modifying a copy of the DataFrame
    
    # Example feature: combining two features
    X['Combined_radius_texture'] = X['mean radius'] * X['mean texture']
    
    return X

# ========================================
# 3. Model Development
# A. Build the Training Pipeline
# ========================================

# Define the feature engineering and preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('feature_engineering', FunctionTransformer(add_combined_feature)),
    ('scaler', StandardScaler())
])

# Define the models and their hyperparameters for GridSearchCV
models = [
    {
        'classifier': [LogisticRegression(max_iter=1000)],
        'classifier__C': [0.1, 1.0, 10]
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1.0, 10],
        'classifier__kernel': ['linear', 'rbf']
    }
]

# Updated pipeline with additional feature engineering and data transformation steps
training_pipeline = Pipeline(steps=[
    ('preprocessing', preprocessing_pipeline),
    ('classifier', LogisticRegression())  # Placeholder, will be replaced by GridSearchCV
])

# Split the data into training and testing sets
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"\nTraining set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# ========================================
# B. Hyperparameter Tuning and Model Selection
# ========================================

# Use GridSearchCV to find the best model and hyperparameters
print("\nStarting GridSearchCV...")
print("Testing 3 models: Logistic Regression, Random Forest, and SVC")
print("This may take a few minutes...\n")

grid_search = GridSearchCV(training_pipeline, models, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

# Best model
best_model = grid_search.best_estimator_

# Identify which model was selected
if 'LogisticRegression' in str(type(best_model.named_steps['classifier'])):
    print("Best model type: Logistic Regression")
elif 'RandomForest' in str(type(best_model.named_steps['classifier'])):
    print("Best model type: Random Forest Classifier")
elif 'SVC' in str(type(best_model.named_steps['classifier'])):
    print("Best model type: Support Vector Classifier")

# ========================================
# C. Evaluate the Best Model
# ========================================

def evaluate_model(model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model's performance
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Evaluate the best Model
print("\n" + "="*50)
print("EVALUATING THE BEST MODEL")
print("="*50)
evaluate_model(best_model, X_test, y_test)

# Save the best model and the preprocessing steps
dump(best_model, 'best_cancer_model_pipeline.joblib')
print("\nModel saved as 'best_cancer_model_pipeline.joblib'")

# ========================================
# Display top models from GridSearch
# ========================================
print("\n" + "="*50)
print("TOP 5 MODELS FROM GRID SEARCH")
print("="*50)

# Get results dataframe
results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values('mean_test_score', ascending=False)

# Display top 5 models
for i in range(min(5, len(results_df))):
    params = results_df.iloc[i]['params']
    score = results_df.iloc[i]['mean_test_score']
    std = results_df.iloc[i]['std_test_score']
    
    # Determine model type
    if 'LogisticRegression' in str(params['classifier']):
        model_type = "Logistic Regression"
    elif 'RandomForest' in str(params['classifier']):
        model_type = "Random Forest"
    elif 'SVC' in str(params['classifier']):
        model_type = "SVC"
    else:
        model_type = "Unknown"
    
    print(f"\nRank {i+1}: {model_type}")
    print(f"  Score: {score:.4f} (+/- {std:.4f})")
    
    # Print relevant parameters for each model type
    if model_type == "Logistic Regression":
        print(f"  C: {params.get('classifier__C', 'N/A')}")
    elif model_type == "Random Forest":
        print(f"  n_estimators: {params.get('classifier__n_estimators', 'N/A')}")
        print(f"  max_depth: {params.get('classifier__max_depth', 'N/A')}")
    elif model_type == "SVC":
        print(f"  C: {params.get('classifier__C', 'N/A')}")
        print(f"  kernel: {params.get('classifier__kernel', 'N/A')}")

print("\n" + "="*50)
print("PIPELINE EXECUTION COMPLETE!")
print("="*50)

"""
## Assignment: Create a new pipeline, with a different feature engineering, and different models:
1- Random Forest:
   - n_estimators: 50, 100, 200
   - max_depth = None, 10, 20

2- Support Vector Classifier:
   - C : 0.1, 1, 10
   - kernel: linear, rbf

COMPLETED: The models list above now includes RandomForest and SVC with the specified hyperparameters.
The GridSearchCV will automatically test all three models and select the best one.
"""
