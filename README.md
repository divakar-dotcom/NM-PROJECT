# NM-PROJECT
# Customer Churn Prediction using Machine Learning
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
# Note: Replace 'churn_data.csv' with your actual dataset path
try:
    df = pd.read_csv('churn_data.csv')
    print("Dataset loaded successfully!")
except:
    # If file not found, we'll use a synthetic dataset
    print("File not found. Creating synthetic dataset...")
    np.random.seed(42)
    n_samples = 1000
    data = {
        'customer_id': np.arange(n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'tenure': np.random.randint(1, 72, n_samples),
        'usage_frequency': np.random.randint(1, 100, n_samples),
        'support_calls': np.random.randint(0, 10, n_samples),
        'payment_delay': np.random.randint(0, 30, n_samples),
        'subscription_type': np.random.choice(['Basic', 'Standard', 'Premium'], n_samples),
        'contract_length': np.random.choice(['Monthly', 'Quarterly', 'Annual'], n_samples),
        'total_spend': np.random.uniform(50, 500, n_samples).round(2),
        'last_purchase': np.random.randint(1, 365, n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    df = pd.DataFrame(data)
    print("Synthetic dataset created for demonstration.")

# Exploratory Data Analysis (EDA)
print("\n=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types and missing values:")
print(df.info())
print("\nDescriptive statistics:")
print(df.describe(include='all'))

# Visualizations
plt.figure(figsize=(15, 10))

# Churn distribution
plt.subplot(2, 2, 1)
sns.countplot(x='churn', data=df)
plt.title('Churn Distribution')

# Age distribution by churn
plt.subplot(2, 2, 2)
sns.boxplot(x='churn', y='age', data=df)
plt.title('Age Distribution by Churn')

# Tenure distribution by churn
plt.subplot(2, 2, 3)
sns.boxplot(x='churn', y='tenure', data=df)
plt.title('Tenure Distribution by Churn')

# Payment delay by churn
plt.subplot(2, 2, 4)
sns.boxplot(x='churn', y='payment_delay', data=df)
plt.title('Payment Delay by Churn')

plt.tight_layout()
plt.show()

# Correlation matrix
numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Data Preprocessing
print("\n=== Data Preprocessing ===")

# Handle missing values if any
print("\nMissing values before handling:")
print(df.isnull().sum())

# For demonstration, we'll fill numeric missing values with median and categorical with mode
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

print("\nMissing values after handling:")
print(df.isnull().sum())

# Feature Engineering
print("\n=== Feature Engineering ===")

# Create new features if needed
df['avg_spend_per_usage'] = df['total_spend'] / df['usage_frequency']
df['support_per_usage'] = df['support_calls'] / df['usage_frequency']

# Encode categorical variables
label_encoders = {}
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col != 'customer_id':  # Exclude ID column
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

print("\nCategorical columns encoded:")
print(categorical_cols)

# Drop unnecessary columns
df = df.drop(['customer_id'], axis=1)  # ID doesn't help in prediction

# Handle class imbalance (SMOTE)
X = df.drop('churn', axis=1)
y = df['churn']

print("\nClass distribution before SMOTE:")
print(y.value_counts())

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

print("\nClass distribution after SMOTE:")
print(y_res.value_counts())

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing completed!")

# Model Training and Evaluation
print("\n=== Model Training and Evaluation ===")

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'KNN': KNeighborsClassifier()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Store results
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }
    
    # Print classification report
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

# Compare model performance
results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df.sort_values(by='ROC AUC', ascending=False))

# Hyperparameter Tuning for the best model
print("\n=== Hyperparameter Tuning for Random Forest ===")
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                          cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print("Best parameters found:")
print(grid_search.best_params_)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_scaled)
y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]

print("\nBest Random Forest Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")

# Feature Importance
print("\n=== Feature Importance ===")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Save the best model
import joblib
joblib.dump(best_rf, 'best_churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nBest model and scaler saved to disk!")
