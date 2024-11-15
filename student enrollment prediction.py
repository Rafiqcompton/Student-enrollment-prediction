# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate Student Enrollment Dataset (or load your dataset here)
np.random.seed(42)
n_samples = 1000

data = {
    'age': np.random.randint(18, 35, n_samples),
    'previous_gpa': np.random.uniform(2.0, 4.0, n_samples),
    'family_income': np.random.randint(20000, 200000, n_samples),
    'parents_education': np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], n_samples),
    'extracurricular_activities': np.random.randint(0, 10, n_samples),
    'standardized_test_score': np.random.uniform(1000, 1600, n_samples),
    'enrollment_status': np.random.choice(['Enrolled', 'Not Enrolled'], n_samples, p=[0.7, 0.3])
}

df = pd.DataFrame(data)

# Preprocessing
le = LabelEncoder()
df['parents_education_encoded'] = le.fit_transform(df['parents_education'])
df['enrollment_status_encoded'] = le.fit_transform(df['enrollment_status'])

# Feature Selection
features = ['age', 'previous_gpa', 'family_income', 'parents_education_encoded', 
            'extracurricular_activities', 'standardized_test_score']
X = df[features]
y = df['enrollment_status_encoded']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Model Evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Visualization
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Student Enrollment Prediction')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()
