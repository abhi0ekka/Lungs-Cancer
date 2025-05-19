import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime

# 1. Load the dataset
df = pd.read_csv('Dataset/lung_cancer_mortality_data_test_v2.csv')

# 2. Convert date columns
date_cols = ['diagnosis_date', 'beginning_of_treatment_date', 'end_treatment_date']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

# 3. Feature engineering: treatment duration
df['treatment_duration_days'] = (df['end_treatment_date'] - df['beginning_of_treatment_date']).dt.days

# 4. Drop ID column (not useful)
df.drop(columns=['id'], inplace=True)

# 5. Handle categorical variables
categorical_cols = ['gender', 'country', 'cancer_stage', 'family_history', 'smoking_status', 'treatment_type']
df[categorical_cols] = df[categorical_cols].astype(str)
df = pd.get_dummies(df, columns=categorical_cols)

# 6. Drop date columns after extraction
df.drop(columns=date_cols, inplace=True)

# 7. Handle missing or infinite values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# 8. Prepare features and labels
X = df.drop(columns=['survived'])
y = df['survived']

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 10. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 12. Histogram of Age
plt.figure(figsize=(8, 5))
sns.histplot(pd.read_csv('Dataset/lung_cancer_mortality_data_test_v2.csv')['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 13. Family History of Cancer Pie Chart
family_counts = pd.read_csv('Dataset/lung_cancer_mortality_data_test_v2.csv')['family_history'].value_counts()
labels = ['Yes', 'No'] if set(family_counts.index.str.lower()) == {'yes', 'no'} else family_counts.index

plt.figure(figsize=(6, 6))
plt.pie(family_counts, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
plt.title('Family History of Cancer (Yes/No)')
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()