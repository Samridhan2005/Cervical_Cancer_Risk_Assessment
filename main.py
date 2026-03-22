import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Create static folder if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# 1. Data Loading & Cleaning
df = pd.read_csv('cervical_cancer.csv')
df = df.replace('?', np.nan).apply(pd.to_numeric)
df = df.fillna(df.median())
df = df.drop(['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1)

# 2. Splitting
X = df.drop(['Biopsy', 'Hinselmann', 'Schiller', 'Citology'], axis=1) 
y = df['Biopsy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Model Training (High Sensitivity)
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Save Charts for UI
# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Model Performance (Confusion Matrix)')
plt.savefig('static/confusion_matrix.png')
plt.close()

# Feature Importance
importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
importances = importances.sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(6,4))
sns.barplot(x='Importance', y='Feature', data=importances)
plt.title('Top 10 Risk Factors')
plt.tight_layout()
plt.savefig('static/feature_importance.png')
plt.close()

# 5. Save Model
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')
print("Model and Charts saved successfully!")