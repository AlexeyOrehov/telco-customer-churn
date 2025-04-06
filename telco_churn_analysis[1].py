# Telco Customer Churn Prediction

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import shap

# 1. Загрузка данных
df = pd.read_csv('telco_churn.csv')

# 2. Преобразование TotalCharges в число
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# 3. Кодирование бинарных признаков
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
               'TechSupport', 'StreamingTV', 'StreamingMovies']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 4. One-hot кодирование
df = pd.get_dummies(df, columns=['InternetService', 'PaymentMethod', 'Contract', 'MultipleLines'], drop_first=True)

# 5. Удаляем customerID
df.drop(['customerID'], axis=1, inplace=True)

# 6. Разделение на X и y
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 7. Модель XGBoost
model = XGBClassifier(n_estimators=100,
                      learning_rate=0.1,
                      max_depth=4,
                      scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
                      use_label_encoder=False,
                      eval_metric='logloss',
                      random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_prob))

# 8. Интерпретация SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
