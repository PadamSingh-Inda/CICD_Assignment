import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.metrics import accuracy_score
import pickle

# Load data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).values
y = df['Disease'].values

# # Split data into train and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train AdaBoost model
adaboost_model = AdaBoostClassifier(n_estimators=10, random_state=42)
adaboost_model.fit(X, y)

# # Evaluate model
# y_pred = adaboost_model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

# Save the model
with open("model.pkl", 'wb') as f:
    pickle.dump(adaboost_model, f)
