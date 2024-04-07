import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
import pickle

# Load data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).values
y = df['Disease'].values

# # Encoding target variable (if necessary)
# labels = np.sort(np.unique(y))
# y = np.array([np.where(labels == x) for x in y]).flatten()
#
# # Define hyperparameters grid
# param_grid = {
#     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Solver for optimization
#     'max_iter': [100, 200, 300]  # Maximum number of iterations
# }

# Initialize logistic regression model
rf = AdaBoostClassifier()

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Perform grid search
rf.fit(X, y)

# # Get best hyperparameters and model
# best_params = grid_search.best_params_
# best_model = grid_search.best_estimator_

# Save the best model
with open("model.pkl", 'wb') as f:
    pickle.dump(rf, f)

# print("Best Hyperparameters:", best_params)
# print("Best Accuracy:", grid_search.best_score_)
