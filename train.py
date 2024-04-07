import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Load data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).values
y = df['Disease'].values

# Define hyperparameters
hyperparameters = {
    'C': 0.1,  # Regularization parameter
    'solver': 'liblinear',  # Solver for optimization
    'max_iter': 100  # Maximum number of iterations
}

# Initialize logistic regression model with hyperparameters
logistic_model = LogisticRegression(**hyperparameters)

# Train the model
logistic_model.fit(X, y)

# Save the best model
with open("model.pkl", 'wb') as f:
    pickle.dump(logistic_model, f)
