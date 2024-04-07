import pickle
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()
# labels = np.sort(np.unique(y))
# y = np.array([np.where(labels == x) for x in y]).flatten()

# Use LabelEncoder to encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

gaussian_model = GaussianNB()
gaussian_model.fit(X, y_encoded)

with open("model.pkl", 'wb') as f:
    pickle.dump(gaussian_model, f)
