import pickle
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Read training data
df = pd.read_csv("data/train.csv")
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()

# Use LabelEncoder to encode categorical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train model
gaussian_model = GaussianNB()
gaussian_model.fit(X, y_encoded)

# save model
with open("model.pkl", 'wb') as f:
    pickle.dump(gaussian_model, f)
