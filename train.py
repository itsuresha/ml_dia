import pandas as pd

from sklearn.svm import SVC
import pickle

# Load dataset​

df = pd.read_csv('diabetes.csv')

# Define features and target​

X = df.iloc[:,:-1]

y = df.iloc[:,-1]

# Train the model​

model = SVC()

model.fit(X, y)

# Save the model​

with open("model.pkl", "wb") as f:

    pickle.dump(model, f)