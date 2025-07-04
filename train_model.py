import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle


data = {
    'Age': [25, 45, 35, 50, 23, 43, 36, 29],
    'Income': [50000, 64000, 58000, 72000, 45000, 69000, 60000, 52000],
    'LoanAmount': [20000, 25000, 23000, 27000, 18000, 26000, 24000, 21000],
    'Default': [0, 1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[['Age', 'Income', 'LoanAmount']]
y = df['Default']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = LogisticRegression()
model.fit(X_scaled, y)


pickle.dump(model, open("loan_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Done! Model and scaler saved as 'loan_model.pkl' and 'scaler.pkl'.")
