import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

predictions = model.predict(X)
print("Dự đoán của mô hình:", predictions)

probabilities = model.predict_proba(X)
print("Xác suất dự đoán:", probabilities)

accuracy = model.score(X, y)
print("Độ chính xác của mô hình:", accuracy)
