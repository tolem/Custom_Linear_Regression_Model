import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.column_stack(([1] * X.shape[0], X))

        beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
        if self.fit_intercept:
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            self.coefficient = beta

    def predict(self, X):
        return np.array(X @ self.coefficient + self.intercept)

    @staticmethod
    def r2_score(y, yhat):
        return 1 - sum((y - yhat) ** 2) / sum((y - y.mean()) ** 2)

    @staticmethod
    def rmse(y, yhat):
        return (sum((y - yhat) ** 2) / len(y)) ** 0.5


# df = pd.DataFrame({
#     'Capacity': [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9],
#     'Age': [11, 11, 9, 8, 7, 7, 6, 5, 5, 4],
#     'Cost/ton': [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]})
df = pd.DataFrame({
    'f1': [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87],
    'f2': [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3],
    'f3': [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2],
    'y': [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
})


def fitCustom():
    regCustom = CustomLinearRegression(fit_intercept=True)
    regCustom.fit(df[['f1', 'f2', 'f3']], df['y'])
    y_pred = regCustom.predict(df[['f1', 'f2', 'f3']])

    res = {'Intercept': regCustom.intercept,
           'Coefficient': regCustom.coefficient,
           'R2': regCustom.r2_score(df['y'], y_pred),
           'RMSE': regCustom.rmse(df['y'], y_pred)}
    return res


def fitSci():
    regSci = LinearRegression(fit_intercept=True)
    regSci.fit(df[['f1', 'f2', 'f3']], df['y'])
    y_pred = regSci.predict(df[['f1', 'f2', 'f3']])

    res = {'Intercept': regSci.intercept_,
           'Coefficient': regSci.coef_,
           'R2': r2_score(df['y'], y_pred),
           'RMSE': mean_squared_error(df['y'], y_pred) ** 0.5}
    return res


r1 = fitCustom()
r2 = fitSci()
diff = {}
for [k, v] in r1.items():
    diff[k] = v - r2[k]
print(diff)