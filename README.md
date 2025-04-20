import pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
df = pd.DataFrame({'X1': range(1, 11), 'X2': range(2, 12), 'Y': [1.1, 1.9, 3.1, 4.2, 5.1, 6.2, 7.1, 8.0, 9.0,
10.1]})
X, y = df[['X1', 'X2']], df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=0)
m = LinearRegression().fit(X_train, y_train)
yp = m.predict(X_test)
print(f"Coefs: {m.coef_}\nIntercept: {m.intercept_}\nMSE: {mean_squared_error(y_test,
yp)}\nR²: {r2_score(y_test, yp)}")
plt.figure(figsize=(10, 4))
for i, col in enumerate(['X1', 'X2'], 1):
    plt.subplot(1, 2, i); plt.scatter(df[col], y); plt.plot(df[col], m.predict(X), 'r')
plt.title(f'{col} vs Y'); plt.xlabel(col); plt.ylabel('Y')
plt.tight_layout(); plt.show()
