import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r'F:/Documents/Universidad Politecnica de Yucatan/Machine Learning/Unit 2/Indicadores_municipales_sabana_DA (1).csv', encoding='latin1')
df = pd.get_dummies(df, columns=['gdo_rezsoc00', 'gdo_rezsoc05', 'gdo_rezsoc10'])
df = df.drop(columns=['ent', 'nom_ent', 'mun', 'clave_mun', 'nom_mun'])
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
df = df.fillna(df.mean())

X = df.drop(columns=['pobreza_patrim_10'])
y = df['pobreza_patrim_10']

X_train = X[:1965]
y_train = y[:1965]
X_test = X[1965:]
y_test = y[1965:]

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse_test = mean_squared_error(y_test, y_pred)

print(f"Coefficients (w): {model.coef_}")
print(f"Intercept (b): {model.intercept_}")
print(f"MSE on the test set: {mse_test:.2f}")
