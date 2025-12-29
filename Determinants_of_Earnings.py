import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

pd.options.display.float_format = '{:,.2f}'.format

df_data = pd.read_csv('NLSY97_subset.csv')

# Data Cleaning

df_data.isna().values.any()
df_data.duplicated().values.any()

df_data = df_data.dropna()
df_data = df_data.drop_duplicates()

print(df_data.describe())

# Split Training & Test Dataset

target = df_data['EARNINGS']
features = df_data[['S']]

X_train, X_test, y_train, y_test = train_test_split(features,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=10)

# Evaluating the Coefficients of the Model

regr = LinearRegression()
regr.fit(X_train, y_train)
rsquared = regr.score(X_train, y_train)

regr = LinearRegression()
regr.fit(X_train, y_train)

coef_schooling = regr.coef_[0]
intercept = regr.intercept_

print("Intercept:", intercept)
print("Coefficient for S:", coef_schooling)
print(f"Extra dollars per additional year of schooling: {coef_schooling:.2f}")

# Analysing the Estimated Values & Regression Residuals

predicted_values = regr.predict(X_train)
residuals = (y_train - predicted_values)

plt.figure(figsize=(6, 4))
sns.histplot(residuals, kde=True, color='steelblue')
plt.xlabel('Residual')
plt.title('Distribution of residuals (train)')
plt.tight_layout()
plt.show()

# Residuals vs fitted values
plt.figure(figsize=(6, 4))
plt.scatter(predicted_values, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted values (ŷ)')
plt.ylabel('Residuals')
plt.title('Residuals vs fitted values (train)')
plt.tight_layout()
plt.show()

# Multivariable Regression

X = df_data[['S', 'EXP']]
y = df_data['EARNINGS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=10
)

regr2 = LinearRegression()
regr2.fit(X_train, y_train)

r2_train_2 = regr2.score(X_train, y_train)
print("R-squared on training data:", r2_train_2)

# Evaluating the Coefficients of the Model

coef_school, coef_exp = regr2.coef_
intercept = regr2.intercept_

print("Intercept:", intercept)
print("Coefficient for S (schooling):", coef_school)
print("Coefficient for EXP (experience):", coef_exp)

# Analysing the Estimated Values & Regression Residuals

y_pred_train = regr2.predict(X_train)
residuals2 = y_train - y_pred_train

plt.figure(figsize=(6, 4))
sns.histplot(residuals2, kde=True, color='steelblue')
plt.xlabel('Residual')
plt.title('Residuals distribution (train)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.scatter(y_pred_train, residuals2, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted earnings (ŷ)')
plt.ylabel('Residuals')
plt.title('Residuals vs fitted values (train)')
plt.tight_layout()
plt.show()


# Make your own predictions with the model by changing the variables below
S_bachelor = 16
EXP_bachelor = 5

X_new = np.array([[S_bachelor, EXP_bachelor]])
predicted_earnings = regr2.predict(X_new)[0]
print(f"Expected hourly earnings: ${predicted_earnings:.2f}")