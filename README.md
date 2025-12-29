# Earnings Regression Analysis

## 🎯 Project Goals
- Estimate years of schooling (`S`) contribution to earnings (`EARNINGS`)
- Test model improvement with experience (`EXP`)
- Regression quality diagnostics via residuals
- Predict wages for typical case (bachelor + 5 years experience)

## 📊 Key Results
| Model     | R² (train) | S Coefficient | EXP Coefficient | Intercept |
|-----------|------------|---------------|-----------------|-----------|
| S only    | [0.XX]     | $XX.XX/year   | —               | $XX.XX    |
| S + EXP   | [0.XX]     | $XX.XX/year   | $XX.XX/year     | $XX.XX    |[web:27]

**Prediction:** Bachelor's (16 years schooling) + 5 years EXP = **$XX.XX/hour**

## 📈 Visualizations

![Residuals Distribution](Screenshot%202025-12-29%20at%2015.53.18.png)
![Residuals vs Fitted](Screenshot%202025-12-29%20at%2015.53.22.png)
![Multivariable Residuals](Screenshot%202025-12-29%20at%2015.53.26.png)
![Multivariable Scatter](Screenshot%202025-12-29%20at%2015.53.29.png)


## 🛠️ Tech Stack

Python 3.9+ | pandas | scikit-learn | seaborn | matplotlib
