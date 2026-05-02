# Ex.No: 1B                     CONVERSION OF NON STATIONARY TO STATIONARY DATA
# Date: 22/04/2026
# Name: Raghu Ram VR
# Reg.No: 212224220075
# AIM:
To perform regular differncing,seasonal adjustment and log transformation on international airline passenger data
# REQUIREMENTS:
1.Dataset-AirlinePassengers

2.Google Colab
# ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.
# PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
data = pd.read_csv('marketing_campaign_performance_10000.csv')
date_col = None
for col in data.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        date_col = col
        break

if date_col is None:
    print("No date column found. Creating artificial date index...")
    data['Date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
    date_col = 'Date'

print("Using date column:", date_col)

data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
data = data.dropna(subset=[date_col])
data.set_index(date_col, inplace=True)
data = data.sort_index()
numeric_cols = data.select_dtypes(include=[np.number]).columns

if len(numeric_cols) == 0:
    raise ValueError("No numeric column found for analysis.")

value_col = numeric_cols[0]  
print("Using value column:", value_col)
data['diff'] = data[value_col].diff()

try:
    result = seasonal_decompose(data[value_col].dropna(), model='additive', period=12)
    data['seasonal_adjusted'] = result.resid
except:
    data['seasonal_adjusted'] = np.nan
    print("Seasonal decomposition failed (maybe not enough data).")
data['log'] = np.log(data[value_col].replace(0, np.nan))
data['log_diff'] = data['log'].diff()

try:
    result_log = seasonal_decompose(data['log_diff'].dropna(), model='additive', period=12)
    data['log_seasonal_diff'] = result_log.resid
except:
    data['log_seasonal_diff'] = np.nan
    print("Log seasonal decomposition failed.")


plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data[value_col])
plt.title('Original Data')

plt.subplot(6, 1, 2)
plt.plot(data['diff'])
plt.title('Differencing')

plt.subplot(6, 1, 3)
plt.plot(data['seasonal_adjusted'])
plt.title('Seasonal Adjustment')

plt.subplot(6, 1, 4)
plt.plot(data['log'])
plt.title('Log Transformation')

plt.subplot(6, 1, 5)
plt.plot(data['log_diff'])
plt.title('Log + Differencing')

plt.subplot(6, 1, 6)
plt.plot(data['log_seasonal_diff'])
plt.title('Final Stationary Series')

plt.tight_layout()
plt.show()

data[[value_col, 'log_seasonal_diff']].plot(figsize=(12, 6))
plt.title("Original vs Final Transformed")
plt.show()
```
# OUTPUT:
<img width="1589" height="1330" alt="download (1)" src="https://github.com/user-attachments/assets/7e1a6ed5-ad01-4c46-adad-c89597e93e4e" />

<img width="947" height="481" alt="image" src="https://github.com/user-attachments/assets/596db873-fc5c-4cfc-abd5-ace376318a5a" />


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on international airline passenger
data.
