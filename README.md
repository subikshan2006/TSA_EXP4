# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 19-05-2026
### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
data = pd.read_csv('chennai_temperature_10years.csv')

# Display columns
print("Columns in dataset:", data.columns)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Use Temperature column
X = data['Temperature']

# Make data stationary
X = X.diff().dropna()

N = 1000

plt.rcParams['figure.figsize'] = [12, 6]

# Plot stationary data
plt.plot(X)
plt.title('Stationary Temperature Data')
plt.xlabel('Index')
plt.ylabel('Temperature Difference')
plt.grid(True)
plt.show()

# ACF and PACF plots
plt.subplot(2,1,1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('ACF')

plt.subplot(2,1,2)
plot_pacf(X, lags=40, ax=plt.gca())
plt.title('PACF')

plt.tight_layout()
plt.show()

# ---------------- ARMA(1,1) ---------------- #

arma11_model = ARIMA(X, order=(1,0,1)).fit()

phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])

ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

# Plot simulated ARMA(1,1)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1)')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# ACF and PACF for ARMA(1,1)
plot_acf(ARMA_1)
plt.title('ACF of Simulated ARMA(1,1)')
plt.show()

plot_pacf(ARMA_1)
plt.title('PACF of Simulated ARMA(1,1)')
plt.show()

# ---------------- ARMA(2,2) ---------------- #

arma22_model = ARIMA(X, order=(2,0,2)).fit()

phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']

theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']

# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N)

# Plot simulated ARMA(2,2)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2)')
plt.xlabel('Samples')
plt.ylabel('Values')
plt.grid(True)
plt.show()

# ACF and PACF for ARMA(2,2)
plot_acf(ARMA_2)
plt.title('ACF of Simulated ARMA(2,2)')
plt.show()

plot_pacf(ARMA_2)
plt.title('PACF of Simulated ARMA(2,2)')
plt.show()
```
## OUTPUT:
#### DATA:
<img width="1012" height="547" alt="image" src="https://github.com/user-attachments/assets/dbec694e-d5a9-4ce2-95ec-54a9b62cc6b9" />

#### ACF:
<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/ce39afa8-88eb-4e45-a77e-1330fa03a991" />


#### PACF:

<img width="1198" height="590" alt="image" src="https://github.com/user-attachments/assets/2121edab-5381-43dc-98a8-76a966119d93" />


#### SIMULATED ARMA(1,1) PROCESS:


<img width="999" height="547" alt="image" src="https://github.com/user-attachments/assets/7690d9bb-0418-430a-a39e-4049d490ccea" />


#### Partial Autocorrelation
<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/9beb6a69-0533-43ab-a488-80d10837c108" />


#### Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/83a7b562-969f-4f5f-9053-2cfb1565f24e" />



#### SIMULATED ARMA(2,2) PROCESS:

<img width="999" height="547" alt="image" src="https://github.com/user-attachments/assets/fc033363-57b2-4192-9ed2-94a27eac7eaf" />

#### Partial Autocorrelation


<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/99950db7-9f7d-4aea-a178-9bfdf103237c" />


#### Autocorrelation

<img width="1002" height="528" alt="image" src="https://github.com/user-attachments/assets/bd5cb401-d9c2-4506-b14f-629801d58b70" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
