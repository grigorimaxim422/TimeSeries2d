import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("BTC-Daily.csv")
df.head()
df.describe()

print("\nRange of Dates Covered:")
print("Start Date:", df['date'].min())
print("End Date:", df['date'].max())

# Sort DataFrame by date
df['date'] = pd.to_datetime(df['date'])  # Convert date column to datetime
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
df = df.sort_values('date')
print("df=")
df

print("\nClosing Prices Statistics:")
print(df['close'].describe())

missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

# Check for anomalous data i.e negative prices
anomalous_data = df[df['close'] < 0]
print("\nAnomalous Data (Negative Prices):")
print(anomalous_data)

###[9]
df.dropna(inplace=True)
df = df[df['close'] >= 0]

# Normalize or standardize the prices (min max)
df['normalized_close'] = (df['close'] - df['close'].min()) / (df['close'].max() - df['close'].min())

print("\nDataFrame after handling missing and anomalous data:")
print(df.head())

###[10]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
###[14]
# Compute the differences between consecutive closing prices to remove trends
df['close_diff'] = df['close'].diff()
print("df['close_diff']=")
print(df['close_diff'])
##[15]
df['normalized_close_diff'] = (df['close_diff'] - df['close_diff'].min()) / (df['close_diff'].max() - df['close_diff'].min())

print("---------------------\ndf=")
print(df)
# ##[16]
# Split the dataset into training and testing sets
X_diff = df[['date']].diff().dropna()  # Use differenced days_since_start
y_diff = df['normalized_close_diff'].iloc[1:]     # Skip the first row due to differencing

print(f"X_diff={X_diff}")
print(f"y_diff={y_diff}")

X_train_diff, X_test_diff, y_train_diff, y_test_diff = train_test_split(X_diff, y_diff, test_size=0.2, random_state=42)
print(f"X_train_diff={X_train_diff}")
print(f"y_train_diff={y_train_diff}")
# ##[17]
model_diff = LinearRegression()
model_diff.fit(X_train_diff, y_train_diff)

# # Make predictions
y_pred_diff = model_diff.predict(X_test_diff)

# # Undo normalization to get predictions in the original scale
y_pred_original_scale = y_pred_diff * (df['close_diff'].max() - df['close_diff'].min()) + df['close_diff'].min()

# ##[21]
from sklearn.metrics import mean_absolute_error, r2_score

mae_diff = mean_absolute_error(y_test_diff, y_pred_diff)
rmse_diff = np.sqrt(mean_squared_error(y_test_diff, y_pred_diff))
r2_diff = r2_score(y_test_diff, y_pred_diff)

print("Mean Absolute Error after removing trends:", mae_diff)
print("Root Mean Squared Error after removing trends:", rmse_diff)
print("R-squared (R2) score after removing trends:", r2_diff)
print("The MSE was 105099748.37549701 If I hadn't removed trends")
# ##[22]
# from sklearn.model_selection import cross_val_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error

# # polynomial features to try
# degrees = [1, 2, 3, 4, 5]
# cv_scores = {}

# for degree in degrees:
#     model_cv = make_pipeline(PolynomialFeatures(degree), LinearRegression())

#     scores = cross_val_score(model_cv, X_diff, y_diff, cv=5, scoring='neg_mean_squared_error')
    
#     cv_scores[degree] = -scores.mean()
    
# ###Why I Chose this model
# best_degree = min(cv_scores, key=cv_scores.get)
# print("Cross-validation scores for different degrees:")
# for degree, score in cv_scores.items():
#     print("Degree:", degree, "- MSE:", score)

# print("\nBest degree based on cross-validation:", best_degree)
# ##[26]
# model_poly_cv = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
# model_poly_cv.fit(X_train_diff, y_train_diff)
# ##[27]
# y_pred_poly_cv = model_poly_cv.predict(X_test_diff)

# mse_poly_cv = mean_squared_error(y_test_diff, y_pred_poly_cv)
# mae_poly_cv = mean_absolute_error(y_test_diff, y_pred_poly_cv)
# rmse_poly_cv = np.sqrt(mse_poly_cv)
# r2_poly_cv = r2_score(y_test_diff, y_pred_poly_cv)

# print("\nMean Squared Error (Polynomial Regression with CV):", mse_poly_cv)
# print("Mean Absolute Error (Polynomial Regression with CV):", mae_poly_cv)
# print("Root Mean Squared Error (Polynomial Regression with CV):", rmse_poly_cv)
# print("R-squared (R2) score (Polynomial Regression with CV):", r2_poly_cv)

# ##[29]
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# from statsmodels.tsa.arima.model import ARIMA

# #LOADING THE DATA AGAIN
# # Load the data
# df = pd.read_csv("BTC-Daily.csv")
# df['date'] = pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)

# ##[32]
# # Test for stationarity
# def test_stationarity(timeseries):
#     # Perform Dickey-Fuller test
#     result = adfuller(timeseries)
#     print('ADF Statistic:', result[0])
#     print('p-value:', result[1])
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print('\t%s: %.3f' % (key, value))
#         # Plot rolling statistics
#     rolmean = timeseries.rolling(window=12).mean()
#     rolstd = timeseries.rolling(window=12).std()
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(timeseries, color='blue', label='Original')
#     plt.plot(rolmean, color='red', label='Rolling Mean')
#     plt.plot(rolstd, color='black', label='Rolling Std')
#     plt.legend()
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show()

# test_stationarity(df['close'])
# ##[33]
# def test_stationarity(timeseries):
#     # Perform Dickey-Fuller test
#     result = adfuller(timeseries)
#     print('ADF Statistic:', result[0])
#     print('p-value:', result[1])
#     print('Critical Values:')
#     for key, value in result[4].items():
#         print('\t%s: %.3f' % (key, value))
    
#     # Plot rolling statistics
#     rolmean = timeseries.rolling(window=12).mean()
#     rolstd = timeseries.rolling(window=12).std()
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(timeseries, color='blue', label='Original')
#     plt.plot(rolmean, color='red', label='Rolling Mean')
#     plt.plot(rolstd, color='black', label='Rolling Std')
#     plt.legend()
#     plt.title('Rolling Mean & Standard Deviation')
#     plt.show()

# test_stationarity(df['close'])
# ##[35]
# d = 1  # first-order differencing
# # Sort the DataFrame by date index in ascending order
# df.sort_index(inplace=True)
# # Specify the frequency of the time series data as daily ('D')
# df.index.freq = 'D'

# # Differencing
# df['close_diff'] = df['close'].diff(d).dropna()

# # Determine the order of the ARIMA model by analyzing ACF and PACF plots
# plot_acf(df['close_diff'], lags=20)
# plt.title('Autocorrelation Function (ACF)')
# plt.show()

# plot_pacf(df['close_diff'], lags=20)
# plt.title('Partial Autocorrelation Function (PACF)')
# plt.show()

# # Fit the ARIMA model
# p = 1  # Order of the AR model
# q = 1  # Order of the MA model

# model = ARIMA(df['close'], order=(p, d, q))
# model_fit = model.fit()

# # Interpret the model summary
# print(model_fit.summary())
# ##[41]
# #DOES NOT WORK


# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import numpy as np

# # Performance Metrics
# def calculate_metrics(actual, predicted):
#     rmse = np.sqrt(mean_squared_error(actual, predicted))
#     mae = mean_absolute_error(actual, predicted)
#     mape = np.mean(np.abs((actual - predicted) / actual)) * 100
#     return rmse, mae, mape

# def time_series_cross_validation(model, data, train_size, test_size):
#     predictions = []
#     for t in range(train_size, len(data) - test_size + 1, test_size):
#         train = data[:t]
#         test = data[t:t+test_size]
#         model_fit = model.fit(train)
#         forecast = model_fit.forecast(steps=test_size)[0]
#         predictions.extend(forecast)
#     return predictions
# # Residual Analysis
# def residual_analysis(model_fit):
#     residuals = model_fit.resid
#     plt.figure(figsize=(10, 4))
#     plt.plot(residuals)
#     plt.title('Residuals Plot')
#     plt.xlabel('Time')
#     plt.ylabel('Residuals')
#     plt.show()

# # Performance Metrics
# actual_values = df['close'].values
# predicted_values = model_fit.fittedvalues
# rmse, mae, mape = calculate_metrics(actual_values, predicted_values)
# print("RMSE:", rmse)
# print("MAE:", mae)
# print("MAPE:", mape)

# # Cross-Validation
# train_size = int(len(df) * 0.8)  # Adjust this percentage as needed
# test_size = len(df) - train_size
# predictions = time_series_cross_validation(model, df['close'], train_size, test_size)
# # Ensure that the lengths of actual values and predictions match
# actual_values_cv = df['close'].values[train_size:]
# cv_rmse, cv_mae, cv_mape = calculate_metrics(actual_values_cv, predictions)
# print("Cross-Validation RMSE:", cv_rmse)
# print("Cross-Validation MAE:", cv_mae)
# print("Cross-Validation MAPE:", cv_mape)

# # Residual Analysis
# residual_analysis(model_fit)


