import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

# 데이터 생성
np.random.seed(0)
n_samples = 100
n_outliers = 20

# 정상 데이터 (inliers)
x = np.linspace(0, 10, n_samples).reshape(-1, 1)
y = 3 * x.squeeze() + 2 + np.random.randn(n_samples) * 2

# 이상치 추가
x_outliers = np.random.uniform(0, 10, n_outliers).reshape(-1, 1)
y_outliers = np.random.uniform(-30, 30, n_outliers)

# 데이터 합치기
x_all = np.vstack((x, x_outliers))
y_all = np.hstack((y, y_outliers))
linear_regressor = LinearRegression()
linear_regressor.fit(x_all, y_all)

y_pred_linear = linear_regressor.predict(x_all)
ransac = RANSACRegressor(LinearRegression(), 
                         min_samples=50, 
                         residual_threshold=5.0, 
                         max_trials=100)
ransac.fit(x_all, y_all)

y_pred_ransac = ransac.predict(x_all)
plt.figure(figsize=(10, 5))

# 데이터 표시
plt.scatter(x_all, y_all, color='yellowgreen', marker='.', label='All Data')

# 선형 회귀 결과
plt.plot(x_all, y_pred_linear, color='red', label='Linear Regression')

# RANSAC 결과
plt.plot(x_all, y_pred_ransac, color='blue', label='RANSAC')

plt.legend()
plt.title("Linear Regression vs RANSAC")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
