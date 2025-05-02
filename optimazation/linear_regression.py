import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 (y = 3x + 2 + noise)
np.random.seed(0)
x = np.random.rand(100, 1)
y = 3 * x + 2 + np.random.randn(100, 1) * 0.1

# 파라미터 초기화
w = np.random.randn(1)
b = np.random.randn(1)
lr = 0.1  # 학습률

# 학습
losses = []
for epoch in range(100):
    y_pred = w * x + b
    error = y_pred - y

    grad_w = np.mean(error * x)
    grad_b = np.mean(error)

    w -= lr * grad_w
    b -= lr * grad_b

    loss = np.mean(error ** 2)
    losses.append(loss)

print(f"선형 회귀 최종 결과: w = {w[0]:.4f}, b = {b[0]:.4f}")

# 그래프
plt.subplot(1, 2, 1)
plt.scatter(x, y, label="Data")
plt.plot(x, w * x + b, color='red', label="Prediction")
plt.title("Linear Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# 데이터 생성 (0 또는 1 클래스를 나눔)
np.random.seed(1)
x_cls = np.linspace(-3, 3, 100).reshape(-1, 1)
y_cls = (x_cls > 0).astype(np.float32)  # 0보다 크면 1, 아니면 0
y_cls = np.where(np.random.rand(100, 1) < 0.1, 1 - y_cls, y_cls)  # 10% noise

# 파라미터 초기화
w_cls = np.random.randn(1)
b_cls = np.random.randn(1)
lr_cls = 0.1

# 시그모이드 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 학습
losses_cls = []
for epoch in range(100):
    z = w_cls * x_cls + b_cls
    y_pred_cls = sigmoid(z)

    error_cls = y_pred_cls - y_cls

    grad_w_cls = np.mean(error_cls * x_cls)
    grad_b_cls = np.mean(error_cls)

    w_cls -= lr_cls * grad_w_cls
    b_cls -= lr_cls * grad_b_cls

    loss_cls = -np.mean(y_cls * np.log(y_pred_cls + 1e-8) + (1 - y_cls) * np.log(1 - y_pred_cls + 1e-8))
    losses_cls.append(loss_cls)

print(f"로지스틱 회귀 최종 결과: w = {w_cls[0]:.4f}, b = {b_cls[0]:.4f}")

# 그래프
plt.subplot(1, 2, 2)
plt.scatter(x_cls, y_cls, label="Data")
plt.plot(x_cls, sigmoid(w_cls * x_cls + b_cls), color='red', label="Prediction")
plt.title("Logistic Regression")
plt.xlabel("x")
plt.ylabel("Probability")
plt.legend()

plt.tight_layout()
plt.show()
