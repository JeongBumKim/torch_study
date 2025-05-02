import numpy as np
import matplotlib.pyplot as plt

# 진짜 모델: y = 2.5 * exp(1.3 * x) + noise
np.random.seed(0)
true_a = 2.5
true_b = 1.3

x_data = np.linspace(0, 1, 20)
y_data = true_a * np.exp(true_b * x_data) + 0.2 * np.random.randn(20)
def model(x, a, b):
    return a * np.exp(b * x)

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(x, y, lr=0.1, epochs=100):
    a, b = 1.0, 0.0  # 초기값
    losses = []

    for _ in range(epochs):
        y_pred = model(x, a, b)
        error = y_pred - y

        grad_a = 2 * np.mean(error * np.exp(b * x))
        grad_b = 2 * np.mean(error * a * x * np.exp(b * x))

        a -= lr * grad_a
        b -= lr * grad_b

        losses.append(loss(y, model(x, a, b)))

    return a, b, losses
def gauss_newton(x, y, epochs=10):
    a, b = 1.0, 0.0  # 초기값
    losses = []

    for _ in range(epochs):
        y_pred = model(x, a, b)
        residual = y - y_pred

        J = np.zeros((len(x), 2))  # Jacobian
        J[:, 0] = np.exp(b * x)          # dy/da
        J[:, 1] = a * x * np.exp(b * x)  # dy/db

        delta = np.linalg.lstsq(J, residual, rcond=None)[0]

        a += delta[0]
        b += delta[1]

        losses.append(loss(y, model(x, a, b)))

    return a, b, losses
def levenberg_marquardt(x, y, epochs=20, lambda_init=0.01):
    a, b = 1.0, 0.0  # 초기값
    losses = []
    lambd = lambda_init

    for _ in range(epochs):
        y_pred = model(x, a, b)
        residual = y - y_pred

        J = np.zeros((len(x), 2))  # Jacobian
        J[:, 0] = np.exp(b * x)
        J[:, 1] = a * x * np.exp(b * x)

        H = J.T @ J  # approximate Hessian
        g = J.T @ residual  # gradient 방향

        # damping 추가
        H_lm = H + lambd * np.eye(2)

        delta = np.linalg.solve(H_lm, g)

        # 새로운 파라미터로 이동
        a_new = a + delta[0]
        b_new = b + delta[1]

        # 이동 결과 loss 비교
        new_loss = loss(y, model(x, a_new, b_new))
        old_loss = loss(y, model(x, a, b))

        if new_loss < old_loss:
            # 더 좋아졌으면 이동하고, damping 줄이기
            a, b = a_new, b_new
            lambd *= 0.7
        else:
            # 나빠졌으면 damping 키우기
            lambd *= 2.0

        losses.append(loss(y, model(x, a, b)))

    return a, b, losses

# 학습
a_gd, b_gd, losses_gd = gradient_descent(x_data, y_data, lr=0.5, epochs=100)
a_gn, b_gn, losses_gn = gauss_newton(x_data, y_data, epochs=10)
a_lm, b_lm, losses_lm = levenberg_marquardt(x_data, y_data, epochs=20)

print(f"경사 하강법 결과: a = {a_gd:.4f}, b = {b_gd:.4f}")
print(f"가우스-뉴턴 결과: a = {a_gn:.4f}, b = {b_gn:.4f}")
print(f"Levenberg-Marquardt 결과: a = {a_lm:.4f}, b = {b_lm:.4f}")

# 그래프
plt.plot(losses_gd, label="Gradient Descent")
plt.plot(np.linspace(0, 100, 10), losses_gn, label="Gauss-Newton")
plt.plot(np.linspace(0, 100, 20), losses_lm, label="Levenberg-Marquardt")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.show()
