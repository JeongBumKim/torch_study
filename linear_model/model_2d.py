import torch
from torch import nn
import matplotlib.pyplot as plt

# 데이터 생성
X = torch.tensor([[float(i)] for i in range(1, 5)])
y = 2 * X**2 + 2 * X + 1

# 모델 (MLP)
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 손실 함수, 옵티마이저
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습
loss_list = []
for epoch in range(2000):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    loss_list.append(loss.item())

# 예측
test_input = torch.tensor([[4.0]])
model.eval()
with torch.no_grad():
    predicted = model(test_input).item()
print(f"\n예측: x=4 → y={predicted:.4f}")

torch.save(model.state_dict(), "linear_model/linear_model_2d.pth")


# 시각화
plt.plot(loss_list)
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
