import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt
# 1. 학습 데이터 정의
values = []

for i in range(1, 5):  # 1부터 20까지
    values.append([float(i)])  # 2D로 만들기 위해 [i]

X = torch.tensor(values)

y = 2 * X + 1

# 2. 모델 정의: 단순 선형 회귀 (Linear 1층)
model = nn.Linear(1, 1)

# 3. 손실 함수 및 옵티마이저 정의
loss_fn = nn.MSELoss()  # 평균제곱오차
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_list = []
# 4. 학습 루프
for epoch in range(2000):
    # ▶️ 순전파
    pred = model(X)

    # 📉 손실 계산
    loss = loss_fn(pred, y)

    # 🔁 역전파 및 최적화
    optimizer.zero_grad()     # 이전 gradient 초기화
    loss.backward()           # 역전파: gradient 계산
    optimizer.step()          # 파라미터 업데이트
    if epoch % 100 == 0:
        print(round(loss.item(),5))
    loss_list.append(loss.item())

# 5. 최종 예측 결과 보기
test_input = torch.tensor([[4.0]])
predicted = model(test_input).item()
print(f"\n예측: x=4 → y={predicted:.4f}")

torch.save(model.state_dict(), "linear_model/linear_model.pth")

epochs = list(range(len(loss_list)))
# 그래프 그리기
plt.plot(epochs, loss_list, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()