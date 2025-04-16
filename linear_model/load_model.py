import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt

model = nn.Sequential(
    nn.Linear(1, 64),  # 첫 레이어
    nn.ReLU(),         # 비선형성
    nn.Linear(64, 64), # 두 번째 레이어
    nn.ReLU(),         # 비선형성
    nn.Linear(64, 1)   # 출력 레이어
)

model.load_state_dict(torch.load("linear_model/linear_model_2d.pth"))
model.eval()

print(model)
# 예측 예시
x_test = torch.tensor([[4.0]])
with torch.no_grad():
    y_pred = model(x_test)

print(f"불러온 모델 예측: x=4 → y={y_pred.item():.4f}")


def model_2d(tensor):
    # 예측
    
    model.eval()
    with torch.no_grad():
        predicted = model(tensor).item()
    print(f"\n예측: x=4 → y={predicted:.4f}")

test_input = torch.tensor([[4.0]])

model_2d(test_input)