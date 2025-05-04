import torch
from torch import nn
import torchvision.models as models
import matplotlib.pyplot as plt

def model_2d(tensor):
    model = nn.Sequential(
        nn.Linear(1, 64),  # 첫 레이어
        nn.ReLU(),         # 비선형성
        nn.Linear(64, 64), # 두 번째 레이어
        nn.ReLU(),         # 비선형성
        nn.Linear(64, 1)   # 출력 레이어
    )

    model.load_state_dict(torch.load("linear_model/linear_model_2d.pth"))
    model.eval()

    # 예측    
    with torch.no_grad():
        predicted = model(tensor).item()
    print(f"\n예측: x=4 → y={predicted:.4f}")

model_2d(torch.tensor([[4.0]]))

def model_1d(tensor):


    # 모델 선언 (학습할 때와 똑같이!)
    model = nn.Linear(1, 1)

    # state_dict 로드
    model.load_state_dict(torch.load("linear_model/linear_model.pth"))

    # 평가 모드
    model.eval()

    # 예측 테스트
    
    with torch.no_grad():
        predicted = model(tensor).item()

    print(f"예측: x=4 → y={predicted:.4f}")

model_1d(torch.tensor([[4.0]]))
