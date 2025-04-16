import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ✅ 데이터 전처리 설정 (이미지 크기 맞춤 + 텐서 변환)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ✅ 데이터셋 로드 (ImageFolder 사용 시 폴더명이 라벨로 자동 매핑)
dataset = datasets.ImageFolder("image_model/dataset/", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# ✅ 클래스 이름 출력
print("클래스 라벨:", dataset.classes)

# ✅ 간단한 CNN 모델 정의
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 입력 채널 3(RGB), 출력 16
    nn.ReLU(),
    nn.MaxPool2d(2),  # 32x32
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),  # 16x16
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 64),
    nn.ReLU(),
    nn.Linear(64, len(dataset.classes))  # 출력 클래스 수에 맞춤
)

# ✅ 손실 함수 & 옵티마이저
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ✅ 학습 루프
loss_history = []
for epoch in range(20):
    total_loss = 0
    for images, labels in dataloader:
        pred = model(images)
        loss = loss_fn(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# ✅ 학습 손실 시각화
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 모델 저장
os.makedirs("saved_model", exist_ok=True)
torch.save(model.state_dict(), "saved_model/sirbot_classifier.pth")
print("✅ 모델 저장 완료!")
