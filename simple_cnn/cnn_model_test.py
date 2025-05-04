import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ✅ 클래스 수 설정 (학습 때 사용한 것과 동일)
num_classes = 2  # <-- 바꾸세요!

# ✅ SimpleCNN 정의 (학습한 모델과 동일 구조여야 함)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ✅ 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes).to(device)
model.load_state_dict(torch.load("simple_cnn.pth", map_location=device))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ✅ 테스트 이미지 하나 예시
image_path = "plane.jpg"  # ← 여기 경로 수정하세요
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)  # (1, C, H, W)

# ✅ 예측
with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

print(f"✅ 예측 클래스 ID: {predicted_class}")
