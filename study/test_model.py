import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from torchvision import transforms

# 학습에 사용할 CPU나 GPU, MPS 장치를 얻습니다.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# 공개 데이터셋에서 학습 데이터를 내려받습니다.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# 공개 데이터셋에서 테스트 데이터를 내려받습니다.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# 모델을 정의합니다.
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')




# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 저장된 모델 불러오기
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Grayscale(),             # 흑백으로 변환
    transforms.Resize((28, 28)),        # 크기 조정
    transforms.ToTensor(),              # 텐서 변환 (0~1 정규화)
])

# 이미지 불러오기
image = Image.open("boots2.jpeg")   # 여기에 사용할 이미지 파일 경로
image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 + 디바이스 이동

# 예측하기
with torch.no_grad():
    output = model(image)
    predicted = torch.argmax(output, 1)

# FashionMNIST 클래스 라벨
labels = [
    "T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

print(f"예측 결과: {labels[predicted.item()]}")

import torch.nn.functional as F

# 예측 결과 가져오기
with torch.no_grad():
    output = model(image)

# 소프트맥스 적용 (확률 값으로 변환)
probabilities = F.softmax(output, dim=1)

# FashionMNIST 클래스 라벨
labels = [
    "T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

# 클래스별 점수 출력
for i, label in enumerate(labels):
    print(f"{label}: {probabilities[0][i]*100:.2f}%")

# 가장 높은 확률의 클래스 출력
predicted = torch.argmax(output, 1)
print(f"\n예측 결과: {labels[predicted.item()]}")
