from PIL import Image
import torch
from torchvision import transforms

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
image = Image.open("test_image.png")   # 여기에 사용할 이미지 파일 경로
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
