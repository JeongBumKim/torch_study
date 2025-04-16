import cv2
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn

# 모델 구조 (클래스 없이 그대로 Sequential만 사용)
model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 64),
    nn.ReLU(),
    nn.Linear(64, 1)  # SIRBOT 유무 (이진 분류)
)

# state_dict 로드
model.load_state_dict(torch.load("saved_model/sirbot_classifier.pth"))
model.eval()


# ✅ 클래스 수와 이미지 변환
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ✅ 원본 이미지 불러오기
image_path = "image_model/image/captured_image_0.png"  # 전체 이미지
image = cv2.imread(image_path)
h, w, _ = image.shape
output_image = image.copy()

# ✅ 슬라이딩 윈도우 설정
window_size = 40
stride = 4  # 윈도우 간 간격
threshold = 0.9  # SIRBOT일 확률이 이 이상이면 박스 그림

# ✅ 윈도우 탐색
for y in range(0, h - window_size + 1, stride):
    for x in range(0, w - window_size + 1, stride):
        crop = image[y:y+window_size, x:x+window_size]
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(crop_pil).unsqueeze(0)  # [1, 3, 64, 64]

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            prob_sirbot = probs[0][0].item()  # 클래스 0이 SIRBOT일 때

            if prob_sirbot > threshold:
                cv2.rectangle(output_image, (x, y), (x+window_size, y+window_size), (0, 255, 0), 2)
                cv2.putText(output_image, f"{prob_sirbot:.2f}", (x, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

# ✅ 결과 출력
cv2.imshow("SIRBOT Detection", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
