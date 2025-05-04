import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ✅ 입력 이미지 불러오기
img = Image.open("cnn_test_img.jpg").convert("RGB")  # 임의의 이미지 파일 사용
transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
])
x = transform(img).unsqueeze(0)  # shape: [1, 3, 128, 128]

# ✅ 간단한 CNN 구성
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
bn = nn.BatchNorm2d(8)
relu = nn.ReLU()

# ✅ 각 레이어 통과
x_conv = conv(x)             # conv 결과
x_bn = bn(x_conv)            # batchnorm 결과
x_relu = relu(x_bn)          # relu 결과

# ✅ 시각화 함수
def show_feature_maps(tensor, title):
    maps = tensor.squeeze(0).detach().cpu().numpy()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):  # 앞 4개의 채널만 표시
        axes[i].imshow(maps[i], cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"{title} #{i}")
    plt.tight_layout()
    plt.show()

# ✅ 시각적으로 비교
show_feature_maps(x_conv, "Conv")
show_feature_maps(x_bn, "BatchNorm")
show_feature_maps(x_relu, "ReLU")
