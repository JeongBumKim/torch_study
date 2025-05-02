import torch

# 모델 정의 (저장할 때 사용한 구조와 동일해야 함)
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

model = get_model(num_classes=2)  # 클래스 수 동일해야 함
model.load_state_dict(torch.load("faster_rcnn_cup.pth", map_location="cpu"))

# ✅ 모델 구조 출력
print(model)

# ✅ 특정 계층 가중치 확인
print("\n[Classifier Layer]")
print(model.roi_heads.box_predictor.cls_score)

print("\n[Classifier Weights]")
print(model.roi_heads.box_predictor.cls_score.weight.shape)
print(model.roi_heads.box_predictor.cls_score.weight)

print("\n[All keys in state_dict]")
for k in model.state_dict().keys():
    print(k)
