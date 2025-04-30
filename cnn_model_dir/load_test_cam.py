import torch
import cv2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# 클래스 정의
classes = ["__background__", "sirbot"]  # label 1 = cup

# 모델 로드 함수
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 불러오기
model = get_model(num_classes=len(classes))
model.load_state_dict(torch.load("faster_rcnn_cup.pth", map_location=device))
model.to(device)
model.eval()

# 웹캠 열기 (0 = 기본 캠)
cap = cv2.VideoCapture(0)

print("🎥 실시간 탐지 시작 (ESC 키로 종료)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 프레임을 PIL처럼 처리
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # 바운딩 박스 그리기
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score.item() > 0.93:
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            class_name = classes[label]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} ({score.item():.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 프레임 출력
    cv2.imshow("Faster R-CNN Real-Time", frame)

    # ESC로 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
