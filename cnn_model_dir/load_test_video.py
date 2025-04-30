import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

# ✅ 클래스 정의
classes = ["__background__", "cup"]
score_sum = 0
# ✅ 모델 로드 함수
def get_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# ✅ 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model(num_classes=len(classes))
model.load_state_dict(torch.load("faster_rcnn_cup2.pth", map_location=device))
model.to(device)
model.eval()

# ✅ 영상 불러오기
cap = cv2.VideoCapture("IMG_1481.mp4")

# ✅ 저장 옵션 (선택)
save_output = True
out = None
if save_output:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

# ✅ 프레임별 객체 탐지
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # RGB로 변환 후 텐서화
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = F.to_tensor(image).unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        prediction = model(image_tensor)[0]

    # 바운딩 박스 표시
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score > 0.5:
            x1, y1, x2, y2 = [int(v.item()) for v in box]
            class_name = classes[label]
            score_sum += score
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 출력
    # cv2.imshow("Detection", frame)
    if save_output:
        out.write(frame)

    # 'q' 누르면 종료
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
print("score_sum :" , score_sum)
cap.release()
if save_output:
    out.release()
cv2.destroyAllWindows()
