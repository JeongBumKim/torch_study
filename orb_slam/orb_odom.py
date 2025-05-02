import cv2

# ✅ ORB 특징 추출기 생성 (검출율 높게 설정)
orb = cv2.ORB_create(
    nfeatures=500,        # 최대 1000개 특징점
    scaleFactor=1.2,       # 피라미드 축소 비율 (기본 1.2)
    nlevels=8,             # 피라미드 레벨 수
    edgeThreshold=15,      # 경계 제외 범위 ↓
    fastThreshold=10       # FAST 민감도 ↓ (더 많이 검출됨)
)

# ✅ 웹캠 열기 (기기 번호는 상황에 따라 0 또는 2 등으로 변경)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ ORB 특징점 검출
    keypoints = orb.detect(gray, None)

    # ✅ 초록색 점으로 특징점 시각화
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # ✅ 특징점 수 표시
    cv2.putText(frame, f"Features: {len(keypoints)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ✅ 결과 출력
    cv2.imshow("ORB Features (Green)", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()
