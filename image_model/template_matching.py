import cv2

# ✅ 영상 입력: 0은 웹캠, 또는 "video.mp4"처럼 파일도 가능
cap = cv2.VideoCapture(2)

# ✅ 템플릿 이미지 로드 (예: 자른 SIRBOT)
template = cv2.imread("image_model/dataset/sirbot/cropped_captured_image_0.png", cv2.IMREAD_GRAYSCALE)
th, tw = template.shape[:2]

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 영상 프레임을 읽을 수 없습니다.")
        break

    # ✅ 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ✅ 템플릿 매칭
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # ✅ 임계값 설정 (높을수록 일치 정도가 더 정확해야 함)
    if max_val > 0.7:  # 필요시 0.8, 0.9 등으로 조정 가능
        top_left = max_loc
        bottom_right = (top_left[0] + tw, top_left[1] + th)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, f"Match: {max_val:.2f}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # ✅ 출력
    cv2.imshow("SIRBOT Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ 정리
cap.release()
cv2.destroyAllWindows()

# 템플릿 이미지 (SIRBOT 잘린 부분)
# template = cv2.imread("image_model/dataset/sirbot/cropped_captured_image_0.png")

