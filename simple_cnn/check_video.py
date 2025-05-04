import cv2

# ✅ 비디오 파일 경로 설정 (파일명 또는 전체 경로)
video_path = "output.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 비디오 파일을 열 수 없습니다.")
else:
    print("✅ 비디오 파일 열기 성공!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🔁 프레임 읽기 실패 또는 영상 끝")
        break

    # 프레임 화면에 출력
    cv2.imshow("MP4 Video Test", frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
