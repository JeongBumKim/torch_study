import cv2
import os
import numpy as np

# ✅ 영상 입력 (2번 웹캠 또는 필요 시 0/1로 바꾸세요)
cap = cv2.VideoCapture(2)

# ✅ 템플릿 이미지 폴더 경로
template_dir = "image_model/dataset/sirbot"
template_files = sorted([f for f in os.listdir(template_dir) if f.endswith(".png")])

# ✅ SIFT 초기화
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# ✅ 템플릿 이미지의 키포인트와 디스크립터 저장
template_data = []
for filename in template_files:
    path = os.path.join(template_dir, filename)
    template_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if template_img is None:
        print(f"❌ 이미지 불러오기 실패: {filename}")
        continue
    kp, des = sift.detectAndCompute(template_img, None)
    template_data.append((template_img, kp, des))

print(f"📦 총 {len(template_data)}개의 템플릿이 로드되었습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_des = sift.detectAndCompute(gray, None)

    best_match_count = 0
    best_template = None
    best_matches = []
    best_homography = None

    for template_img, template_kp, template_des in template_data:
        if frame_des is None or template_des is None:
            continue

        matches = bf.match(template_des, frame_des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > best_match_count and len(matches) > 10:
            best_match_count = len(matches)
            best_template = (template_img, template_kp)
            best_matches = matches[:20]

            src_pts = np.float32([template_kp[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            best_homography = H

    if best_template and best_homography is not None:
        h, w = best_template[0].shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected_corners = cv2.perspectiveTransform(corners, best_homography)
        cv2.polylines(frame, [np.int32(projected_corners)], True, (0, 255, 0), 2)
        cv2.putText(frame, f"SIRBOT Match ({best_match_count} features)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("SIFT SIRBOT Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
