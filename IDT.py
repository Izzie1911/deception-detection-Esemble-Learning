import cv2
import numpy as np

# Đọc ảnh đầu vào
frame = cv2.imread('landmark/trial_lie_014_aligned/frame_det_00_000001.bmp', cv2.IMREAD_GRAYSCALE)

# Tạo bản sao ảnh để vẽ keypoint
output_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

# Chia lưới: cấp độ ban đầu
grid_size = 16  # Kích thước ô ban đầu
scales = [1, 1 / np.sqrt(2), 1 / 2]  # Tỷ lệ lưới

# Lặp qua từng tỷ lệ
for scale in scales:
    scaled_size = int(grid_size * scale)  # Kích thước ô tại tỷ lệ hiện tại
    height, width = frame.shape

    # Chia lưới
    for y in range(0, height, scaled_size):
        for x in range(0, width, scaled_size):
            # Tạo một ô lưới (cell)
            cell = frame[y:y + scaled_size, x:x + scaled_size]

            # Phát hiện các điểm đặc trưng trong ô
            keypoints = cv2.goodFeaturesToTrack(
                cell,
                maxCorners=10,
                qualityLevel=0.3,
                minDistance=3
            )

            # Nếu phát hiện được điểm đặc trưng, vẽ lên ảnh
            if keypoints is not None:
                for kp in keypoints:
                    # Điểm keypoint trong tọa độ của ô
                    x_kp, y_kp = kp.ravel()
                    # Chuyển tọa độ về hệ tọa độ gốc của ảnh
                    x_global = int(x + x_kp)
                    y_global = int(y + y_kp)
                    # Vẽ điểm keypoint
                    cv2.circle(output_frame, (x_global, y_global), 1, (0, 0, 255), -1)  # Màu đỏ

# Lưu kết quả và hiển thị ảnh
cv2.imshow('Keypoints', output_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
