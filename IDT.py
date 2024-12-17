import cv2
import numpy as np
import os
from sklearn.linear_model import RANSACRegressor
import math

# Tính toán dòng quang học dày đặc
def compute_dense_optical_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    return flow

# Chuẩn hóa quỹ đạo theo tổng độ lớn các vector dịch chuyển
def normalize_trajectory(trajectory):
    if len(trajectory) < 2:
        return trajectory
    displacement_vectors = [np.linalg.norm(np.array(trajectory[i + 1]) - np.array(trajectory[i]))
                            for i in range(len(trajectory) - 1)]
    magnitude_sum = sum(displacement_vectors)
    if magnitude_sum == 0:
        return trajectory
    normalized_trajectory = [(x / magnitude_sum, y / magnitude_sum) for x, y in trajectory]
    return normalized_trajectory

# Theo dõi các điểm với RANSAC và chuẩn hóa
def track_points_with_trajectories(points, flow, T=15):
    h, w = flow.shape[:2]
    trajectories = []

    for point in points:
        x, y = point
        trajectory = [(x, y)]

        for _ in range(T):
            dx, dy = flow[int(y), int(x)]

            # Kiểm tra giá trị hợp lệ của dòng quang học
            if not (math.isfinite(dx) and math.isfinite(dy)):
                print(f"Invalid flow at ({int(x)}, {int(y)}) -> dx: {dx}, dy: {dy}")
                break

            # Cập nhật tọa độ
            x, y = x + dx, y + dy

            # Kiểm tra giới hạn biên ảnh
            if x < 0 or x >= w or y < 0 or y >= h:
                print(f"Out of bounds at x: {x}, y: {y}")
                break

            trajectory.append((x, y))

        trajectories.append(trajectory)
    return trajectories

# Vẽ quỹ đạo

def visualize_dense_optical_flow(image, trajectories):
    vis = image.copy()
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            # Lấy tọa độ điểm hiện tại và tiếp theo
            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]

            # Kiểm tra tọa độ hợp lệ và ép kiểu về int
            if isinstance(x1, (int, float)) and isinstance(y1, (int, float)) and \
               isinstance(x2, (int, float)) and isinstance(y2, (int, float)):
                if math.isfinite(x1) and math.isfinite(y1) and \
                   math.isfinite(x2) and math.isfinite(y2):
                    cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        # Vẽ điểm bắt đầu
        if trajectory:
            x, y = trajectory[0]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and \
               math.isfinite(x) and math.isfinite(y):
                cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    return vis


def compute_normalized_feature_vector(trajectory):
    if len(trajectory) < 2:
        return []  # Không đủ điểm để tính

    # Tính các vector dịch chuyển ∆P_t
    displacement_vectors = [np.array(trajectory[i + 1]) - np.array(trajectory[i])
                            for i in range(len(trajectory) - 1)]

    # Tính tổng độ lớn của các vector dịch chuyển
    magnitude_sum = sum(np.linalg.norm(v) for v in displacement_vectors)

    # Chuẩn hóa các vector dịch chuyển
    if magnitude_sum > 0:
        normalized_vectors = [v / magnitude_sum for v in displacement_vectors]
        return normalized_vectors  # Trả về danh sách vector đã chuẩn hóa
    else:
        return displacement_vectors  # Trả về vector gốc nếu tổng độ lớn là 0


if __name__ == '__main__':
    # Folder chứa ảnh đầu vào và đầu ra
    input_folder = "landmark/trial_lie_014_aligned"  # Đường dẫn folder chứa ảnh
    output_folder = "output_folder"  # Thư mục lưu kết quả
    os.makedirs(output_folder, exist_ok=True)

    # Lấy danh sách các file ảnh và sắp xếp
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.bmp', '.jpeg'))])

    for i in range(len(image_files) - 1):
        image1_path = os.path.join(input_folder, image_files[i])
        image2_path = os.path.join(input_folder, image_files[i + 1])

        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)

        if img1 is None or img2 is None:
            print(f"Không thể đọc ảnh: {image1_path} hoặc {image2_path}")
            continue

        # Tính dòng quang học
        flow = compute_dense_optical_flow(img1, img2)

        # Tạo lưới điểm
        h, w = img1.shape[:2]
        step = 10
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
        points = np.vstack((x, y)).T

        # Theo dõi quỹ đạo
        trajectories = track_points_with_trajectories(points, flow)
        normalized_feature_vectors = [compute_normalized_feature_vector(trajectory) for trajectory in trajectories]
        for idx, fv in enumerate(normalized_feature_vectors):
            print(f"Feature vector S' của quỹ đạo {idx}: {fv}")
        # Vẽ quỹ đạo
        result = visualize_dense_optical_flow(img1, trajectories)
        output_path = os.path.join(output_folder, f"optical_flow_{i}.jpg")
        cv2.imwrite(output_path, result)
