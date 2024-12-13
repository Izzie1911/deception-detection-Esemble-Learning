import cv2
import numpy as np
import os
from sklearn.linear_model import RANSACRegressor

# Hàm tính toán dòng quang học dày đặc
def compute_dense_optical_flow(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 10, 3, 5, 1.2, 0)
    return flow

# Hàm lọc trajectory bằng RANSAC
def filter_trajectory_with_ransac(trajectory):
    if len(trajectory) < 2:
        return trajectory  # Không đủ điểm để lọc

    X = np.array([p[0] for p in trajectory]).reshape(-1, 1)
    y = np.array([p[1] for p in trajectory])

    model = RANSACRegressor()
    try:
        model.fit(X, y)
        inlier_mask = model.inlier_mask_
        filtered_trajectory = [trajectory[i] for i in range(len(trajectory)) if inlier_mask[i]]
        return filtered_trajectory
    except ValueError:
        return trajectory  # Không thể áp dụng RANSAC

# Theo dõi các điểm với RANSAC lọc trajectory
def track_points_with_trajectories(points, flow, T=15):
    trajectories = []
    for point in points:
        x, y = point
        trajectory = [(x, y)]
        for _ in range(T):
            dx, dy = flow[int(y), int(x)]
            x, y = x + dx, y + dy
            trajectory.append((x, y))
            if x < 0 or x >= flow.shape[1] or y < 0 or y >= flow.shape[0]:
                break
        # Lọc trajectory bằng RANSAC
        filtered_trajectory = filter_trajectory_with_ransac(trajectory)
        trajectories.append(filtered_trajectory)
    return trajectories

# Vẽ các trajectory

def visualize_dense_optical_flow(image, flow, trajectories):
    vis = image.copy()
    for trajectory in trajectories:
        for i in range(len(trajectory) - 1):
            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]
            cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        if trajectory:
            x, y = trajectory[0]
            cv2.circle(vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    return vis

# Chạy trên folder ảnh
input_folder = "landmark/trial_lie_014_aligned"  # Thay bằng đường dẫn folder chứa ảnh
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

    flow = compute_dense_optical_flow(img1, img2)

    # Tạo lưới điểm và theo dõi trajectory
    h, w = img1.shape[:2]
    step = 10
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    points = np.vstack((x, y)).T

    trajectories = track_points_with_trajectories(points, flow)

    # Vẽ và lưu kết quả
    result = visualize_dense_optical_flow(img1, flow, trajectories)
    output_path = os.path.join(output_folder, f"optical_flow_{i}.jpg")
    cv2.imwrite(output_path, result)
    print(f"Lưu kết quả: {output_path}")