import cv2
import numpy as np
import os
from sklearn.linear_model import RANSACRegressor

def compute_dense_optical_flow(img1, img2):
    # Chuyển đổi sang ảnh xám
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Tính toán dòng quang học dày đặc
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def detect_dense_feature_points(image):
    # Phát hiện điểm đặc trưng bằng cách lấy mẫu dày đặc từ lưới đa tỉ lệ
    h, w = image.shape[:2]
    step = 10  # Kích thước lưới
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1)
    points = np.vstack((x, y)).T
    return points

def filter_camera_motion_ransac(points, flow):
    # Sử dụng RANSAC để loại bỏ chuyển động camera
    valid_points = []
    valid_trajectories = []

    for point in points:
        x, y = point
        dx, dy = flow[int(y), int(x)]
        model_ransac = RANSACRegressor()
        try:
            model_ransac.fit([[x, y]], [[x + dx, y + dy]])
            valid_points.append((x, y))
            valid_trajectories.append((x + dx, y + dy))
        except Exception as e:
            continue

    return np.array(valid_points), np.array(valid_trajectories)

def visualize_dense_optical_flow(image, points, trajectories):
    # Vẽ các điểm đỏ và các đường trajectory
    vis = image.copy()
    for (x1, y1), (x2, y2) in zip(points, trajectories):
        cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)  # Vẽ đường xanh lá
        cv2.circle(vis, (int(x1), int(y1)), 4, (0, 0, 255), -1)  # Vẽ keypoint màu đỏ lớn hơn

    return vis

# Chạy trên folder ảnh
input_folder = "landmark/trial_lie_006_aligned"  # Thay bằng đường dẫn folder chứa ảnh
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

    # Phát hiện điểm đặc trưng dày đặc từ lưới đa tỉ lệ
    dense_points = detect_dense_feature_points(img1)

    # Tính dòng quang học
    flow = compute_dense_optical_flow(img1, img2)

    # Lọc chuyển động camera bằng RANSAC
    filtered_points, filtered_trajectories = filter_camera_motion_ransac(dense_points, flow)

    # Vẽ và lưu kết quả
    result = visualize_dense_optical_flow(img1, filtered_points, filtered_trajectories)
    output_path = os.path.join(output_folder, f"optical_flow_ransac_{i}.jpg")
    cv2.imwrite(output_path, result)
    print(f"Lưu kết quả: {output_path}")