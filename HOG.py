import os
import cv2
import numpy as np
from skimage.feature import hog

# Tham số HOG
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
block_norm = 'L2-Hys'

input_dir = "landmark"
output_dir = "output/hog"

# Tạo thư mục output/hog nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Chỉ duyệt qua các folder có tên kết thúc bằng "_aligned"
for folder_name in os.listdir(input_dir):
    if folder_name.endswith("_aligned"):
        folder_path = os.path.join(input_dir, folder_name)
        if os.path.isdir(folder_path):
            # Đọc tất cả frame trong folder có đuôi _aligned
            frames = []
            for filename in sorted(os.listdir(folder_path)):
                if filename.lower().endswith(('.bmp', '.jpg', '.png')):
                    frame_path = os.path.join(folder_path, filename)
                    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        frames.append(frame)

            # Nếu không có frame nào hợp lệ
            if len(frames) == 0:
                print(f"Không có frame trong thư mục {folder_path}")
                continue

            # Tính HOG và lấy trung bình
            hog_sums = None
            count = 0
            for fr in frames:
                hog_features = hog(fr,
                                   orientations=orientations,
                                   pixels_per_cell=pixels_per_cell,
                                   cells_per_block=cells_per_block,
                                   block_norm=block_norm,
                                   visualize=False)
                if hog_sums is None:
                    hog_sums = np.zeros_like(hog_features, dtype=np.float64)
                hog_sums += hog_features
                count += 1

            avg_hog = hog_sums / count

            # Tạo tên file đầu ra
            # Thay "_aligned" bằng "_feature"
            base_name = folder_name.replace("_aligned", "_feature")
            output_path = os.path.join(output_dir, base_name + ".npy")
            np.save(output_path, avg_hog)
            print(f"Đã lưu đặc trưng HOG cho {folder_name} tại {output_path}")
