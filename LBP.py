import os
import cv2
import numpy as np

# Các hàm tính LBP như trước
def compute_lbp_pixel_value(center, neighbors):
    binary_pattern = [1 if neighbor >= center else 0 for neighbor in neighbors]
    return sum([val * (2 ** i) for i, val in enumerate(binary_pattern)])


def extract_lbp_features(image, radius=1, points=8):
    height, width = image.shape
    lbp_image = np.zeros_like(image, dtype=np.uint8)

    circle_offsets = [
        (int(np.round(radius * np.cos(2 * np.pi * i / points))),
         int(np.round(radius * np.sin(2 * np.pi * i / points))))
        for i in range(points)
    ]

    for x in range(radius, width - radius):
        for y in range(radius, height - radius):
            neighbors = [image[y + dy, x + dx] for dx, dy in circle_offsets]
            lbp_image[y, x] = compute_lbp_pixel_value(image[y, x], neighbors)
    return lbp_image


def compute_lbp_histogram(image, num_regions=4, bins=256):
    height, width = image.shape
    region_height, region_width = height // num_regions, width // num_regions
    histograms = []

    for i in range(num_regions):
        for j in range(num_regions):
            region = image[i * region_height:(i + 1) * region_height,
                     j * region_width:(j + 1) * region_width]
            hist, _ = np.histogram(region, bins=bins, range=(0, bins))
            histograms.append(hist)
    return np.concatenate(histograms)


def compute_and_save_lbp(input_dir, output_dir, radius=1, points=8, num_regions=4, bins=256):
    os.makedirs(output_dir, exist_ok=True)

    for folder_name in os.listdir(input_dir):
        if folder_name.endswith("_aligned"):
            folder_path = os.path.join(input_dir, folder_name)
            if os.path.isdir(folder_path):
                feature_sums = None
                count = 0

                # Duyệt qua các ảnh trong thư mục
                for filename in sorted(os.listdir(folder_path)):
                    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        image_path = os.path.join(folder_path, filename)
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            continue

                        # Tính LBP và histogram
                        lbp_image = extract_lbp_features(image, radius, points)
                        lbp_feature_vector = compute_lbp_histogram(lbp_image, num_regions=num_regions, bins=bins)

                        if feature_sums is None:
                            feature_sums = np.zeros_like(lbp_feature_vector, dtype=np.float64)
                        feature_sums += lbp_feature_vector
                        count += 1

                # Tính vector trung bình và lưu
                if count > 0:
                    avg_lbp = feature_sums / count
                    base_name = folder_name.replace("_aligned", "_feature")
                    output_path = os.path.join(output_dir, base_name + ".npy")
                    np.save(output_path, avg_lbp)
                    print(f"Đã lưu đặc trưng LBP cho {folder_name} tại {output_path}")
                else:
                    print(f"Không có ảnh hợp lệ trong {folder_path}")


# Thư mục input và output
input_dir = "landmark"
output_dir = "output/lbp"

# Gọi hàm để tính và lưu LBP trung bình
compute_and_save_lbp(input_dir, output_dir, radius=1, points=8, num_regions=4, bins=256)
