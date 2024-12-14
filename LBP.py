import numpy as np
import cv2
import matplotlib.pyplot as plt


def compute_lbp_pixel_value(center, neighbors):
    """
    Tính toán giá trị LBP cho một pixel trung tâm.

    Args:
        center (int): Giá trị grayscale của pixel trung tâm.
        neighbors (list of int): Các giá trị grayscale của các pixel lân cận.

    Returns:
        int: Giá trị LBP được mã hóa dưới dạng số nguyên.
    """
    # Áp dụng hàm s(x) cho từng điểm lân cận
    binary_pattern = [1 if neighbor >= center else 0 for neighbor in neighbors]
    # Chuyển nhị phân sang số thập phân
    return sum([val * (2 ** i) for i, val in enumerate(binary_pattern)])


def extract_lbp_features(image, radius=1, points=8):
    """
    Tính toán mã LBP cho toàn bộ ảnh.

    Args:
        image (numpy.ndarray): Ảnh xám đầu vào.
        radius (int): Bán kính của vòng tròn điểm lân cận.
        points (int): Số lượng điểm lân cận.

    Returns:
        numpy.ndarray: Ảnh mã hóa LBP.
    """
    height, width = image.shape
    lbp_image = np.zeros_like(image, dtype=np.uint8)

    # Tính tọa độ lân cận trên vòng tròn
    circle_offsets = [
        (int(np.round(radius * np.cos(2 * np.pi * i / points))),
         int(np.round(radius * np.sin(2 * np.pi * i / points))))
        for i in range(points)
    ]

    # Duyệt qua từng pixel trong ảnh
    for x in range(radius, width - radius):
        for y in range(radius, height - radius):
            neighbors = []
            for dx, dy in circle_offsets:
                neighbors.append(image[y + dy, x + dx])
            lbp_image[y, x] = compute_lbp_pixel_value(image[y, x], neighbors)

    return lbp_image


def compute_lbp_histogram(image, num_regions=4, bins=256):
    """
    Chia ảnh thành các vùng nhỏ và tính histogram cho từng vùng.

    Args:
        image (numpy.ndarray): Ảnh mã hóa LBP.
        num_regions (int): Số vùng chia trên mỗi chiều (tổng cộng là num_regions^2 vùng).
        bins (int): Số lượng bins trong histogram.

    Returns:
        numpy.ndarray: Vector đặc trưng LBP từ toàn bộ ảnh.
    """
    height, width = image.shape
    region_height, region_width = height // num_regions, width // num_regions
    histograms = []

    for i in range(num_regions):
        for j in range(num_regions):
            # Lấy một vùng nhỏ từ ảnh
            region = image[i * region_height:(i + 1) * region_height,
                     j * region_width:(j + 1) * region_width]
            # Tính histogram của vùng
            hist, _ = np.histogram(region, bins=bins, range=(0, bins))
            histograms.append(hist)

    # Ghép tất cả các histogram lại thành một vector
    return np.concatenate(histograms)


# Đọc ảnh và chuyển sang ảnh xám
image_path = "landmark/trial_lie_014_aligned/frame_det_00_000082.bmp"  # Thay bằng đường dẫn ảnh
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Tính mã LBP
lbp_image = extract_lbp_features(image, radius=1, points=8)

# Tính vector histogram đặc trưng
lbp_feature_vector = compute_lbp_histogram(lbp_image, num_regions=4, bins=256)

# Hiển thị ảnh LBP
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.subplot(1, 2, 2)
plt.title("LBP Encoded Image")
plt.imshow(lbp_image, cmap="gray")
plt.show()

# In kích thước của vector đặc trưng LBP
print("LBP Feature Vector Shape:", lbp_feature_vector.shape)


def plot_histogram(lbp_image, num_regions=4, bins=256):
    """
    Vẽ biểu đồ histogram của ảnh LBP, cả tổng quát và từng vùng nhỏ.

    Args:
        lbp_image (numpy.ndarray): Ảnh mã hóa LBP.
        num_regions (int): Số vùng chia trên mỗi chiều.
        bins (int): Số lượng bins trong histogram.
    """
    height, width = lbp_image.shape
    region_height, region_width = height // num_regions, width // num_regions

    # Tính histogram toàn bộ ảnh
    total_hist, _ = np.histogram(lbp_image, bins=bins, range=(0, bins))

    # Vẽ histogram tổng quát
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Overall LBP Histogram")
    plt.bar(range(bins), total_hist, color='blue', alpha=0.7)
    plt.xlabel("LBP Value")
    plt.ylabel("Frequency")

    # Tính và vẽ histogram từng vùng
    plt.subplot(2, 1, 2)
    plt.title(f"LBP Histograms for {num_regions}x{num_regions} Regions")
    for i in range(num_regions):
        for j in range(num_regions):
            # Lấy một vùng nhỏ
            region = lbp_image[i * region_height:(i + 1) * region_height,
                     j * region_width:(j + 1) * region_width]
            # Tính histogram cho vùng
            region_hist, _ = np.histogram(region, bins=bins, range=(0, bins))
            # Vẽ histogram
            plt.bar(range(bins), region_hist, alpha=0.5, label=f"Region {i},{j}")

    plt.xlabel("LBP Value")
    plt.ylabel("Frequency")
    plt.legend(loc="upper right", fontsize='small', ncol=2)
    plt.tight_layout()
    plt.show()