import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# 1. Đọc hình ảnh
image = cv2.imread('trial_lie_022_aligned/frame_det_00_000001.bmp', cv2.IMREAD_GRAYSCALE)  # Chuyển hình ảnh thành grayscale

# 2. Tính toán HOG
hog_features, hog_image = hog(image,
                              orientations=9,
                              pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2),
                              block_norm='L2-Hys',
                              visualize=True)
# 3. Hiển thị hình ảnh gốc và biểu đồ HOG
plt.figure(figsize=(10, 5))
# Hình ảnh gốc
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

# Biểu đồ HOG
plt.subplot(1, 2, 2)
plt.title("HOG Visualization")
plt.imshow(hog_image, cmap='gray')

plt.show()
