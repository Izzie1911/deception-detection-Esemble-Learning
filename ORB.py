import cv2
import numpy as np


def compute_orientation(moment10, moment01, moment00):
    """
    Compute orientation based on moments.
    According to the formula: theta = atan2(m01, m10)
    """
    if moment00 == 0:
        return 0.0
    cx = moment10 / moment00
    cy = moment01 / moment00
    angle = np.arctan2(cy, cx)
    return angle


def rotate_pairs(random_pairs, angle):
    """
    Rotate the sampling pairs according to the orientation angle theta.
    Q_theta = R_theta * Q
    R_theta = [[cos(theta), -sin(theta)],
               [sin(theta),  cos(theta)]]
    """
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])

    rotated = []
    for pair in random_pairs:
        pt1, pt2 = pair
        # pt1, pt2 đang là offset từ điểm đặc trưng (kc)
        # Áp dụng phép quay cho từng điểm
        rotated_pt1 = R.dot(pt1)
        rotated_pt2 = R.dot(pt2)
        rotated.append((rotated_pt1, rotated_pt2))
    return rotated


def brief_descriptor(patch, rotated_pairs):
    """
    Generate a binary descriptor from the rotated pairs.
    patch: vùng ảnh xung quanh keypoint, kích thước 31x31 (giả sử)
    rotated_pairs: danh sách các cặp điểm sau khi xoay
    """
    h, w = patch.shape
    half_patch = h // 2
    binary_descriptor = []

    for (pt1, pt2) in rotated_pairs:
        # Chuyển từ tọa độ tương đối sang chỉ số trong patch
        pt1_x = half_patch + int(round(pt1[0]))
        pt1_y = half_patch + int(round(pt1[1]))
        pt2_x = half_patch + int(round(pt2[0]))
        pt2_y = half_patch + int(round(pt2[1]))

        # Giới hạn tọa độ trong patch
        pt1_x = np.clip(pt1_x, 0, w - 1)
        pt1_y = np.clip(pt1_y, 0, h - 1)
        pt2_x = np.clip(pt2_x, 0, w - 1)
        pt2_y = np.clip(pt2_y, 0, h - 1)

        intensity1 = patch[pt1_y, pt1_x]
        intensity2 = patch[pt2_y, pt2_x]

        binary_descriptor.append(1 if intensity1 < intensity2 else 0)

    return np.array(binary_descriptor, dtype=np.uint8)


# Load the image in grayscale
image = cv2.imread("landmark/trial_lie_022_aligned/frame_det_00_000001.bmp", cv2.IMREAD_GRAYSCALE)

# Detect keypoints using FAST
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(image, None)

# Create random pairs for BRIEF (N=256)
# Các cặp điểm được chọn ngẫu nhiên trong một vùng xung quanh keypoint
# Ở đây ta coi (0,0) là tâm, tọa độ cặp điểm trong [-15,15]
num_pairs = 256
random_pairs = np.random.uniform(-15, 15, size=(num_pairs, 2, 2))

keypoint_descriptors = []
patch_size = 31
half_patch = patch_size // 2

for keypoint in keypoints:
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])

    # Trích xuất patch xung quanh keypoint
    # Chú ý cần kiểm tra biên
    if (y - half_patch < 0 or y + half_patch >= image.shape[0] or
            x - half_patch < 0 or x + half_patch >= image.shape[1]):
        continue
    patch = image[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]

    # Tính mô-men
    # moment00 = sum of intensities
    moment00 = patch.sum()
    # moment10 = sum(x * I(x,y)), moment01 = sum(y * I(x,y))
    # x và y ở đây tính từ 0 đến patch_size-1, ta có thể coi (0,0) ở góc trên trái.
    # Theo công thức: m10 = Σx Σy x*I(x,y), m01 = Σx Σy y*I(x,y)
    # Ta có thể tính:
    xs = np.arange(patch.shape[1])
    ys = np.arange(patch.shape[0])
    moment10 = np.sum(xs * patch.sum(axis=0))  # sum theo hàng dọc
    moment01 = np.sum(ys * patch.sum(axis=1))  # sum theo hàng ngang

    # Tính góc quay
    orientation = compute_orientation(moment10, moment01, moment00)

    # Xoay các cặp random theo orientation
    rotated_pairs = rotate_pairs(random_pairs, orientation)

    # Tạo descriptor
    descriptor = brief_descriptor(patch, rotated_pairs)

    keypoint_descriptors.append((keypoint, descriptor))

# Vẽ keypoint để kiểm tra
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

cv2.imshow('Keypoints with ORB-like Descriptors', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
