import cv2
import numpy as np

def compute_orientation(moment10, moment01, moment00):
    """
    Compute orientation based on moments.
    """
    if moment00 == 0:
        return 0
    cx = moment10 / moment00
    cy = moment01 / moment00
    angle = np.arctan2(cy, cx)
    return angle

def brief_descriptor(patch, keypoint, random_pairs):
    """
    Generate a binary descriptor for a given keypoint using rBRIEF.
    """
    x, y = keypoint
    h, w = patch.shape
    binary_descriptor = []
    for pair in random_pairs:
        pt1, pt2 = pair
        pt1_x = min(max(int(pt1[0] + x), 0), w - 1)
        pt1_y = min(max(int(pt1[1] + y), 0), h - 1)
        pt2_x = min(max(int(pt2[0] + x), 0), w - 1)
        pt2_y = min(max(int(pt2[1] + y), 0), h - 1)

        intensity1 = patch[pt1_y, pt1_x]
        intensity2 = patch[pt2_y, pt2_x]

        binary_descriptor.append(1 if intensity1 < intensity2 else 0)

    return np.array(binary_descriptor)

# Load the image
image = cv2.imread("trial_lie_022_aligned/frame_det_00_000001.bmp", cv2.IMREAD_GRAYSCALE)

# Detect keypoints using FAST
fast = cv2.FastFeatureDetector_create()
keypoints = fast.detect(image, None)

# Create random pairs for BRIEF
num_pairs = 256
random_pairs = np.random.uniform(-15, 15, size=(num_pairs, 2, 2))

# Compute descriptors
keypoint_descriptors = []
for keypoint in keypoints:
    x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
    patch_size = 31
    half_patch = patch_size // 2

    # Extract the patch around the keypoint
    patch = image[max(0, y - half_patch):y + half_patch + 1, max(0, x - half_patch):x + half_patch + 1]

    if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
        continue

    # Compute moments
    moment00 = patch.sum()
    moment10 = np.sum(np.arange(patch.shape[1]) * patch.sum(axis=0))
    moment01 = np.sum(np.arange(patch.shape[0]) * patch.sum(axis=1))

    # Compute orientation
    orientation = compute_orientation(moment10, moment01, moment00)

    # Compute binary descriptor
    descriptor = brief_descriptor(patch, (x, y), random_pairs)

    keypoint_descriptors.append((keypoint, descriptor))

# Draw keypoints for visualization
output_image = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0))

# Display the keypoints with OpenCV
cv2.imshow('Keypoints with ORB-like Descriptors', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
