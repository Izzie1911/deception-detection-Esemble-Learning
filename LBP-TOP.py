import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import os
import matplotlib.pyplot as plt
# Function to compute LBP codes
def compute_lbp(image, radius=1, n_points=8):
    return local_binary_pattern(image, n_points, radius, method="uniform")

# Function to extract LBP-TOP features
def lbp_top(video_frames, radius=1, n_points=8):
    num_frames, height, width = video_frames.shape

    # Initialize histograms for XY, XT, and YT planes
    hist_xy = np.zeros((256,), dtype=np.float32)
    hist_xt = np.zeros((256,), dtype=np.float32)
    hist_yt = np.zeros((256,), dtype=np.float32)

    # XY Plane
    for frame in range(num_frames):
        lbp_xy = compute_lbp(video_frames[frame], radius, n_points)
        hist_xy += np.histogram(lbp_xy, bins=256, range=(0, 255))[0]

    # XT Plane
    for y in range(height):
        slice_xt = video_frames[:, y, :]
        lbp_xt = compute_lbp(slice_xt, radius, n_points)
        hist_xt += np.histogram(lbp_xt, bins=256, range=(0, 255))[0]

    # YT Plane
    for x in range(width):
        slice_yt = video_frames[:, :, x]
        lbp_yt = compute_lbp(slice_yt, radius, n_points)
        hist_yt += np.histogram(lbp_yt, bins=256, range=(0, 255))[0]

    # Normalize histograms
    hist_xy /= np.sum(hist_xy)
    hist_xt /= np.sum(hist_xt)
    hist_yt /= np.sum(hist_yt)

    # Concatenate histograms
    lbp_top_descriptor = np.concatenate([hist_xy, hist_xt, hist_yt])
    return lbp_top_descriptor

# Example usage
if __name__ == "__main__":
    # Path to the directory containing frames
    frame_directory = "landmark/trial_lie_014_aligned"

    # Load frames from directory
    frames = []
    for filename in sorted(os.listdir(frame_directory)):
        if filename.endswith(".bmp") or filename.endswith(".jpg"):
            frame_path = os.path.join(frame_directory, filename)
            frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            frames.append(frame)

    video_frames = np.array(frames)

    # Compute LBP-TOP features
    lbp_top_features = lbp_top(video_frames)
    print("LBP-TOP Descriptor:", lbp_top_features)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(lbp_top_features)), lbp_top_features, width=1.0)
    plt.title("Concatenated LBP-TOP Descriptor Histogram")
    plt.xlabel("LBP Code Index")
    plt.ylabel("Frequency")
    plt.show()
