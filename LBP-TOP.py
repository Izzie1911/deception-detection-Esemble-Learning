import numpy as np
import cv2
import os
from skimage.feature import local_binary_pattern


def compute_lbp(image, radius=1, n_points=8):
    return local_binary_pattern(image, n_points, radius, method="uniform")


def lbp_top(video_frames, radius=1, n_points=8):
    num_frames, height, width = video_frames.shape

    hist_xy = np.zeros((256,), dtype=np.float32)
    hist_xt = np.zeros((256,), dtype=np.float32)
    hist_yt = np.zeros((256,), dtype=np.float32)

    # XY Plane
    for frame_idx in range(num_frames):
        lbp_xy = compute_lbp(video_frames[frame_idx], radius, n_points)
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


if __name__ == "__main__":
    # Thư mục cha chứa nhiều thư mục video
    parent_directory = "landmark/trial_lie_001_aligned"

    lbp_top_sum = None
    video_count = 0

    # Duyệt qua mỗi thư mục con (mỗi thư mục con là một video)
    for video_folder in os.listdir(parent_directory):
        video_path = os.path.join(parent_directory, video_folder)
        if os.path.isdir(video_path):
            # Đọc tất cả frames trong thư mục video_path
            frames = []
            for filename in sorted(os.listdir(video_path)):
                if filename.lower().endswith(('.bmp', '.jpg', '.png')):
                    frame_path = os.path.join(video_path, filename)
                    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        frames.append(frame)

            if len(frames) == 0:
                continue

            video_frames = np.array(frames)
            # Tính LBP-TOP cho video này
            lbp_top_features = lbp_top(video_frames)

            # Cộng dồn vào sum
            if lbp_top_sum is None:
                lbp_top_sum = np.zeros_like(lbp_top_features, dtype=np.float64)
            lbp_top_sum += lbp_top_features
            video_count += 1

    if video_count > 0:
        avg_lbp_top = lbp_top_sum / video_count
        print("Số lượng video:", video_count)
        print("Kích thước vector LBP-TOP trung bình:", avg_lbp_top.shape)
        print("Vector LBP-TOP trung bình:", avg_lbp_top)
    else:
        print("Không tìm thấy video hợp lệ trong thư mục.")
