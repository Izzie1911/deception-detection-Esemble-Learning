import cv2
from facenet_pytorch import MTCNN
import torch
import os

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load video file
video_path = "trial_lie_002.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Process video frame by frame
frame_count = 0
face_save_path = "faces"  # Directory to save cropped faces

# Ensure the save directory exists
if not os.path.exists(face_save_path):
    os.makedirs(face_save_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Convert frame to RGB (MTCNN expects RGB input)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    boxes, probs = mtcnn.detect(frame_rgb, landmarks=False)

    # Extract and save detected faces
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]

            # Ensure the box coordinates are within the frame dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop the face region
            face = frame[y1:y2, x1:x2]

            # Save the cropped face
            face_filename = os.path.join(face_save_path, f"frame_{frame_count:04d}_face_{i}.jpg")
            cv2.imwrite(face_filename, face)

    # Optional: Display the frame with detected boxes (without masking)
    # cv2.imshow("Video Frame", frame)

    # Press 'q' to quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Total frames processed: {frame_count}")
print(f"Faces saved in directory: {face_save_path}")
