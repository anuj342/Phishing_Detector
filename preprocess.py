#can further integrate, this file is not used anywhere, the deepfake feature is an ongoing process
# preprocess.py
import cv2
import os
import random
from mtcnn.mtcnn import MTCNN

# Initialize the MTCNN detector
detector = MTCNN()

# Define paths
video_source_path = 'datasets/videos/'
face_output_path = 'datasets/faces/'

# Create output directories if they don't exist
os.makedirs(os.path.join(face_output_path, 'real'), exist_ok=True)
os.makedirs(os.path.join(face_output_path, 'fake'), exist_ok=True)

def extract_faces_from_video(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_name = os.path.basename(video_path).split('.')[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 15th frame to save time
        if frame_count % 15 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(frame_rgb)

            if results:
                x1, y1, width, height = results[0]['box']
                # Add a small margin
                x1, y1 = max(0, x1 - int(width * 0.15)), max(0, y1 - int(height * 0.15))
                x2, y2 = x1 + int(width * 1.3), y1 + int(height * 1.3)
                
                face = frame[y1:y2, x1:x2]

                if face.size != 0:
                    face_filename = f"{video_name}_frame_{frame_count}.jpg"
                    cv2.imwrite(os.path.join(output_folder, face_filename), face)
        
        frame_count += 1

    cap.release()
    print(f"Finished processing {video_path}")

# 1. Extract faces from all videos
print("--- Starting Face Extraction ---")
for label_type in ['real', 'fake']:
    folder_path = os.path.join(video_source_path, label_type)
    output_dir = os.path.join(face_output_path, label_type)
    if not os.path.exists(folder_path):
        print(f"Warning: Directory not found at {folder_path}")
        continue
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_file_path = os.path.join(folder_path, filename)
            extract_faces_from_video(video_file_path, output_dir)

# 2. Balance the dataset by down-sampling
print("\n--- Balancing Dataset ---")
real_faces_dir = os.path.join(face_output_path, 'real')
fake_faces_dir = os.path.join(face_output_path, 'fake')

real_files = os.listdir(real_faces_dir)
fake_files = os.listdir(fake_faces_dir)

if len(fake_files) > len(real_files):
    print(f"Found {len(fake_files)} fake images and {len(real_files)} real images.")
    files_to_remove = random.sample(fake_files, len(fake_files) - len(real_files))
    for file_name in files_to_remove:
        os.remove(os.path.join(fake_faces_dir, file_name))
    print(f"Removed {len(files_to_remove)} fake images to balance the dataset.")
elif len(real_files) > len(fake_files):
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images.")
    files_to_remove = random.sample(real_files, len(real_files) - len(fake_files))
    for file_name in files_to_remove:
        os.remove(os.path.join(real_faces_dir, file_name))
    print(f"Removed {len(files_to_remove)} real images to balance the dataset.")
else:
    print("Dataset is already balanced.")

print("\n--- Data Preprocessing Complete! ---")
