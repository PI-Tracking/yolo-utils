import cv2
import json
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
    return frame_count

def annotate_frames(json_path, frames_folder, output_folder_txt, output_folder_img, num_frames):
    with open(json_path) as f:
        data = json.load(f)
        annotations = data.get('annotations', data) 

    frame_annotations = {i: [] for i in range(num_frames)}
    
    for annotation in annotations:
        frame_number = annotation['image_id'] - 1 
        frame_annotations[frame_number].append(annotation)
    
    for frame_number in range(num_frames):
        frame_path = os.path.join(frames_folder, f"frame_{frame_number}.jpg")
        frame = cv2.imread(frame_path)
        annotation_file_path = os.path.join(output_folder_txt, f"frame_{frame_number}.txt")
        
        if frame is not None:
            try:
                if frame_annotations[frame_number]:
                    for annotation in frame_annotations[frame_number]:
                        bbox = annotation['bbox']
                        category_id = 0  
                        x, y, width, height = map(int, bbox)
                        
                        frame_height, frame_width = frame.shape[:2]
                        center_x = (x + width / 2) / frame_width
                        center_y = (y + height / 2) / frame_height
                        norm_width = width / frame_width
                        norm_height = height / frame_height

                        annotation_str = f"{category_id} {center_x:.16f} {center_y:.16f} {norm_width:.16f} {norm_height:.16f}\n"
                        with open(annotation_file_path, 'a') as annotation_file:
                            annotation_file.write(annotation_str)

                        start_point = (x, y)
                        end_point = (x + width, y + height)
                        color = (0, 255, 0)
                        thickness = 2
                        cv2.rectangle(frame, start_point, end_point, color, thickness)
                    
                    annotated_frame_path = os.path.join(output_folder_img, f"annotated_frame_{frame_number}.jpg")
                    cv2.imwrite(annotated_frame_path, frame)
                else:
                    open(annotation_file_path, 'w').close()

            except Exception as e:
                print(f"Error processing annotation for frame {frame_number}: {e}")
        else:
            print(f"Frame {frame_number} not found.")

video_path = 'weapon-video.mp4'
json_path = 'json.json'
frames_folder = 'imgs/frames'
output_folder_txt = 'imgs/annotations_txt'
output_folder_img = 'imgs/annotated_images'

os.makedirs(frames_folder, exist_ok=True)
os.makedirs(output_folder_txt, exist_ok=True)
os.makedirs(output_folder_img, exist_ok=True)

num_frames = extract_frames(video_path, frames_folder)
print(f"Extracted {num_frames} frames from the video.")

annotate_frames(json_path, frames_folder, output_folder_txt, output_folder_img, num_frames)
print("Annotations saved in .txt files and annotated frames saved as images.")
