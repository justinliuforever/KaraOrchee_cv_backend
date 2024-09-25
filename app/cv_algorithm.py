import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

mouth_open_threshold = 0.35

def calculate_mouth_aspect_ratio(landmarks, indices):
    left_point = landmarks[indices[0]]
    right_point = landmarks[indices[1]]
    top_mid_point = landmarks[indices[2]]
    bottom_mid_point = landmarks[indices[3]]

    vertical_dist = ((top_mid_point.x - bottom_mid_point.x) ** 2 + 
                     (top_mid_point.y - bottom_mid_point.y) ** 2) ** 0.5
    
    horizontal_dist = ((left_point.x - right_point.x) ** 2 + 
                       (left_point.y - right_point.y) ** 2) ** 0.5
    
    return vertical_dist / horizontal_dist

def process_frame(frame):
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,        
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        status = "No Face Detected"
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
                
                MOUTH = [61, 291, 13, 14]
                mar = calculate_mouth_aspect_ratio(face_landmarks.landmark, MOUTH)

                if mar > mouth_open_threshold:
                    status = "Mouth Open"
                else:
                    status = "Mouth Closed"
        
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        frame_encoded = buffer.tobytes()
        return status, frame_encoded
