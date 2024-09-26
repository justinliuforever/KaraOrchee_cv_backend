import cv2
import mediapipe as mp
import time 

# Initialize MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def detect_eye_state(frame, show_landmarks=True):
    # Configuration parameters
    EAR_THRESHOLD = 0.15  # Adjusted threshold for determining if eyes are open or closed

    # Variables for eye state detection
    current_status = "Eyes Open"
    ear_history = []

    def calculate_eye_aspect_ratio(landmarks):
        left_eye_indices = [33, 160, 158, 133, 153, 144]  # Left eye landmarks
        right_eye_indices = [362, 385, 387, 263, 373, 380]  # Right eye landmarks

        def eye_aspect_ratio(eye_indices):
            # Vertical eye landmarks
            v1 = landmarks[eye_indices[1]].y - landmarks[eye_indices[5]].y
            v2 = landmarks[eye_indices[2]].y - landmarks[eye_indices[4]].y

            # Horizontal eye landmarks
            h = landmarks[eye_indices[0]].x - landmarks[eye_indices[3]].x

            # Calculate EAR
            ear = (v1 + v2) / (2.0 * h)
            return max(ear, 0)  # Ensure non-negative value

        left_ear = eye_aspect_ratio(left_eye_indices)
        right_ear = eye_aspect_ratio(right_eye_indices)

        return (left_ear + right_ear) / 2  # Return average EAR

    # Create a FaceMesh object
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,        
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with FaceMesh
        results = face_mesh.process(frame_rgb)

        # Convert back to BGR for rendering with OpenCV
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Detect and process face landmarks
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if show_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame_bgr,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
                
                # Calculate EAR for both eyes
                ear = calculate_eye_aspect_ratio(face_landmarks.landmark)
                
                # EAR smoothing
                ear_history.append(ear)
                if len(ear_history) > 5:  # Keep the last 5 EAR values
                    ear_history.pop(0)
                smoothed_ear = sum(ear_history) / len(ear_history)

                # Determine eye state
                if smoothed_ear < EAR_THRESHOLD:
                    current_status = "Eyes Closed"
                else:
                    current_status = "Eyes Open"

                # Display EAR value for debugging
                cv2.putText(frame_bgr, f"EAR: {smoothed_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display eye state on the frame
        cv2.putText(frame_bgr, f"Status: {current_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return current_status, frame_bgr  # Return status and processed frame
