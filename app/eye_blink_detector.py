import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def detect_eye_blinks(frame, show_landmarks=True):
    # Configuration parameters
    EAR_THRESHOLD = 0.2  # Adjusted threshold for determining blinks
    BLINK_TIME_THRESHOLD = 1.0  # Maximum time between blinks (in seconds)
    BLINK_FRAMES_THRESHOLD = 2  # Minimum number of frames for a blink

    # Variables for blink detection
    blink_counter = 0
    blink_start_time = None
    last_blink_time = None
    current_status = "No Blink"
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

                # Blink detection logic
                if smoothed_ear < EAR_THRESHOLD:
                    if blink_start_time is None:
                        blink_start_time = time.time()
                        current_status = "Blinking"
                else:
                    if blink_start_time is not None:
                        blink_duration = time.time() - blink_start_time
                        if blink_duration >= BLINK_FRAMES_THRESHOLD * (1/30):  # Assuming 30 FPS
                            blink_counter += 1
                            current_status = f"Blink Detected ({blink_counter})"
                            if last_blink_time is not None and time.time() - last_blink_time <= BLINK_TIME_THRESHOLD:
                                current_status = f"Double Blink Detected ({blink_counter})"
                            last_blink_time = time.time()
                        blink_start_time = None

                # Display EAR value and blink count
                cv2.putText(frame_bgr, f"EAR: {smoothed_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Blinks: {blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Status: {current_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display blink status on the frame
        cv2.putText(frame_bgr, f"Status: {current_status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return current_status, frame_bgr  # Return status and processed frame
