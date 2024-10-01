import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

def detect_head_pose(frame):
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 定义2D图像点
                image_points = np.array([
                    (face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]),   # 鼻尖
                    (face_landmarks.landmark[152].x * frame.shape[1], face_landmarks.landmark[152].y * frame.shape[0]), # 下巴
                    (face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]), # 右眼右角
                    (face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]), # 左眼左角
                    (face_landmarks.landmark[287].x * frame.shape[1], face_landmarks.landmark[287].y * frame.shape[0]), # 右嘴角
                    (face_landmarks.landmark[57].x * frame.shape[1], face_landmarks.landmark[57].y * frame.shape[0])  # 左嘴角
                ], dtype="double")

                # 3D模型点
                model_points = np.array([
                    (0.0, 0.0, 0.0),             # 鼻尖
                    (0.0, -63.6, -12.5),         # 下巴
                    (43.3, 32.7, -26.0),         # 右眼右角
                    (-43.3, 32.7, -26.0),        # 左眼左角
                    (28.9, -28.9, -24.1),        # 右嘴角
                    (-28.9, -28.9, -24.1)        # 左嘴角
                ])

                # 相机参数
                focal_length = frame.shape[1]
                center = (frame.shape[1] / 2, frame.shape[0] / 2)
                camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype="double")

                dist_coeffs = np.zeros((4,1))  # 假设没有畸变
                success, rotation_vector, translation_vector = cv2.solvePnP(
                    model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                # 将旋转向量转换为欧拉角
                rotation_mat, _ = cv2.Rodrigues(rotation_vector)
                pose_mat = cv2.hconcat((rotation_mat, translation_vector))
                _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(pose_mat)
                pitch, yaw, roll = eulerAngles.flatten()

                # 显示角度信息
                # cv2.putText(frame_bgr, f"Pitch: {pitch:.2f}", (10, 30),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame_bgr, f"Yaw: {yaw:.2f}", (10, 60),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame_bgr, f"Roll: {roll:.2f}", (10, 90),
                #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 根据Pitch角度判断是否触发事件
                if pitch > 10:  # 当头部抬起超过10度
                    status = "Music Play"
                else:
                    status = "Music Pause"

                return status, frame_bgr, pitch, yaw, roll

        return "No Face Detected", frame_bgr, None, None, None
