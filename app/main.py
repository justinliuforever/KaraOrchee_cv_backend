from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from .mouth_state_detector import process_frame
from .eye_state_detector import detect_eye_state  # Import the new eye state detection function
from .eye_blink_detector import detect_eye_blinks  # Import the new eye blink detection function
from .head_pose_detector import detect_head_pose  # 导入头部姿态检测函数
from starlette.websockets import WebSocketState
import json

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    status, _ = process_frame(img)
    return JSONResponse(content={"status": status})

@app.websocket("/ws/mouth-state")
async def mouth_state_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            status, frame_encoded = process_frame(frame)
            await websocket.send_json({"status": status, "frame": frame_encoded.hex()})
    except WebSocketDisconnect:
        print("Mouth state WebSocket connection closed")
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.websocket("/ws/eye-state")
async def eye_state_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            frame_data = parsed_data['frame']
            show_landmarks = parsed_data['showLandmarks']
            
            nparr = np.array(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            status, processed_frame = detect_eye_state(frame, show_landmarks)
            _, frame_encoded = cv2.imencode('.jpg', processed_frame)
            
            await websocket.send_json({"status": status, "frame": frame_encoded.tobytes().hex()})
    except WebSocketDisconnect:
        print("Eye state WebSocket connection closed")
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.websocket("/ws/eye-blink")
async def eye_blink_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            frame_data = parsed_data['frame']
            show_landmarks = parsed_data['showLandmarks']
            
            nparr = np.array(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            status, processed_frame, double_blink_detected = detect_eye_blinks(frame, show_landmarks)
            _, frame_encoded = cv2.imencode('.jpg', processed_frame)
            
            response_data = {
                "status": status,
                "frame": frame_encoded.tobytes().hex(),
                "doubleBlinkDetected": double_blink_detected
            }
            
            await websocket.send_json(response_data)
            
            if double_blink_detected:
                # Trigger your action here, e.g., play/stop music
                print("Double blink detected! Trigger action.")
                # You can add your action logic here or send a separate message to the frontend
    except WebSocketDisconnect:
        print("Eye blink WebSocket connection closed")
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()

@app.websocket("/ws/head-pose")
async def head_pose_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            parsed_data = json.loads(data)
            frame_data = parsed_data['frame']

            nparr = np.frombuffer(bytes.fromhex(frame_data), dtype=np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            status, processed_frame, pitch, yaw, roll = detect_head_pose(frame)
            _, frame_encoded = cv2.imencode('.jpg', processed_frame)

            await websocket.send_json({
                "status": status,
                "frame": frame_encoded.tobytes().hex(),
                "pitch": pitch,
                "yaw": yaw,
                "roll": roll
            })
    except WebSocketDisconnect:
        print("Head pose WebSocket connection closed")
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
