from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from .cv_algorithm import process_frame
from starlette.websockets import WebSocketState

app = FastAPI()

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    status, _ = process_frame(img)
    return JSONResponse(content={"status": status})

@app.websocket("/ws/video")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            status, frame_encoded = process_frame(frame)
            await websocket.send_json({"status": status, "frame": frame_encoded.hex()})
    except WebSocketDisconnect:
        print("WebSocket connection closed")
    except Exception as e:
        print(f"Connection closed with error: {e}")
    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()
