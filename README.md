python -m venv venv
source venv/bin/activate

pip install fastapi uvicorn opencv-python-headless mediapipe

uvicorn app.main:app --reload

https://docs.render.com/deploy-fastapi
