import cv2
import time
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from ultralytics import YOLO
from threading import Thread
import uvicorn

app = FastAPI()

# Load YOLO models /home/skkr/Documents/AgriAI/sequential/models/pest.pt
model_paths = [r"/home/skkr/Documents/AgriAI/sequential/models/rice.pt", r"/home/skkr/Documents/AgriAI/sequential/models/sugarcane.pt", r"/home/skkr/Documents/AgriAI/sequential/models/wheat.pt", r"/home/skkr/Documents/AgriAI/sequential/models/pest.pt"]
models = [YOLO(model) for model in model_paths]

last_result = {"model": None, "detections": [], "image": None}

def capture_image():
    """ Captures an image from the webcam and saves it. """
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite("captured.jpg", frame)
        return "captured.jpg"
    return None

def draw_bounding_boxes(image_path, detections):
    """ Draw bounding boxes on detected objects. """
    image = cv2.imread(image_path)

    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        class_id = det["class"]
        confidence = det["confidence"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class {class_id} ({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("processed.jpg", image)
    return "processed.jpg"

def run_models():
    """ Runs YOLO models sequentially every 45 seconds. """
    global last_result
    model_index = 0

    while True:
        img_path = capture_image()
        if img_path:
            model = models[model_index]
            results = model(img_path)

            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        "class": int(box.cls),
                        "confidence": float(box.conf),
                        "bbox": box.xyxy[0].tolist()
                    })

            processed_img_path = draw_bounding_boxes(img_path, detections)

            last_result = {
                "model": model_paths[model_index],
                "detections": detections,
                "image": processed_img_path
            }

        model_index = (model_index + 1) % len(models)
        time.sleep(5)

@app.get("/get_results")
def get_results():
    """ API to fetch the latest detection results. """
    return last_result

@app.get("/get_image")
def get_image():
    """ API to fetch the processed image with bounding boxes. """
    return FileResponse("processed.jpg", media_type="image/jpeg")

if __name__ == "__main__":
    thread = Thread(target=run_models, daemon=True)
    thread.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
