from ultralytics import YOLO

def detect_yolov8(source_path, image_width, model_path="/home/jiaxi/cs525/Assets/models/yolov8n.pt", save_result=False):
    model = YOLO(model_path)
    results = model(source_path, imgsz=image_width, save=save_result)

if __name__=="__main__":
    detect_yolov8("/home/jiaxi/cs525/Assets/120_1K", 640)
