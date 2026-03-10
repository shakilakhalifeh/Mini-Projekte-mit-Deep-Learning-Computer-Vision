from ultralytics import YOLO

# load pretrained model
model = YOLO("yolov8n.pt")

# run object detection
results = model("drone_image.jpg", save=True)

print("Detection completed")
