from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('input/08fd33_4.mp4', save=True)