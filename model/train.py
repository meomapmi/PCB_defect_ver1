from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="/content/pcb.yaml",
    epochs=50,
    imgsz=600,
    batch=16,
    patience=15,
    project="pcb_detection",
    name="yolov8_baseline"
)

metrics = model.val(
    data="/content/pcb.yaml",
    split="test"
)

precision = metrics.box.p
recall = metrics.box.r
map50 = metrics.box.map50
map5095 = metrics.box.map

f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
fpr = 1 - precision

print("Precision:", precision)
print("Recall:", recall)
print("mAP50:", map50)
print("mAP50-95:", map5095)
print("F1-score:", f1)
print("False Positive Rate:", fpr)