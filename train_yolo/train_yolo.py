from ultralytics import YOLO, checks, hub
checks()

hub.login('0284d16a9b4a61598d22c2ac30f0a18b1c62f5762e')

model = YOLO('https://hub.ultralytics.com/models/nZIyzZFsMkkSq0uQHrMs')
results = model.train()