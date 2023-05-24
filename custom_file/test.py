import torch
import cv2
import os

home = os.path.expanduser("~")
print(home)
model = torch.hub.load('/home/xaviernx01/yolov5', 'custom', path='../yolov5s_tosro.pt', source='local') 
# Image
img = '../data/images/bus.jpg'

#vid
vid = cv2.VideoCapture(0)

while(True) :

	ret,frame = vid.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# Inference
	results = model(frame)
	#results = model(img)
	# Results, change the flowing to: results.show()
	# results.print()
	# print(results.pandas())  # or .show(), .save(), .crop(), .pandas(), etc
	print(results.pandas().xyxy[0])
	print("============================================================")
