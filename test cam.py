import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

model = YOLO('yolov8n.pt')  #importer le model
print(model.names)
webcamera = cv2.VideoCapture(0)  # capturer l'image 
 

while True:
    success, frame = webcamera.read()  # si la capture de l'image a  rréussi 
    if not success or frame is None:
        print("impossible de lire la camera")
        continue
    results = model.track(frame, classes=[0,39], conf=0.8, imgsz=480)
    annotator = Annotator(frame, example=model.names)
    for i, box in enumerate(results[0].boxes.xyxy.cpu()):
            width,height,area=annotator.get_bbox_dimension(box)
            print(f"baunding box   : width: {width.item():.2f},|height: {height.item():.2f} , |area:   {area.item():.2f}")
            print(type(width.item()))
    cv2.putText(frame, f"Total: {len(results[0].boxes)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 2, cv2.LINE_AA)
    cv2.imshow("Live Camera", results[0].plot())
    if cv2.waitKey(1) == ord('q'):
     break

webcamera.release()
cv2.destroyAllWindows()