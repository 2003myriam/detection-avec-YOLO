import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from flask import Flask, render_template, Response
import time 
import test_moteur as motor

# Charger le modèle
model = YOLO('yolov8n.pt')
app = Flask(__name__)
webcamera =cv2.VideoCapture(0)
if not webcamera.isOpened():
    raise RuntimeError(" Impossible d’ouvrir la caméra /dev/video0")


def gen_frames():
    
    while True:
        success, frame = webcamera.read()
        if not success:
            # Si la vidéo est terminée, recommence au début
            webcamera.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Détection avec YOLO
        results = model.predict(frame, classes=[0, 39], conf=0.8, imgsz=480,verbose=False)

        # Annotateur pour calculer largeur/hauteur/aire des BBox
        annotator = Annotator(frame, example=model.names)
        for i, box in enumerate(results[0].boxes.xyxy.cpu()):
            width, height, area = annotator.get_bbox_dimension(box)
            print(f"Bounding box : width={width.item():.2f}, height={height.item():.2f}, area={area.item():.2f}")
            """print(type(width.item()))
            if width<100:
               print("condition activé")
               motor.forward()
            else:
               motor.stop()"""
    

        # Compteur d’objets détectés
        cv2.putText(frame, f"Total: {len(results[0].boxes)}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 155), 2, cv2.LINE_AA)
        time.sleep(0.03)

         
        ret, buffer = cv2.imencode('.jpg', results[0].plot())
        if not ret :
            continue
        frame_bytes = buffer.tobytes()
        print("frame envoyée")
        

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

 
 

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
      <head><title>Flux YOLO threadé</title></head>
      <style>
      body {
        background-color: #14213d;
      }
      h1 {
        color: #ffff03;
        text-align: center;
      }
      p {
        display: flex;
        align-items: flex-end;
        height: 200px;
        text-align: center;
        font-size: 25px;
        color: #ffff;
      }
      img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 600px;
        height: 600px;
      }
      </style>
      <body>
        <h1>Robot Vision : Détection et evitement d'obstacles en temps réel</h1>
        <p>Ce projet consiste a développer un robot intelligent capable d'éviter les obstacles en utilisant une caméra embarqée et un raspberry Pi. Grace au modele de détection YOLO , le robot analyse son environnement en temps réel et adaptes ses mouvements. <br> L'application web associée permet de visualiser en direct le flux vidéo de la caméra ,avec les objets détectés et les zones de danger mises en évidence.</p>
        <img src="/video_feed" width="640" height="480">
      </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001 , threaded=True,debug=False)



 
