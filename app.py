import os
from datetime import datetime
import eventlet
import logging
eventlet.monkey_patch()
logging.basicConfig(level=logging.DEBUG)

import numpy as np
from flask import Flask, render_template, send_from_directory
from flask import request
from flask_socketio import SocketIO, emit

import cv2 as cv
import pytesseract

#run deployment cmd: gunicorn -w 4 -b 0.0.0.0 "app:app"
#run tunnel: cloudflared tunnel run bulk-tunnel
app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

@app.route("/uploads/<path:filename>")
def show_image(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

@socketio.on('message')
def handle_message(data):
    logging.info('received message:', data)
    emit('response', {'data': 'Message received'})


@app.get('/test')
def test_get():
    logging.info("received")
    return str(int(request.args.get("hi")))


def scan_text(b):
    t1 = cv.getTickCount()

    arr = np.asarray(bytearray(b), dtype=np.uint8)

    frame = cv.imdecode(arr, cv.IMREAD_COLOR)

    #save and make sure
    cv.imwrite("upload.png", frame)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    #blur = cv.GaussianBlur(gray,(5,5),0)

    denoise = cv.fastNlMeansDenoising(gray)

    thresh = cv.adaptiveThreshold(denoise,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #_,otsuThresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    #orb = cv.ORB().create()

    #kp = orb.detect(thresh,None)

    #kp, des = orb.compute(thresh, kp)

    #FIGURE A BETTER BOX DETECTOR THAN ORB

    #img2 = cv.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)

    #cv.imwrite("keypoints.png", img2)

    #cv.imshow("pts",img2)

    final = thresh
    rn = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_file = f"scan-processed-{rn}.png"
    cv.imwrite(os.path.join(UPLOAD_FOLDER,final_file), final)

    logging.info("finished")
    logging.info("elapsed: "+str((cv.getTickCount()-t1)/cv.getTickFrequency()))

    return final_file,pytesseract.image_to_string(final)

@app.route("/api/calorie-image",methods=["POST"])
def calorie_image_upload():
    logging.info("received")
    logging.info(request.files["image"])
    if "image" not in request.files:
        return {"message":"no file submitted"}
    file = request.files["image"]
    if file:
        rn = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_file = f"scan-{rn}.png"
        file.save(os.path.join(UPLOAD_FOLDER,scan_file))

        file.stream.seek(0)
        buffer = file.stream.read()
        f_file, text = scan_text(buffer)

        socketio.emit("scan_result", {
            "result": text,
            "scan_file": scan_file,
            "final_file": f_file
        })
        return {"message":text}


if __name__ == '__main__':
    working_port = 3005
    logging.info(f"server running at port {working_port}")
    socketio.run(app, host='0.0.0.0', port=working_port)
