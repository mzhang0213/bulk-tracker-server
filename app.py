import os

import numpy as np
from flask import Flask, render_template
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

@socketio.on('message')
def handle_message(data):
    print('received message:', data)
    emit('response', {'data': 'Message received'})


@app.get('/test')
def test_get():
    print("received")
    return str(int(request.args.get("hi")))

@app.route("/calorie-image",methods=["POST"])
def calorie_image_upload():
    print("received")
    print(request.files["image"])
    if "image" not in request.files:
        return {"message":"no file submitted"}
    file = request.files["image"]
    if file:
        #file.save("./upload.png")
        file.stream.seek(0)
        buffer = file.stream.read()
        text = scan_text(buffer)
        emit("text", {"data": text})
        return {"message":text}


if __name__ == '__main__':
    socketio.run(app)

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

    orb = cv.ORB().create()

    kp = orb.detect(thresh,None)

    kp, des = orb.compute(thresh, kp)

    #FIGURE A BETTER BOX DETECTOR THAN ORB

    img2 = cv.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)

    cv.imwrite("keypoints.png", img2)

    #cv.imshow("pts",img2)

    final = thresh
    cv.imwrite('final_frame.png', final)
    print("finished")
    print("elapsed: "+str((cv.getTickCount()-t1)/cv.getTickFrequency()))
    return pytesseract.image_to_string(final)
