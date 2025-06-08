import os
from datetime import datetime
import eventlet
import logging

from object_detection import cut_to_object, get_main_object

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


def scan_text(img):
    t1 = cv.getTickCount()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #blur = cv.GaussianBlur(gray,(5,5),0)

    denoise = cv.fastNlMeansDenoising(gray)

    thresh = cv.adaptiveThreshold(denoise,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #_,otsuThresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

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

        # load image into buffer properly
        file.stream.seek(0)
        buffer = file.stream.read()

        # buffer >> Mat
        arr = np.asarray(bytearray(buffer), dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)

        #save initial img
        cv.imwrite("upload.png", img)

        # process to find edges and contours
        f_file,text = "",""
        pos_objs = get_main_object(img)
        cv.drawContours(img, pos_objs, -1, (0,255,0), 2)
        for obj in pos_objs:
            cut_contour = cut_to_object(img, obj)
            if cut_contour is None:
                print("cut contour unable to be detected")
                continue
            else:
                #for now, just run loop until you find an obj that works
                f_file, text = scan_text(img)
                break


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
