import os
from datetime import datetime
import eventlet
import logging

eventlet.monkey_patch()
logging.basicConfig(level=logging.DEBUG)

from flask import Flask, render_template, send_from_directory
from flask import request
from flask_socketio import SocketIO, emit

from object_detection import get_main_objects, get_valid_contours, scan_text
from utils import UPLOAD_FOLDER
import numpy as np
import cv2 as cv

#run deployment cmd: gunicorn -w 4 -b 0.0.0.0 "app:app"
#run tunnel: cloudflared tunnel run bulk-tunnel
#run docker build: docker-compose up --build
app = Flask(__name__)
socketio = SocketIO(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


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


@app.route("/api/calorie-image",methods=["POST"])
def calorie_image_upload():
    output_files = []
    logging.info("received")
    logging.info(request.files["image"])
    if "image" not in request.files:
        return {"message":"no file submitted"}
    file = request.files["image"]
    if file:
        #save initial img
        #cv.imwrite("upload.png", img)
        rn = datetime.now().strftime("%Y%m%d_%H%M%S")
        scan_file = f"scan-{rn}.png"
        file.save(os.path.join(UPLOAD_FOLDER,scan_file))
        output_files.append({
            "name":scan_file,
            "scan_type":"original"
        })

        # load image into buffer properly
        file.stream.seek(0)
        buffer = file.stream.read()

        # buffer >> Mat
        arr = np.asarray(bytearray(buffer), dtype=np.uint8)
        img = cv.imdecode(arr, cv.IMREAD_COLOR)

        # process to find edges and contours
        t1 = cv.getTickCount() #for processing time

        f_file,text = "",""
        pos_objs, out1 = get_main_objects(img)
        for outfile in out1: output_files.append(outfile)
        cv.drawContours(img, pos_objs, -1, (0,255,0), 2)

        #IN THE FUTURE make this so that the object with most process text (that isnt whitespace) is dominant in being the final processed image
        #IN THE FUTURE also include columns scanned
        valid_contours, out_vcontours = get_valid_contours(img,pos_objs)
        for outfile in out_vcontours: output_files.append(outfile)
        for vcont in valid_contours:
            res,out2 = scan_text(vcont)
            for outfile in out2: output_files.append(outfile)
            for line in res:
                print(line)
        print("elapsed: "+str((cv.getTickCount()-t1)/cv.getTickFrequency()))

        socketio.emit("scan_result", {
            "result": text,
            "files":output_files
        })
        return {"message":text}




if __name__ == '__main__':
    working_port = 3005
    logging.info(f"server running at port {working_port}")
    socketio.run(app, host='0.0.0.0', port=working_port)
