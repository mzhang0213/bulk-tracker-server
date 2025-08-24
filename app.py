import eventlet
eventlet.monkey_patch()

import logging
logging.basicConfig(level=logging.DEBUG)
import os
from datetime import datetime



from object_detection import get_main_objects, get_valid_contours, scan_text
from utils import UPLOAD_FOLDER, uprint

from flask import Flask, send_from_directory, request, jsonify
from flask_socketio import SocketIO, emit
import numpy as np
import cv2 as cv
from openai import OpenAI
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text

#run deployment cmd: gunicorn -w 4 -b 0.0.0.0 "app:app"
#run tunnel: cloudflared tunnel run bulk-tunnel
#run docker build: docker-compose up --build
app = Flask(__name__)
#socketio = SocketIO(app)

app.config['SQLALCHEMY_DATABASE_URI'] = f'postgresql://{os.environ.get("POSTGRES_USER")}:{os.environ.get("POSTGRES_PASSWORD")}@{os.environ.get("POSTGRES_ENDPOINT")}:{os.environ.get("POSTGRES_PORT")}/{os.environ.get("POSTGRES_DB")}'
db = SQLAlchemy(app)

@app.route("/health")
def health_check():
    return "OK"

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/uploads/<path:filename>")
def show_image(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)


#@socketio.on('message')
#def handle_message(data):
#    logging.info('received message:', data)
#    emit('response', {'data': 'Message received'})


@app.get('/test')
def test_get():
    logging.info("received")
    return str(int(request.args.get("hi")))

'''
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
        pos_objs, out_main_obj = get_main_objects(img)
        for outfile in out_main_obj: output_files.append(outfile)
        #cv.drawContours(img, pos_objs, -1, (0,255,0), 2)
        uprint("\nfinished getting main objects\n")

        if pos_objs is None:
            socketio.emit("scan_result", {
                "result": "Failed when obtaining main objects",
                "files":output_files
            })
            return {"message":"Failed when obtaining main objects"}

        valid_contours, out_vcontours = get_valid_contours(img,pos_objs)
        for outfile in out_vcontours: output_files.append(outfile)
        uprint("\nfinished getting valid contours\n")

        if valid_contours is None:
            socketio.emit("scan_result", {
                "result": "Failed when obtaining valid contours",
                "files":output_files
            })
            return {"message": "Failed when obtaining valid contours"}


        for vcont in valid_contours:
            res,out_scan_text = scan_text(vcont)
            for outfile in out_scan_text: output_files.append(outfile)

            if res is None: continue

            for line in res:
                uprint(line)
                text+=line+"\n"

        uprint("total elapsed: "+str((cv.getTickCount()-t1)/cv.getTickFrequency()))

        socketio.emit("scan_result", {
            "result": text,
            "files":output_files
        })
        return {"message":text}

        #IN THE FUTURE make this so that the object with most process text (that isnt whitespace) is dominant in being the final processed image
        #IN THE FUTURE also include columns scanned
'''


def get_gpt_message(gpt_msg, res_format, img64, app_context):
    messages = []
    user_content = [gpt_msg]

    if app_context is not None:
        messages.append({"role": "system", "content": app_context})
    if img64 is not None:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img64}"}
            }
        )

    messages.append(user_content)
    chat_message = {
        "messages": messages,
        "response_format":{"type":res_format}
    }
    return chat_message

def send_gpt_message(gpt_msg):
    if gpt_msg is None:
        return jsonify({
            "error":"No messages provided"
        })

    client = OpenAI(api_key=os.getenv("LLM_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=gpt_msg["messages"],
        temperature=0.5,
        response_format=gpt_msg["response_format"]
    )

    return response.choices[0].message.content

@app.route("/api/draftMessage-progress", methods=["POST"])
def draft_message():
    user_message = request.json.get("message")
    app_context = request.json.get("context")
    img_b64 = request.json.get("image")

    drafted_message = get_gpt_message(user_message,"json",img_b64,app_context)
    logging.info(drafted_message)

    return jsonify(drafted_message)

@app.route("/api/sendMessage-progress", methods=["POST"])
def send_message_progress():
    img_b64 = request.json.get("image")
    '''
    drafts
    Grade muscles 0–1 from image. Concise.
    '''
    '''
    {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{image_base64}"
        },
    },
    '''

    gpt_msg_input = get_gpt_message("Grade muscles 0–1 from image. Provide new workout plan based on old. Format:{response: ..., progress: %, plan: []}", "json", img_b64,None)

    return jsonify(send_gpt_message(gpt_msg_input))

@app.route('/api/sendMessage', methods=["POST"])
def chat():
    gpt_msg = request.json.get("chat_message")
    res_format = request.json.get("response_format")
    img_b64 = request.json.get("image")
    app_context = request.json.get("context")

    return jsonify(send_gpt_message(get_gpt_message(gpt_msg,res_format,img_b64,app_context)))



# routing up api calls to the food apis





if __name__ == '__main__':
    working_port = 3005
    logging.info(f"server running at port {working_port}")
    #socketio.run(app, host='0.0.0.0', port=working_port)
