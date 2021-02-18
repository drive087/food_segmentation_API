import os
import cv2
import io
from flask import Flask, flash, request, redirect, url_for, session, send_file, jsonify, send_from_directory
from segmentation_model import Food_Segmentation_model 
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
import numpy as np
from prepare_data import pre_img
from PIL import Image


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger('HELLO WORLD')

model_segment = Food_Segmentation_model()

UPLOAD_FOLDER = '/path/to/the/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def hello_world():
    return "Hello World"

@app.route('/upload', methods=['POST'])
@cross_origin()
def fileUpload():
    #read image file string data
    filestr = request.files['file'].read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    print('*'*20)
    print(img.shape)
    print('*'*20)
    image = pre_img(img)
    print(image.shape)
    y_pred = model_segment.predict(image)
    y_pred = model_segment.decode_segmap(y_pred)

    data = Image.fromarray(y_pred, "RGB")
    print(data)
    data.save('response.jpg')
    
    return send_file('response.jpg', 'image/jpg')

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(debug=True,host="0.0.0.0",use_reloader=False,port=8080,threaded=False)

flask_cors.CORS(app, expose_headers='Authorization')


