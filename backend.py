import os
import cv2
import io
# from segmentation_model import Food_Segmentation_model 
from segmentation_model_pytorch import Food_Segmentation_model_pytorch
from werkzeug.utils import secure_filename
import logging
import numpy as np
from prepare_data_pytorch import pre_img
from PIL import Image
from fastapi import FastAPI,Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import base64

from starlette.responses import StreamingResponse

model_segment = Food_Segmentation_model_pytorch()


class  ItemObject(BaseModel):
    item : str


h, w = 320, 320

app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:3000",
    "http://localhost:8080",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/food2020/food_segmentation")
async def upload(file: bytes = File(...)):
    with open('tmp.jpg', 'wb') as f:
        f.write(file)    
    mask = np.zeros((h,w))
    output = np.zeros((h,w))
    img = cv2.imread('tmp.jpg')

    image = pre_img(img)
    y_pred = model_segment.predict(image)


    y_pred = model_segment.decode_segmap(y_pred)
    
    data = Image.fromarray(y_pred, "RGB")
    print(data)
    data.save('before_post.jpg')
    y_pred2 = cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)
    res, im_jpg = cv2.imencode(".jpg", y_pred2)

    return StreamingResponse(io.BytesIO(im_jpg.tobytes()), media_type="image/jpg")
