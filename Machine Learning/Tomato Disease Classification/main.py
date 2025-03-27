from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    'http://localhost',
    'http://localhost:3000',]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    )

MODEL = tf.keras.models.load_model('Models/1.keras')
CLASS_NAMES = ['Early Blight','Late Blight','Healthy']

@app.get("/ping")

async def ping():
    return "Hello, I am alive. How are you? I am Nipuna Thennakoon"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')

async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    
    prediction = MODEL.predict(img_batch) 
    image_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]))
    
    return JSONResponse(content={
        'class': image_class,
        'confidence':confidence
    })

if __name__=="__main__":
    uvicorn.run(app,host='localhost',port=8000)