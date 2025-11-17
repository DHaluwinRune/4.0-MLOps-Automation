# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow import keras
# import os

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# ANIMALS = ["Cat", "Dog", "Panda"]
# model_path = os.path.join("animal-classification", "INPUT_model_path", "animal-cnn")
# # model = tf.keras.models.load_model(model_path)
# layer = keras.layers.TFSMLayer(
#     model_path, call_endpoint="serving_default", name="animal-cnn"
# )
# inp = keras.Input(shape=(64, 64, 3))
# out = layer(inp)
# model = keras.Model(inputs=inp, outputs=out)


# @app.post("/upload/image")
# async def uploadImage(img: UploadFile = File(...)):
#     original_image = Image.open(img.file)
#     resized_image = original_image.resize((64, 64))
#     images_to_predict = np.expand_dims(np.array(resized_image), axis=0)
#     predictions = model.predict(images_to_predict)
#     classifications = predictions.argmax(axis=1)
#     return ANIMALS[classifications.tolist()[0]]

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ANIMALS = ["Cat", "Dog", "Panda"]
# Geef het volledige pad naar het .keras bestand
model_path = os.path.join("animal-classification", "INPUT_model_path", "animal-cnn", "model.keras")
model = tf.keras.models.load_model(model_path)

@app.post("/upload/image")
async def uploadImage(img: UploadFile = File(...)):
    original_image = Image.open(img.file)
    resized_image = original_image.resize((64, 64))
    images_to_predict = np.expand_dims(np.array(resized_image), axis=0)
    predictions = model.predict(images_to_predict)
    classifications = predictions.argmax(axis=1)
    return ANIMALS[classifications.tolist()[0]]
