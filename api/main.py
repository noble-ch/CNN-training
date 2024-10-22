from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POTATO_MODEL = tf.keras.models.load_model("../saved-models/model.keras", compile=False)
POTATO_CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

NON_POTATO_MODEL = EfficientNetB0(weights='imagenet', include_top=True)

for layer in NON_POTATO_MODEL.layers[:200]:
    layer.trainable = False  

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert('RGB')
    image = image.resize((256, 256))  # Resize image
    image = np.array(image) / 255.0   # Rescale image
    return image

def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = Image.fromarray((image * 255).astype(np.uint8))  
    image = image.resize((224, 224))  
    image = preprocess_input(np.array(image))  
    return np.expand_dims(image, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())

    non_potato_image_batch = preprocess_image(image)
    non_potato_predictions = NON_POTATO_MODEL.predict(non_potato_image_batch)

    logger.info(f"EfficientNet Predictions: {non_potato_predictions}")

    decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(non_potato_predictions, top=5)
    logger.info(f"top 5 predictions from EfficientNet: {decoded_predictions}")

    potato_related_classes = [
        'leaf', 'plant', 'tree', 'flower', 'shrub', 'vegetation', 'botanical',
        'foliage', 'herb', 'grass', 'fern', 'vine', 'fruit'
    ]

    is_potato = any(
        any(potato_class in pred[1].lower() for potato_class in potato_related_classes)
        for pred in decoded_predictions[0]
    )

    if is_potato:
        img_batch = np.expand_dims(image, 0)  
        potato_predictions = POTATO_MODEL.predict(img_batch)

        predicted_potato_class_index = np.argmax(potato_predictions[0])
        predicted_class = POTATO_CLASS_NAMES[predicted_potato_class_index]
        potato_confidence = float(np.max(potato_predictions[0]))

        logger.info(f"Potato Model Predictions: {potato_predictions}, Confidence: {potato_confidence}")

        if potato_confidence < 0.7:
            predicted_class = "couldn't identify"
        confidence = potato_confidence  
    else:
        predicted_class = "couldn't identify"
        confidence = float(np.max(non_potato_predictions[0]))  

    return {
        'class': predicted_class,
        'confidence': confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
