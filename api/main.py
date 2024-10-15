from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

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

MODEL = tf.keras.models.load_model("../models/1")

CLASS_NAMES = ["AD", "CN", "EMCI", "MCI"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    # Resize image to match model's expected input size
    image = image.resize((256, 170), Image.LANCZOS)  # (width, height)
    image = np.array(image)
    if len(image.shape) == 2:  # If grayscale, add channel dimension
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)  # Convert grayscale to RGB
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    try:
        image = read_file_as_image(await file.read())
        predictions = MODEL.predict(image)  # `image` already has the correct shape
        
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        return {
            'class': predicted_class,
            'confidence': float(confidence)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
