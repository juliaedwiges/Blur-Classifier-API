
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI()

session = ort.InferenceSession("model/blur_classifier.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

label_map = {0: "Sharp", 1: "Defocused", 2: "Motion"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess(image)
    result = session.run([output_name], {input_name: input_tensor})[0]
    predicted_class = int(np.argmax(result))
    label = label_map[predicted_class]
    return JSONResponse(content={"class": predicted_class, "label": label})
