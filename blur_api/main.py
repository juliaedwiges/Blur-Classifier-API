
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import numpy as np
from PIL import Image
import onnxruntime as ort
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

app = FastAPI()

# Middleware para limitar o tamanho do upload
class LimitUploadSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_upload_size: int):
        super().__init__(app)
        self.max_upload_size = max_upload_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > self.max_upload_size:
            return JSONResponse(content={"detail": "Request too large"}, status_code=413)
        return await call_next(request)

# Ajuste aqui: exemplo → 10 MB
app.add_middleware(LimitUploadSizeMiddleware, max_upload_size=10 * 1024 * 1024)  # 10MB

# Modelo ONNX
session = ort.InferenceSession("model/blur_classifier.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

label_map = {0: "Sharp", 1: "Defocused", 2: "Motion"}

def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = preprocess(image)
    result = session.run([output_name], {input_name: input_tensor})[0]
    predicted_class = int(np.argmax(result))
    label = label_map[predicted_class]
    return JSONResponse(content={"class": predicted_class, "label": label})

