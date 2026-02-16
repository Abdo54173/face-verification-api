from fastapi import FastAPI, UploadFile, File
import shutil
import os
import torch
from torchvision import transforms, models
from PIL import Image
import uvicorn
import io
from embedder import FaceEmbedder
from verify import verify_embeddings
from utils import get_device


app =FastAPI(title ="Face Verification API")

device =get_device()
embedder =FaceEmbedder()

@app.post("/verify")
async def verify_faces(
    id_image: UploadFile = File(...),
    selfie_image: UploadFile = File(...),
    threshold: float = 0.5
):
    # Save uploaded files temporarily
    id_path = "temp_id.jpg"
    selfie_path = "temp_selfie.jpg"

    with open(id_path, "wb") as f:
        shutil.copyfileobj(id_image.file, f)

    with open(selfie_path, "wb") as f:
        shutil.copyfileobj(selfie_image.file, f)

    # Extract embeddings    
    id_emb =embedder.extract(id_path)
    selfie_emb =embedder.extract(selfie_path)

    # Verify
    is_match ,dist =verify_embeddings(id_emb ,selfie_emb ,threshold)

    # remove temp files
    os.remove(id_path)
    os.remove(selfie_path)

    return {
        "match" : is_match,
        "distance" : round(dist, 4),
        "threshold" : threshold
    }

#-----------------------------------------------------------
# check ID or NOT

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 2)

model.load_state_dict(torch.load("id_detector.pth", map_location=device))
model = model.to(device)
model.eval()


def predict(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

    label = predicted.item()
    return "ID" if label == 1 else "NOT_ID"


@app.post("/ID_check")
async def predict_api(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predict(img_bytes)
    return {"prediction": result}    