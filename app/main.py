from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io
import json
import os
import random
import requests
import re

app = FastAPI(title="Pet Disease Classifier API", version="1.0.0")

# CORS middleware - Fixed for your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://group042025.ceitesystems.com",
        "https://www.group042025.ceitesystems.com",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
class_mapping = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Your Google Drive model URLs
MODEL_URLS = {
    'proper_medical_model.pth': 'https://drive.google.com/uc?export=download&id=1whaZqfTGHg_60w4Yxct3x67N9JDFieXp',
    'proper_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1ym9R9KD6CBTUf9fVQuJWpPpEFQ00gDPq',
    'real_pet_disease_model.pth': 'https://drive.google.com/uc?export=download&id=1x4l-FHJ10q8JD20Mxmaff-WILkIgQ_td',
    'real_class_mapping.json': 'https://drive.google.com/uc?export=download&id=16WXwVVHAcny7yGTXeih9cPm4FL33p_SV'
}

def download_file_from_gdrive(file_id, destination):
    """Download file from Google Drive"""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle confirmation for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    if response.status_code != 200:
        return False
    
    # Download the file
    with open(destination, 'wb') as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)
    
    return os.path.exists(destination) and os.path.getsize(destination) > 0

def extract_file_id_from_url(url):
    """Extract file ID from Google Drive URL"""
    patterns = [
        r'/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'file/d/([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def download_model_files():
    """Download all model files from Google Drive"""
    print("🔄 Downloading models from Google Drive...")
    os.makedirs('models', exist_ok=True)
    
    for filename, url in MODEL_URLS.items():
        file_path = f'models/{filename}'
        
        if os.path.exists(file_path):
            print(f"✅ {filename} already exists")
            continue
            
        print(f"📥 Downloading {filename}...")
        file_id = extract_file_id_from_url(url)
        if file_id and download_file_from_gdrive(file_id, file_path):
            print(f"✅ Downloaded {filename}")
        else:
            print(f"❌ Failed to download {filename}")

def load_model():
    """Load the trained model - SIMPLIFIED like your local version"""
    global model, class_mapping
    
    try:
        # Try different model paths in order of preference
        model_paths = [
            ('models/proper_medical_model.pth', 'models/proper_class_mapping.json'),
            ('models/real_pet_disease_model.pth', 'models/real_class_mapping.json'),
            ('models/pet_disease_model.pth', 'models/class_mapping.json')
        ]
        
        model_path = None
        class_mapping_path = None
        
        for mp, cmp in model_paths:
            if os.path.exists(mp) and os.path.exists(cmp):
                model_path = mp
                class_mapping_path = cmp
                break
        
        if not model_path:
            print("❌ No model files found")
            return False
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        num_classes = len(class_mapping['idx_to_label'])
        print(f"📊 Number of classes: {num_classes}")
        
        # Create model architecture based on which model we're using
        if 'proper_medical' in model_path:
            model = models.efficientnet_b0(pretrained=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            print("🔬 Using EfficientNet (Medical Model)")
        else:
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print("🔧 Using ResNet18")
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"📋 Classes: {list(class_mapping['label_to_idx'].keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def get_demo_prediction(filename):
    """Generate demo predictions when model isn't loaded"""
    classes = [
        'Dental Disease in Cat', 'Dental Disease in Dog', 'distemper', 
        'Distemper in Dog', 'Ear Mites in Cat', 'ear_infection', 
        'Eye Infection in Cat', 'Eye Infection in Dog', 'Feline Leukemia',
        'Feline Panleukopenia', 'Fungal Infection in Cat', 'Fungal Infection in Dog',
        'healthy', 'Hot Spots in Dog', 'Kennel Cough in Dog', 'kennel_cough',
        'Mange in Dog', 'parvovirus', 'Parvovirus in Dog', 'Ringworm in Cat',
        'Scabies in Cat', 'Skin Allergy in Cat', 'Skin Allergy in Dog',
        'Tick Infestation in Dog', 'Urinary Tract Infection in Cat',
        'Worm Infection in Cat', 'Worm Infection in Dog'
    ]
    
    file_hash = hash(filename) % 100
    
    if file_hash < 15:
        primary_class = "Ear Mites in Cat"
        confidence = random.uniform(75, 90)
    elif file_hash < 30:
        primary_class = "Parvovirus in Dog"
        confidence = random.uniform(70, 85)
    elif file_hash < 45:
        primary_class = "Skin Allergy in Dog"
        confidence = random.uniform(65, 80)
    elif file_hash < 60:
        primary_class = "Dental Disease in Cat"
        confidence = random.uniform(60, 75)
    elif file_hash < 75:
        primary_class = "Kennel Cough in Dog"
        confidence = random.uniform(55, 70)
    else:
        primary_class = "healthy"
        confidence = random.uniform(80, 95)
    
    predictions = []
    for i, cls in enumerate(classes):
        if cls == primary_class:
            pred_confidence = confidence
        else:
            pred_confidence = random.uniform(1, 20)
        
        predictions.append({
            "class": cls,
            "confidence": round(pred_confidence, 2),
            "class_id": i
        })
    
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    return predictions[:5], predictions[0]

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("🚀 Starting Pet Disease Classifier API...")
    print("🌐 CORS enabled for: https://group042025.ceitesystems.com")
    
    # Download models first
    download_model_files()
    
    # Then load the model
    if load_model():
        print("🎉 API ready with trained model!")
    else:
        print("⚠️  Running in demo mode")

@app.get("/")
def root():
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None,
        "api_url": "https://vetcare-prediction-api-production.up.railway.app",
        "endpoints": {
            "health": "/health",
            "classes": "/classes", 
            "predict": "/predict",
            "batch_predict": "/predict-batch"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "classes_available": len(class_mapping['label_to_idx']) if class_mapping else 0
    }

@app.get("/classes")
def get_classes():
    if class_mapping:
        return {
            "success": True,
            "classes": list(class_mapping['label_to_idx'].keys()),
            "total_classes": len(class_mapping['label_to_idx'])
        }
    else:
        return {
            "success": True,
            "classes": [
                'Dental Disease in Cat', 'Dental Disease in Dog', 'Distemper in Dog',
                'Ear Mites in Cat', 'Eye Infection in Cat', 'Eye Infection in Dog',
                'Feline Leukemia', 'Fungal Infection in Cat', 'Fungal Infection in Dog',
                'healthy', 'Hot Spots in Dog', 'Kennel Cough in Dog', 'Mange in Dog',
                'Parvovirus in Dog', 'Ringworm in Cat', 'Skin Allergy in Cat',
                'Skin Allergy in Dog', 'Tick Infestation in Dog', 'Urinary Tract Infection in Cat',
                'Worm Infection in Cat', 'Worm Infection in Dog'
            ],
            "total_classes": 21
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Use demo mode if no model loaded
        if model is None:
            predictions, primary_prediction = get_demo_prediction(file.filename)
            return {
                "success": True,
                "predictions": predictions,
                "primary_prediction": primary_prediction,
                "file_name": file.filename,
                "message": "Demo mode - using sample predictions"
            }
        
        # Real prediction with trained model
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probabilities, min(5, len(class_mapping['idx_to_label'])))
        
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = class_mapping['idx_to_label'][str(idx.item())]
            predictions.append({
                "class": class_name,
                "confidence": round(prob.item() * 100, 2),
                "class_id": int(idx.item())
            })
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "message": "AI model prediction using trained weights"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict multiple images at once"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed")
    
    results = []
    for file in files:
        try:
            result = await predict(file)
            results.append({
                "file_name": file.filename,
                "success": True,
                "prediction": result["primary_prediction"]
            })
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
