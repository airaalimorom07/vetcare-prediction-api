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

app = FastAPI(title="Pet Disease Classifier API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def download_model_files():
    """Download model files from Google Drive if they don't exist"""
    print("🔄 Starting model file download...")
    os.makedirs('models', exist_ok=True)  # Changed to 'models' to match local
    
    # Your Google Drive direct download links
    model_files = {
        'models/proper_medical_model.pth': 'https://drive.google.com/uc?export=download&id=1UZRn58UHXKFZ38661xNDW1WU-QrHWa3b',
        'models/proper_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1PtWQq2Wk8IanKil6hsD_7RCG8IFnDcIe',
        'models/real_pet_disease_model.pth': 'https://drive.google.com/uc?export=download&id=1p2_wSpeNoftlByCLDcxdfk3nOG8rw9pN',
        'models/real_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1dw46v0t6sIbjVMAkCzuBIGDQL-KiWq-G'
    }
    
    for file_path, url in model_files.items():
        print(f"📥 Checking {file_path}...")
        if not os.path.exists(file_path):
            print(f"   Downloading from {url}...")
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ Downloaded {file_path} ({file_size} bytes)")
                else:
                    print(f"   ❌ Failed to download {file_path}: HTTP {response.status_code}")
            except Exception as e:
                print(f"   ❌ Error downloading {file_path}: {e}")
        else:
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {file_path} already exists ({file_size} bytes)")
    
    # List all files in models directory
    print("📁 Files in models directory:")
    if os.path.exists('models'):
        for file in os.listdir('models'):
            file_path = os.path.join('models', file)
            size = os.path.getsize(file_path)
            print(f"   - {file}: {size} bytes")
    else:
        print("   Models directory doesn't exist!")

def load_model():
    """Load the trained model"""
    global model, class_mapping
    
    try:
        # Try to load the PROPER medical model first
        model_path = 'models/proper_medical_model.pth'  # Changed to 'models'
        class_mapping_path = 'models/proper_class_mapping.json'  # Changed to 'models'
        
        # If proper model doesn't exist, fall back to real model
        if not os.path.exists(model_path):
            model_path = 'models/real_pet_disease_model.pth'  # Changed to 'models'
            class_mapping_path = 'models/real_class_mapping.json'  # Changed to 'models'
            print("⚠️  Proper medical model not found, using real model")
        else:
            print("✅ Proper medical model found, loading...")
        
        # If real model doesn't exist, fall back to demo model
        if not os.path.exists(model_path):
            model_path = 'models/pet_disease_model.pth'  # Changed to 'models'
            class_mapping_path = 'models/class_mapping.json'  # Changed to 'models'
            print("⚠️  Real model not found, using demo model")
        
        # Check if model files exist
        if not os.path.exists(model_path):
            print("❌ No model file found. Running in demo mode.")
            return False
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        # Get number of classes
        num_classes = len(class_mapping['idx_to_label'])
        
        # Create model - Use EfficientNet for proper medical model
        if 'proper' in model_path:
            model = models.efficientnet_b0(pretrained=False)  # FIXED: pretrained=False like local
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
            print("🔬 Using EfficientNet (medical optimized)")
        else:
            # Fallback to ResNet18 for other models
            model = models.resnet18(pretrained=False)  # FIXED: pretrained=False like local
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            print("🔧 Using ResNet18 (compatible)")
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        print("✅ Model loaded successfully!")
        print(f"📊 Classes: {list(class_mapping['label_to_idx'].keys())}")
        
        # Determine which model is being used
        if 'proper' in model_path:
            model_type = "PROPER MEDICAL"
        elif 'real' in model_path:
            model_type = "REAL" 
        else:
            model_type = "DEMO"
            
        print(f"💾 Using: {model_type} model")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️  Running in demo mode")
        return False

def get_demo_prediction(filename):
    """Generate demo predictions when model isn't loaded"""
    # Use the actual classes from your real dataset
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
    
    # Generate consistent "predictions" based on filename
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
    
    # Create predictions list
    predictions = []
    primary_idx = classes.index(primary_class)
    
    for i, cls in enumerate(classes):
        if cls == primary_class:
            pred_confidence = confidence
        else:
            pred_confidence = random.uniform(1, 20)  # Lower confidence for other classes
        
        predictions.append({
            "class": cls,
            "confidence": round(pred_confidence, 2),
            "class_id": i
        })
    
    # Sort by confidence (descending) and take top 5
    predictions.sort(key=lambda x: x['confidence'], reverse=True)
    predictions = predictions[:5]
    
    return predictions, predictions[0]

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("🚀 Starting Pet Disease Classifier API...")
    print(f"🌐 Environment: {os.getenv('ENVIRONMENT', 'production')}")
    print(f"🔧 PORT: {os.getenv('PORT', '8000')}")
    download_model_files()  # Download models first
    load_model()  # Then load them

@app.get("/")
def root():
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/health")
def health_check():
    model_type = "None"
    if model is not None:
        if 'efficientnet' in str(model.__class__).lower():
            model_type = "PROPER MEDICAL"
        elif 'resnet' in str(model.__class__).lower():
            model_type = "REAL"
        else:
            model_type = "DEMO"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "device": str(device),
        "classes_available": len(class_mapping['label_to_idx']) if class_mapping else 27,
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/model-info")
def model_info():
    """Check model information"""
    if model is None:
        return {"error": "No model loaded"}
    
    model_info = {
        "model_type": model.__class__.__name__,
        "model_architecture": "EfficientNet" if 'efficientnet' in str(model.__class__).lower() else "ResNet",
        "num_classes": len(class_mapping['idx_to_label']) if class_mapping else 0,
        "classes_loaded": list(class_mapping['label_to_idx'].keys()) if class_mapping else [],
        "pretrained_used": False  # This should now be False to match local
    }
    
    return {
        "model_info": model_info,
        "device": str(device)
    }

@app.get("/debug-files")
def debug_files():
    """Debug file system"""
    import os
    
    result = {
        "current_directory": os.getcwd(),
        "files_in_root": os.listdir('.'),
        "models_dir_exists": os.path.exists('models'),
    }
    
    if os.path.exists('models'):
        result["files_in_models"] = os.listdir('models')
        # Check each model file
        model_files = ['proper_medical_model.pth', 'proper_class_mapping.json']
        for file in model_files:
            path = f'models/{file}'
            result[file] = {
                "exists": os.path.exists(path),
                "size": os.path.getsize(path) if os.path.exists(path) else 0
            }
    
    return result

@app.get("/classes")
def get_classes():
    if class_mapping:
        return {"classes": list(class_mapping['label_to_idx'].keys())}
    else:
        # Return the real classes from your dataset
        return {"classes": [
            'Dental Disease in Cat', 'Dental Disease in Dog', 'distemper', 
            'Distemper in Dog', 'Ear Mites in Cat', 'ear_infection', 
            'Eye Infection in Cat', 'Eye Infection in Dog', 'Feline Leukemia',
            'Feline Panleukopenia', 'Fungal Infection in Cat', 'Fungal Infection in Dog',
            'healthy', 'Hot Spots in Dog', 'Kennel Cough in Dog', 'kennel_cough',
            'Mange in Dog', 'parvovirus', 'Parvovirus in Dog', 'Ringworm in Cat',
            'Scabies in Cat', 'Skin Allergy in Cat', 'Skin Allergy in Dog',
            'Tick Infestation in Dog', 'Urinary Tract Infection in Cat',
            'Worm Infection in Cat', 'Worm Infection in Dog'
        ]}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, etc.)")
    
    try:
        # If model is not loaded, use demo mode
        if model is None:
            predictions, primary_prediction = get_demo_prediction(file.filename)
            
            return {
                "success": True,
                "predictions": predictions,
                "primary_prediction": primary_prediction,
                "file_name": file.filename,
                "file_type": file.content_type,
                "message": "Demo mode - using sample predictions"
            }
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make real prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
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
        
        # Determine message based on model type
        model_type = "PROPER MEDICAL" if 'efficientnet' in str(model.__class__).lower() else "REAL"
        message = f"{model_type} model prediction - trained on medical images"
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "file_type": file.content_type,
            "message": message
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
            # Use the predict function for each file
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
    
    # Get port from environment variable (Railway provides this)
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        app, 
        host="0.0.0.0",  # Important for Railway - listen on all interfaces
        port=port,
        reload=os.getenv("ENVIRONMENT") == "development"  # Auto-reload only in development
    )
