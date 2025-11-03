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
import traceback

app = FastAPI(title="Pet Disease Classifier API", version="1.0.0")

# Add CORS middleware to allow requests from your PHP application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
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

# Correct Google Drive direct download URLs
MODEL_URLS = {
    'proper_medical_model.pth': 'https://drive.google.com/uc?export=download&id=1whaZqfTGHg_60w4Yxct3x67N9JDFieXp',
    'proper_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1ym9R9KD6CBTUf9fVQuJWpPpEFQ00gDPq',
    'real_pet_disease_model.pth': 'https://drive.google.com/uc?export=download&id=1x4l-FHJ10q8JD20Mxmaff-WILkIgQ_td',
    'real_class_mapping.json': 'https://drive.google.com/uc?export=download&id=16WXwVVHAcny7yGTXeih9cPm4FL33p_SV'
}

def download_file_from_gdrive(file_id, destination):
    """Download file from Google Drive with proper handling"""
    URL = "https://drive.google.com/uc?export=download"
    
    session = requests.Session()
    
    # Initial download request
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Check if we got a confirmation page for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            # If confirmation needed, add confirm parameter
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break
    
    # Check if response is successful
    if response.status_code != 200:
        print(f"   ❌ HTTP Error: {response.status_code}")
        return False
    
    # Check if we got an HTML page (error)
    content_type = response.headers.get('content-type', '')
    if 'text/html' in content_type:
        # Check for error message in HTML
        if 'Google Drive - Virus scan warning' in response.text:
            print("   ⚠️  Google Drive virus scan warning - cannot download automatically")
            return False
        elif 'Quota exceeded' in response.text:
            print("   ❌ Google Drive quota exceeded")
            return False
        else:
            print("   ❌ Got HTML page instead of file")
            return False
    
    # Get file size from headers
    total_size = int(response.headers.get('content-length', 0))
    
    # Download the file
    with open(destination, 'wb') as f:
        downloaded_size = 0
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                downloaded_size += len(chunk)
    
    # Verify download
    if os.path.exists(destination) and os.path.getsize(destination) > 0:
        print(f"   ✅ Downloaded {destination} ({os.path.getsize(destination)} bytes)")
        return True
    else:
        print(f"   ❌ Download failed or file is empty")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def extract_file_id_from_url(url):
    """Extract file ID from Google Drive URL"""
    # Handle different Google Drive URL formats
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
    """Download model files from Google Drive if they don't exist"""
    print("🔄 Starting model file download...")
    os.makedirs('models', exist_ok=True)
    
    all_success = True
    
    for filename, url in MODEL_URLS.items():
        file_path = f'models/{filename}'
        print(f"📥 Checking {filename}...")
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ Already exists ({file_size} bytes)")
            
            # Validate existing files
            if filename.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    print(f"   ✅ Valid JSON file")
                except json.JSONDecodeError:
                    print(f"   ❌ Corrupted JSON, redownloading...")
                    os.remove(file_path)
                    all_success = False
            continue
        
        print(f"   Downloading from {url}...")
        
        # Extract file ID and download
        file_id = extract_file_id_from_url(url)
        if not file_id:
            print(f"   ❌ Could not extract file ID from URL")
            all_success = False
            continue
            
        success = download_file_from_gdrive(file_id, file_path)
        if not success:
            all_success = False
    
    # List all files in models directory
    print("📁 Files in models directory:")
    if os.path.exists('models'):
        for file in os.listdir('models'):
            file_path = os.path.join('models', file)
            size = os.path.getsize(file_path)
            print(f"   - {file}: {size} bytes")
            
            # Validate JSON files
            if file.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        json.load(f)
                    print(f"   ✅ {file} is valid JSON")
                except json.JSONDecodeError as e:
                    print(f"   ❌ {file} is invalid JSON: {e}")
                    all_success = False
    else:
        print("   ❌ Models directory doesn't exist!")
        all_success = False
    
    return all_success

def load_model():
    """Load the trained model"""
    global model, class_mapping
    
    try:
        # Check if we have the required files for proper medical model
        model_path = 'models/proper_medical_model.pth'
        class_mapping_path = 'models/proper_class_mapping.json'
        
        if not os.path.exists(model_path):
            print("❌ Proper medical model file not found")
            return False
        
        if not os.path.exists(class_mapping_path):
            print("❌ Proper class mapping file not found")
            return False
        
        print("✅ Found proper medical model files, loading...")
        
        # Validate and load class mapping
        try:
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            print("✅ Class mapping loaded successfully")
        except json.JSONDecodeError as e:
            print(f"❌ Error loading class mapping: {e}")
            return False
        
        # Get number of classes
        num_classes = len(class_mapping['idx_to_label'])
        print(f"📊 Number of classes: {num_classes}")
        print(f"📋 Classes: {list(class_mapping['label_to_idx'].keys())}")
        
        # Create model - Use EfficientNet for proper medical model
        print("🔬 Creating EfficientNet model...")
        model = models.efficientnet_b0(pretrained=False)
        
        # Update the classifier for our number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        print(f"🔄 Updated classifier for {num_classes} classes")
        
        # Load trained weights
        try:
            # Check if model file is valid
            model_size = os.path.getsize(model_path)
            print(f"📦 Model file size: {model_size} bytes")
            
            if model_size < 1000:  # Too small to be a real model
                print(f"❌ Model file seems too small ({model_size} bytes)")
                return False
                
            print("🔄 Loading model weights...")
            # Use map_location to handle device compatibility
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            print("✅ Model weights loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model weights: {e}")
            # Try to get more detailed error info
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return False
        
        # Test the model with a dummy input to make sure it works
        try:
            print("🧪 Testing model with dummy input...")
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(device)
                output = model(dummy_input)
                print(f"✅ Model test passed! Output shape: {output.shape}")
        except Exception as e:
            print(f"❌ Model test failed: {e}")
            return False
        
        print("💾 Using: PROPER MEDICAL model")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"🔍 Detailed traceback: {traceback.format_exc()}")
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
    
    # Download models first
    download_success = download_model_files()
    
    if download_success:
        print("✅ All models downloaded successfully, loading...")
        model_loaded = load_model()
        if model_loaded:
            print("🎉 API ready with real model!")
        else:
            print("⚠️  API running in demo mode")
    else:
        print("❌ Model download failed, running in demo mode")

@app.get("/")
def root():
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None
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
        "classes_available": len(class_mapping['label_to_idx']) if class_mapping else 27
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
        "pretrained_used": False,
        "source": "Google Drive"
    }
    
    return {
        "model_info": model_info,
        "device": str(device)
    }

@app.get("/debug-files")
def debug_files():
    """Debug file system"""
    result = {
        "current_directory": os.getcwd(),
        "files_in_root": os.listdir('.'),
        "models_dir_exists": os.path.exists('models'),
    }
    
    if os.path.exists('models'):
        result["files_in_models"] = os.listdir('models')
        # Check each model file
        for file in ['proper_medical_model.pth', 'proper_class_mapping.json']:
            path = f'models/{file}'
            result[file] = {
                "exists": os.path.exists(path),
                "size": os.path.getsize(path) if os.path.exists(path) else 0
            }
    
    return result

@app.get("/download-status")
def download_status():
    """Check download status of all model files"""
    status = {}
    for filename, url in MODEL_URLS.items():
        file_path = f'models/{filename}'
        exists = os.path.exists(file_path)
        status[filename] = {
            "exists": exists,
            "size": os.path.getsize(file_path) if exists else 0,
            "url": url
        }
    
    return status

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
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "file_type": file.content_type,
            "message": "PROPER MEDICAL model prediction - trained on medical images"
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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
