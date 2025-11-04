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

def verify_model_file(model_path):
    """Verify if model file is valid"""
    import os
    if not os.path.exists(model_path):
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"📏 Model file size: {file_size} bytes")
    
    # A proper model should be at least 5MB
    if file_size < 5 * 1024 * 1024:  # 5MB
        print("❌ Model file appears corrupted - too small")
        return False
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                print("✅ Model contains state_dict")
            elif 'model' in checkpoint:
                print("✅ Model contains model key")
            else:
                print("✅ Model appears to be state_dict")
        else:
            print("✅ Model appears to be direct state_dict")
        return True
    except Exception as e:
        print(f"❌ Cannot load model file: {e}")
        return False

def download_model_files():
    """Download model files from Google Drive if they don't exist"""
    print("🔄 Starting model file download...")
    os.makedirs('models', exist_ok=True)
    
    # FIXED: Use direct download links for Google Drive
    model_files = {
        'models/proper_medical_model.pth': 'https://drive.google.com/uc?export=download&id=1x4l-FHJ10q8JD20Mxmaff-WILkIgQ_td',
        'models/proper_class_mapping.json': 'https://drive.google.com/uc?export=download&id=1PtWQq2Wk8IanKil6hsD_7RCG8IFnDcIe',
        'models/real_pet_disease_model.pth': 'https://drive.google.com/uc?export=download&id=1p2_wSpeNoftlByCLDcxdfk3nOG8rw9pN',
        'models/real_class_mapping.json': 'https://drive.google.com/uc?export=download&id=16WXwVVHAcny7yGTXeih9cPm4FL33p_SV'
    }
    
    for file_path, url in model_files.items():
        print(f"📥 Checking {file_path}...")
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
            print(f"   Downloading from {url}...")
            try:
                # Handle Google Drive virus scan warning for large files
                session = requests.Session()
                
                if 'real_pet_disease_model.pth' in file_path:
                    # For large files, we need to handle the confirmation
                    response = session.get(url, stream=True)
                    # Check if we got the virus scan warning page
                    if 'confirm=' in response.url:
                        # Extract confirmation token and make new request
                        confirm_token = None
                        for key, value in response.cookies.items():
                            if key.startswith('download_warning'):
                                confirm_token = value
                                break
                        
                        if confirm_token:
                            url = f"{url}&confirm={confirm_token}"
                            response = session.get(url, stream=True)
                
                else:
                    # For smaller files, direct download
                    response = session.get(url)
                
                if response.status_code == 200:
                    with open(file_path, 'wb') as f:
                        if 'real_pet_disease_model.pth' in file_path:
                            # Stream large file
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        else:
                            # Write small files directly
                            f.write(response.content)
                    
                    file_size = os.path.getsize(file_path)
                    print(f"   ✅ Downloaded {file_path} ({file_size} bytes)")
                    
                    # Verify the downloaded file
                    if '.pth' in file_path:
                        verify_model_file(file_path)
                else:
                    print(f"   ❌ Failed to download {file_path}: HTTP {response.status_code}")
                    # Create empty file as placeholder
                    with open(file_path, 'wb') as f:
                        f.write(b'')
                        
            except Exception as e:
                print(f"   ❌ Error downloading {file_path}: {e}")
                # Create empty file as placeholder
                with open(file_path, 'wb') as f:
                    f.write(b'')
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

def load_efficientnet_model(model_path, num_classes):
    """Load EfficientNet model using torchvision"""
    try:
        # Create EfficientNet model using torchvision
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel training)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # remove 'module.' if present
            new_state_dict[name] = v
            
        # Load with strict=False to handle architecture differences
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print("✅ EfficientNet model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading EfficientNet model: {e}")
        return None

def load_resnet_model(model_path, num_classes):
    """Load ResNet model with proper architecture"""
    try:
        # Create ResNet18 model
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        # Load state dict
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        
        print("✅ ResNet model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"❌ Error loading ResNet model: {e}")
        return None

def detect_model_architecture(model_path):
    """Detect what architecture the model was trained with"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Check keys to determine architecture
        first_key = next(iter(state_dict.keys()))
        
        if 'efficientnet' in first_key.lower() or 'features.0.0.weight' in first_key:
            return 'efficientnet'
        elif 'resnet' in first_key.lower() or 'conv1.weight' in first_key:
            return 'resnet'
        elif 'features' in first_key:
            return 'efficientnet'  # Likely efficientnet
        else:
            return 'unknown'
            
    except Exception as e:
        print(f"❌ Error detecting architecture: {e}")
        return 'unknown'

def load_model():
    """Load the trained model"""
    global model, class_mapping
    
    try:
        # Try to load the PROPER medical model first (EfficientNet)
        model_path = 'models/proper_medical_model.pth'
        class_mapping_path = 'models/proper_class_mapping.json'
        
        # Check if files are valid
        if (os.path.exists(model_path) and verify_model_file(model_path) and
            os.path.exists(class_mapping_path) and os.path.getsize(class_mapping_path) > 10):
            
            print("✅ Proper medical model found, loading...")
            
            # Load class mapping
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            
            # Get number of classes
            num_classes = len(class_mapping['idx_to_label'])
            
            print("🔬 Using EfficientNet (medical optimized)")
            model = load_efficientnet_model(model_path, num_classes)
            
            if model is not None:
                print("✅ Medical model loaded successfully!")
                print(f"📊 Classes: {list(class_mapping['label_to_idx'].keys())}")
                return True
            else:
                print("❌ Failed to load medical model")
                
        else:
            print("❌ Proper medical model files are invalid or empty")
            
    except Exception as e:
        print(f"❌ Error loading proper medical model: {e}")
    
    # Try to load the REAL model as fallback
    try:
        model_path = 'models/real_pet_disease_model.pth'
        class_mapping_path = 'models/real_class_mapping.json'
        
        if (os.path.exists(model_path) and verify_model_file(model_path) and
            os.path.exists(class_mapping_path) and os.path.getsize(class_mapping_path) > 10):
            
            print("🔄 Falling back to real model...")
            
            # Load class mapping
            with open(class_mapping_path, 'r') as f:
                class_mapping = json.load(f)
            
            # Get number of classes
            num_classes = len(class_mapping['idx_to_label'])
            
            # Detect architecture first
            architecture = detect_model_architecture(model_path)
            print(f"🔍 Detected architecture: {architecture}")
            
            if architecture == 'efficientnet':
                print("🔬 Loading as EfficientNet...")
                model = load_efficientnet_model(model_path, num_classes)
            elif architecture == 'resnet':
                print("🔧 Loading as ResNet...")
                model = load_resnet_model(model_path, num_classes)
            else:
                # Try both architectures
                print("🔄 Architecture unknown, trying EfficientNet first...")
                model = load_efficientnet_model(model_path, num_classes)
                if model is None:
                    print("🔄 Trying ResNet...")
                    model = load_resnet_model(model_path, num_classes)
            
            if model is not None:
                print("✅ Real model loaded successfully!")
                return True
            else:
                print("❌ Failed to load real model with any architecture")
                
        else:
            print("❌ Real model files are invalid or empty")
            
    except Exception as e:
        print(f"❌ Error loading real model: {e}")
    
    print("⚠️  Running in demo mode - no valid model files found")
    return False

def predict_with_confidence(image_tensor, confidence_threshold=0.6):
    """Make prediction with confidence checking"""
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence = confidence.item()
        
        class_idx = predicted.item()
        class_name = class_mapping['idx_to_label'].get(str(class_idx), "Unknown")
        
        # If confidence is too low, mark as uncertain
        if confidence < confidence_threshold:
            return "Uncertain - Low Confidence", confidence, class_idx
        
        return class_name, confidence, class_idx

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
    download_model_files()  # Download models first
    load_model()  # Then load them

@app.get("/")
def root():
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None,
        "demo_mode": model is None
    }

@app.get("/health")
def health_check():
    model_type = "None"
    if model is not None:
        if 'efficientnet' in str(model.__class__).lower():
            model_type = "EfficientNet"
        elif 'resnet' in str(model.__class__).lower():
            model_type = "ResNet"
        else:
            model_type = "Unknown"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": model_type,
        "demo_mode": model is None,
        "device": str(device),
        "classes_available": len(class_mapping['label_to_idx']) if class_mapping else 27
    }

@app.get("/model-info")
def model_info():
    """Check model information"""
    if model is None:
        return {
            "error": "No model loaded", 
            "demo_mode": True,
            "message": "Running in demo mode with simulated predictions"
        }
    
    model_info = {
        "model_type": model.__class__.__name__,
        "model_architecture": "EfficientNet" if 'efficientnet' in str(model.__class__).lower() else "ResNet",
        "num_classes": len(class_mapping['idx_to_label']) if class_mapping else 0,
        "classes_loaded": list(class_mapping['label_to_idx'].keys()) if class_mapping else [],
        "pretrained_used": False
    }
    
    return {
        "model_info": model_info,
        "device": str(device),
        "demo_mode": False
    }

@app.get("/debug-files")
def debug_files():
    """Debug file system"""
    import os
    
    result = {
        "current_directory": os.getcwd(),
        "files_in_root": os.listdir('.'),
        "models_dir_exists": os.path.exists('models'),
        "demo_mode": model is None
    }
    
    if os.path.exists('models'):
        result["files_in_models"] = os.listdir('models')
        # Check each model file
        model_files = ['proper_medical_model.pth', 'proper_class_mapping.json', 
                      'real_pet_disease_model.pth', 'real_class_mapping.json']
        for file in model_files:
            path = f'models/{file}'
            result[file] = {
                "exists": os.path.exists(path),
                "size": os.path.getsize(path) if os.path.exists(path) else 0,
                "valid": verify_model_file(path) if os.path.exists(path) else False
            }
    
    return result

@app.get("/model-diagnostic")
def model_diagnostic():
    """Detailed model diagnostic information"""
    model_path = "models/real_pet_disease_model.pth"
    
    try:
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            architecture = detect_model_architecture(model_path)
            
            return {
                "file_exists": True,
                "file_size": os.path.getsize(model_path),
                "file_valid": verify_model_file(model_path),
                "detected_architecture": architecture,
                "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "Direct state_dict",
                "model_loaded": model is not None,
                "model_architecture": model.__class__.__name__ if model else "None"
            }
        else:
            return {"error": "Model file not found", "file_exists": False}
    except Exception as e:
        return {"error": str(e), "file_exists": os.path.exists(model_path)}

@app.get("/classes")
def get_classes():
    if class_mapping:
        return {
            "classes": list(class_mapping['label_to_idx'].keys()),
            "demo_mode": False
        }
    else:
        # Return the real classes from your dataset
        return {
            "classes": [
                'Dental Disease in Cat', 'Dental Disease in Dog', 'distemper', 
                'Distemper in Dog', 'Ear Mites in Cat', 'ear_infection', 
                'Eye Infection in Cat', 'Eye Infection in Dog', 'Feline Leukemia',
                'Feline Panleukopenia', 'Fungal Infection in Cat', 'Fungal Infection in Dog',
                'healthy', 'Hot Spots in Dog', 'Kennel Cough in Dog', 'kennel_cough',
                'Mange in Dog', 'parvovirus', 'Parvovirus in Dog', 'Ringworm in Cat',
                'Scabies in Cat', 'Skin Allergy in Cat', 'Skin Allergy in Dog',
                'Tick Infestation in Dog', 'Urinary Tract Infection in Cat',
                'Worm Infection in Cat', 'Worm Infection in Dog'
            ],
            "demo_mode": True
        }

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
                "message": "Demo mode - using sample predictions",
                "demo_mode": True
            }
        
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make real prediction with confidence checking
        primary_class, confidence, class_idx = predict_with_confidence(input_tensor)
        
        # Get top 5 predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
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
        model_type = "EfficientNet" if 'efficientnet' in str(model.__class__).lower() else "ResNet"
        message = f"{model_type} model prediction"
        
        # Add confidence warning if low
        if confidence < 0.6:
            message += " - Low confidence prediction"
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "file_type": file.content_type,
            "message": message,
            "demo_mode": False,
            "confidence_warning": confidence < 0.6
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
                "prediction": result["primary_prediction"],
                "demo_mode": result.get("demo_mode", False)
            })
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "success": False,
                "error": str(e),
                "demo_mode": True
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "results": results,
        "demo_mode": model is None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
