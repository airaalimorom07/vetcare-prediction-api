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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def load_model_safe():
    """Load model with comprehensive error handling"""
    global model, class_mapping
    
    try:
        logger.info("🔄 Attempting to load model...")
        
        # Load class mapping
        class_mapping_path = 'models/proper_class_mapping.json'
        if not os.path.exists(class_mapping_path):
            logger.error(f"❌ Class mapping file not found: {class_mapping_path}")
            return False
            
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        logger.info(f"📊 Loaded {len(class_mapping['idx_to_label'])} classes")
        
        # Load model weights
        model_path = 'models/proper_medical_model.pth'
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return False
        
        # Get number of classes
        num_classes = len(class_mapping['idx_to_label'])
        
        # Create model - use the same architecture as during training
        logger.info("🔧 Creating EfficientNet model...")
        model = models.efficientnet_b0(pretrained=False)
        
        # Update classifier for our number of classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        logger.info(f"📐 Model configured for {num_classes} classes")
        
        # Load the checkpoint
        logger.info("📦 Loading model weights...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✅ Loaded from model_state_dict")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                logger.info("✅ Loaded from state_dict")
            else:
                # Try direct loading
                model.load_state_dict(checkpoint)
                logger.info("✅ Loaded checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            logger.info("✅ Loaded checkpoint directly")
        
        # Move model to device and set to eval mode
        model.to(device)
        model.eval()
        
        logger.info("🎯 Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting Pet Disease Classifier API...")
    
    # Check if model files exist
    required_files = [
        'models/proper_medical_model.pth',
        'models/proper_class_mapping.json'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            logger.info(f"📁 {file_path}: {file_size} bytes")
        else:
            logger.error(f"❌ Missing file: {file_path}")
    
    # Load model
    if not load_model_safe():
        logger.error("💥 Failed to load model on startup")
    else:
        logger.info("✅ API started successfully with loaded model")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(device)
    }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "EfficientNet-B0",
        "num_classes": len(class_mapping['idx_to_label']),
        "classes": list(class_mapping['label_to_idx'].keys()),
        "device": str(device)
    }

@app.get("/debug")
async def debug_info():
    """Debug information"""
    model_files = {}
    for file_name in ['proper_medical_model.pth', 'proper_class_mapping.json']:
        path = f'models/{file_name}'
        exists = os.path.exists(path)
        model_files[file_name] = {
            'exists': exists,
            'size': os.path.getsize(path) if exists else 0
        }
    
    return {
        "model_loaded": model is not None,
        "model_files": model_files,
        "current_directory": os.getcwd(),
        "files_in_models": os.listdir('models') if os.path.exists('models') else []
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please try again later.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        logger.info(f"📨 Received prediction request for: {file.filename}")
        
        # Read image
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Open and validate image
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
        
        logger.info(f"📷 Image loaded: {image.size} {image.mode}")
        
        # Apply transformations
        try:
            input_tensor = transform(image).unsqueeze(0).to(device)
            logger.info(f"🔧 Input tensor shape: {input_tensor.shape}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")
        
        # Make prediction
        try:
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get top 5 predictions
                top5_probs, top5_indices = torch.topk(probabilities, min(5, len(class_mapping['idx_to_label'])))
            
            logger.info(f"📊 Prediction completed. Top confidence: {top5_probs[0][0].item()*100:.2f}%")
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
        
        # Format predictions
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = class_mapping['idx_to_label'][str(idx.item())]
            confidence = prob.item() * 100
            
            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "class_id": int(idx.item())
            })
        
        # Log top prediction
        top_pred = predictions[0]
        logger.info(f"🏆 Top prediction: {top_pred['class']} ({top_pred['confidence']}%)")
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "file_type": file.content_type
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during prediction: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/simple-predict")
async def simple_predict(file: UploadFile = File(...)):
    """Simplified prediction endpoint for testing"""
    if model is None:
        return {
            "success": False,
            "error": "Model not loaded",
            "demo_mode": True,
            "prediction": {
                "class": "healthy",
                "confidence": 85.0,
                "message": "Demo mode - model not loaded"
            }
        }
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_idx = torch.max(probabilities, 1)
        
        class_name = class_mapping['idx_to_label'][str(top_idx.item())]
        confidence = top_prob.item() * 100
        
        return {
            "success": True,
            "prediction": {
                "class": class_name,
                "confidence": round(confidence, 2)
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "demo_mode": True,
            "prediction": {
                "class": "unknown",
                "confidence": 0.0,
                "message": f"Error: {str(e)}"
            }
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
