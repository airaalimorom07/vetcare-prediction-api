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
import hashlib
import numpy as np

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

# Image transformations - MUST MATCH YOUR TRAINING
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

def load_model():
    """Load the trained model with proper error handling"""
    global model, class_mapping
    
    try:
        model_path = 'models/proper_medical_model.pth'
        class_mapping_path = 'models/proper_class_mapping.json'
        
        print("🔄 Loading model and class mapping...")
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
        
        print(f"📊 Loaded {len(class_mapping['idx_to_label'])} classes")
        
        # Create model architecture - MUST MATCH YOUR TRAINING
        num_classes = len(class_mapping['idx_to_label'])
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
        print("🔧 Model architecture created")
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        print(f"📦 Checkpoint type: {type(checkpoint)}")
        print(f"📦 Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                print("✅ Loaded from state_dict")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Loaded from model_state_dict")
            else:
                # Try to load directly
                model.load_state_dict(checkpoint)
                print("✅ Loaded checkpoint directly")
        else:
            model.load_state_dict(checkpoint)
            print("✅ Loaded checkpoint directly")
        
        model.to(device)
        model.eval()
        
        print("🎯 Model loaded successfully!")
        print(f"📋 Classes: {list(class_mapping['label_to_idx'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_image_characteristics(image):
    """Analyze image to help with prediction accuracy"""
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)
        
        characteristics = {
            'size': image.size,
            'mode': image.mode,
            'brightness': np.mean(img_array),
            'contrast': np.std(img_array),
            'color_balance': {
                'r': np.mean(img_array[:,:,0]) if len(img_array.shape) == 3 else 0,
                'g': np.mean(img_array[:,:,1]) if len(img_array.shape) == 3 else 0,
                'b': np.mean(img_array[:,:,2]) if len(img_array.shape) == 3 else 0,
            }
        }
        return characteristics
    except Exception as e:
        return {'error': str(e)}

@app.on_event("startup")
async def startup_event():
    """Load model when API starts"""
    print("🚀 Starting Pet Disease Classifier API...")
    if not load_model():
        print("💥 Failed to load model!")
    else:
        print("✅ API ready with trained model!")

@app.get("/")
def root():
    return {
        "message": "Pet Disease Classifier API", 
        "status": "running",
        "model_loaded": model is not None,
        "classes_loaded": len(class_mapping['idx_to_label']) if class_mapping else 0
    }

@app.get("/model-debug")
def model_debug():
    """Detailed model debugging information"""
    if model is None:
        return {"error": "Model not loaded"}
    
    # Test model with random input
    try:
        test_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            test_output = model(test_input)
            test_probs = torch.nn.functional.softmax(test_output, dim=1)
        
        debug_info = {
            "model_architecture": str(model.__class__.__name__),
            "model_device": str(next(model.parameters()).device),
            "num_classes": len(class_mapping['idx_to_label']),
            "test_output_shape": list(test_output.shape),
            "test_output_range": {
                "min": float(test_output.min()),
                "max": float(test_output.max()),
                "mean": float(test_output.mean())
            },
            "test_probs_range": {
                "min": float(test_probs.min()),
                "max": float(test_probs.max()),
                "sum": float(test_probs.sum())  # Should be ~1.0
            },
            "class_examples": list(class_mapping['idx_to_label'].values())[:10]
        }
        
        return debug_info
        
    except Exception as e:
        return {"error": f"Model test failed: {str(e)}"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict disease from uploaded image with detailed debugging"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Analyze image characteristics
        img_analysis = analyze_image_characteristics(image)
        print(f"📷 Image analysis: {img_analysis}")
        
        # Apply transformations
        input_tensor = transform(image).unsqueeze(0).to(device)
        print(f"🔧 Input tensor shape: {input_tensor.shape}")
        print(f"🔧 Input tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            print(f"📊 Raw outputs range: [{outputs.min():.3f}, {outputs.max():.3f}]")
            print(f"📊 Probabilities range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
            print(f"📊 Probabilities sum: {probabilities.sum().item():.3f}")
            
            # Get top 5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5)
            
            print(f"🏆 Top 5 probabilities: {[f'{p*100:.2f}%' for p in top5_probs[0].tolist()]}")
            print(f"🏆 Top 5 indices: {top5_indices[0].tolist()}")
        
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = class_mapping['idx_to_label'][str(idx.item())]
            confidence = prob.item() * 100
            
            predictions.append({
                "class": class_name,
                "confidence": round(confidence, 2),
                "class_id": int(idx.item())
            })
            
            print(f"🔍 Prediction: {class_name} - {confidence:.2f}%")
        
        # Check if predictions make sense
        max_confidence = predictions[0]['confidence']
        confidence_gap = max_confidence - (predictions[1]['confidence'] if len(predictions) > 1 else 0)
        
        prediction_quality = "HIGH" if max_confidence > 70 and confidence_gap > 20 else "LOW" if max_confidence < 30 else "MEDIUM"
        
        return {
            "success": True,
            "predictions": predictions,
            "primary_prediction": predictions[0],
            "file_name": file.filename,
            "image_analysis": img_analysis,
            "prediction_quality": prediction_quality,
            "debug_info": {
                "max_confidence": max_confidence,
                "confidence_gap": confidence_gap,
                "top_confidence": [p['confidence'] for p in predictions[:3]]
            }
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Predict multiple images with comparison"""
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 files allowed")
    
    results = []
    for file in files:
        try:
            result = await predict(file)
            results.append({
                "file_name": file.filename,
                "success": True,
                "prediction": result["primary_prediction"],
                "prediction_quality": result["prediction_quality"]
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

@app.get("/test-prediction")
async def test_prediction():
    """Test endpoint with a sample prediction"""
    try:
        # Create a random test image
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # Transform and predict
        input_tensor = transform(test_image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top5_probs, top5_indices = torch.topk(probabilities, 5)
        
        predictions = []
        for prob, idx in zip(top5_probs[0], top5_indices[0]):
            class_name = class_mapping['idx_to_label'][str(idx.item())]
            predictions.append({
                "class": class_name,
                "confidence": round(prob.item() * 100, 2),
                "class_id": int(idx.item())
            })
        
        return {
            "test_type": "random_red_image",
            "predictions": predictions,
            "model_output_stats": {
                "output_min": float(outputs.min()),
                "output_max": float(outputs.max()),
                "prob_sum": float(probabilities.sum())
            }
        }
        
    except Exception as e:
        return {"error": f"Test failed: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
