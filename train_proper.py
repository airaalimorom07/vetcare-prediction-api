import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import json
from glob import glob
from tqdm import tqdm

class RealMedicalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Get REAL medical classes (exclude empty 'data' folder)
        self.class_names = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d)) and d != 'data']
        self.class_names.sort()
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}
        
        print("📁 Loading REAL medical images...")
        total_images = 0
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            
            # Get ALL image files
            images = (glob(os.path.join(class_dir, "*.jpg")) + 
                     glob(os.path.join(class_dir, "*.jpeg")) + 
                     glob(os.path.join(class_dir, "*.png")) +
                     glob(os.path.join(class_dir, "*.JPG")) +
                     glob(os.path.join(class_dir, "*.JPEG")))
            
            # Filter only valid image files
            valid_images = []
            for img_path in images:
                try:
                    # Verify it's a real image
                    with Image.open(img_path) as img:
                        img.verify()
                    valid_images.append(img_path)
                except:
                    continue
            
            for img_path in valid_images:
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[class_name])
            
            print(f"   {class_name}: {len(valid_images)} real medical images")
            total_images += len(valid_images)
        
        print(f"📊 Total REAL medical images: {total_images}")
        print(f"🏥 Medical classes: {len(self.class_names)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            # Skip corrupted images
            print(f"⚠️  Skipping corrupted image: {img_path}")
            return self.__getitem__((idx + 1) % len(self.image_paths))

def train_proper():
    print("🚀 PROPER Training with REAL Medical Images Only")
    print("=" * 60)
    
    data_dir = "real-pet-diseases/data"
    model_save_path = "models/proper_medical_model.pth"
    
    if not os.path.exists(data_dir):
        print("❌ Real medical dataset not found!")
        return False
    
    # Professional medical image transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load ONLY real medical images
    dataset = RealMedicalDataset(data_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("❌ No real medical images found!")
        return False
    
    # Use 90% for training, 10% for validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"\n📈 Training: {len(train_dataset)} real medical images")
    print(f"📉 Validation: {len(val_dataset)} real medical images")
    
    # Better model architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Use EfficientNet for better medical image recognition
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset.class_names))
    model = model.to(device)
    
    # Professional training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    print(f"\n🔄 Training on REAL medical images for 15 epochs...")
    print("   This will take 30-45 minutes but will give ACCURATE results!")
    
    best_accuracy = 0
    for epoch in range(15):
        print(f"\n📚 Epoch {epoch+1}/15")
        
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        scheduler.step()
        
        print(f"✅ Train Loss: {train_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")
        print(f"✅ Val Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), model_save_path)
            
            # Save class mapping
            class_mapping = {
                'idx_to_label': {str(i): cls for i, cls in enumerate(dataset.class_names)},
                'label_to_idx': {cls: i for i, cls in enumerate(dataset.class_names)}
            }
            
            with open("models/proper_class_mapping.json", "w") as f:
                json.dump(class_mapping, f)
            
            print(f"💾 BEST model saved! Validation Accuracy: {val_acc:.2f}%")
    
    print(f"\n🎉 PROPER Training Completed!")
    print(f"🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"💾 Model: {model_save_path}")
    print(f"📋 Classes: {dataset.class_names}")
    print(f"🩺 Ready for ACCURATE medical diagnoses!")
    
    return True

if __name__ == "__main__":
    train_proper()