import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time
import numpy as np
import json
from PIL import Image
 
class CustomDataset(Dataset):
    """
    Custom dataset to enable usage of standard train-validation split.
    """
    def __init__(self, root_dir, transform=None):
        self.dataset = ImageFolder(root=root_dir)
        self.transform = transform
 
    def __len__(self):
        return len(self.dataset)
 
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)  # Apply transformation to the image
        return img, label
 
# Function to log training metrics to a JSON file
def log_metrics(log_file, metrics):
    with open(log_file, 'a') as f:
        json.dump(metrics, f)
        f.write('\n')
 
def train_model_1gpu(data_dir, epochs, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),          # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
 
    # Load dataset
    dataset = CustomDataset(os.path.join(data_dir, 'train'), transform=transform)
 
    # Split dataset into training and validation subsets
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
 
    # Define the model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes: melanoma, nevus, seborrheic_keratosis
    model = model.to(device)
 
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
 
    # Prepare log file
    log_file = 'training_metrics_1gpu.json'  # Log file for metrics
 
    # Training loop
    total_training_time = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        start_time = time()
 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
 
            # Zero gradients, perform a backward pass, and update weights
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            epoch_loss += loss.item()
 
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)
 
        # Calculate epoch-level accuracy
        epoch_accuracy = correct_preds / total_preds
        epoch_time = time() - start_time
        total_training_time += epoch_time
 
        # Print training statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Accuracy: {epoch_accuracy:.4f}, Epoch Time: {epoch_time:.2f}s")
 
        # Log metrics per epoch
        metrics = {
            'epoch': epoch + 1,
            'loss': epoch_loss / len(train_loader),
            'accuracy': epoch_accuracy,
            'epoch_time': epoch_time
        }
        log_metrics(log_file, metrics)
 
    # Validate the model
    model.eval()
    all_preds = []
    all_labels = []
 
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
 
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
    # Calculate metrics for the validation set
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
 
    # Print validation results
    print(f"Validation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1-Score: {f1:.4f}")
 
    # Save total training time
    timings = {'1 GPU': total_training_time}
    print(f"\n--- Results ---")
    print("Timings:", timings)
 
    # Save results to file
    results_file = 'results_1gpu.json'
    with open(results_file, 'w') as f:
        json.dump({'timings': timings, 'accuracy': accuracy, 'precision': precision, 
                   'recall': recall, 'f1': f1}, f)
 
    print(f"Results saved to {results_file}")
 
if __name__ == "__main__":
    data_dir = '/home/mudavadkar.g/HPC-Project'  # Adjust the data path as needed
    batch_size = 32
    epochs = 10
 
    print("Starting training on 1 GPU...")
    train_model_1gpu(data_dir, epochs, batch_size)