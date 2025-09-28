import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time import time
import numpy as np
import json
from PIL import Image

# Set up DDP
def setup(rank, world_size):
    """
    Set up the distributed environment for DDP.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    """
    Clean up the distributed environment after training.
    """
    torch.distributed.destroy_process_group()

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

def train_model(rank, world_size, data_dir, epoch_losses, epoch_accuracies, lr_log, metrics_list, epochs=10, batch_size=32):
    setup(rank, world_size)

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image
        transforms.ToTensor(),          # Convert PIL Image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Load dataset
    dataset = CustomDataset(os.path.join(data_dir, 'train'), transform=transform)

    # Split dataset into training and validation subsets
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    
    # Validation split (same dataset, using a random sampler for simplicity)
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 3)  # 3 classes: melanoma, nevus, seborrheic_keratosis
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Mixed precision tools
    scaler = torch.cuda.amp.GradScaler()

    # Prepare log file
    log_file = 'training_metrics_mixed_precision.json'  # Log file for metrics

    # Store metrics
    epoch_losses = []
    epoch_accuracies = []
    lr_log = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        start_time = time()  # Start timing the epoch

        # Track learning rate per batch
        lr_values = []

        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)

            # Zero gradients
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Scale loss and perform backward pass
            scaler.scale(loss).backward()

            # Scale optimizer step
            scaler.step(optimizer)

            # Update scaler
            scaler.update()

            epoch_loss += loss.item()

            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_preds += labels.size(0)

            # Log learning rate (for each batch)
            lr_values.append(optimizer.param_groups[0]['lr'])

        # Calculate epoch-level accuracy
        epoch_accuracy = correct_preds / total_preds

        # Log metrics per epoch
        epoch_losses.append(epoch_loss / len(train_loader))
        epoch_accuracies.append(epoch_accuracy)

        # Log the learning rate per batch
        lr_log.append(np.mean(lr_values))

        # Calculate epoch time
        epoch_time = time() - start_time

        # Print training statistics
        print(f"Rank {rank}, Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}, "
              f"Accuracy: {epoch_accuracy:.4f}, Time: {epoch_time:.2f}s")

        # Log metrics to file
        metrics = {
            'epoch': epoch + 1,
            'loss': epoch_loss / len(train_loader),
            'accuracy': epoch_accuracy,
            'learning_rate': np.mean(lr_values),
            'epoch_time': epoch_time,
        }
        log_metrics(log_file, metrics)

    # Validate the model
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(rank), labels.to(rank)

            # Mixed precision validation pass
            with torch.cuda.amp.autocast():
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

    # Cleanup DDP
    cleanup()

    return epoch_losses, epoch_accuracies, lr_log, accuracy, precision, recall, f1

if __name__ == "__main__":
    data_dir = '/home/mudavadkar.g/HPC-Project'  # Adjust the data path as needed
    world_size = torch.cuda.device_count()  # Total number of GPUs

    # Shared variables for aggregation
    epoch_losses = []
    epoch_accuracies = []
    lr_log = []
    metrics_list = []

    # Spawn processes for DDP training
    mp.spawn(
        train_model,
        args=(world_size, data_dir, epoch_losses, epoch_accuracies, lr_log, metrics_list),
        nprocs=world_size,
        join=True
    )
