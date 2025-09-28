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
    os.environ['MASTER_ADDR'] = 'localhost'  # Use the IP of the master node
    os.environ['MASTER_PORT'] = '12345'  # Port for communication
    os.environ['WORLD_SIZE'] = str(world_size)  # Total number of processes
    os.environ['RANK'] = str(rank)  # Rank of this process
    
    # Initialize the process group for distributed training
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',  # This will use the environment variables we set
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)  # Set the device for this rank

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

def train_model(rank, world_size, data_dir, epochs, batch_size, timings):
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

    # Prepare log file
    log_file = 'training_metrics_gpu_4.json'  # Log file for metrics

    # Initialize metrics storage
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    # Training loop
    total_training_time = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_preds = 0
        total_preds = 0
        all_labels = []
        all_preds = []
        start_time = time()

        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)

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
            
            # Collect labels and predictions for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # Calculate epoch-level accuracy
        epoch_accuracy = correct_preds / total_preds
        epoch_time = time() - start_time
        total_training_time += epoch_time

        # Calculate precision, recall, and F1 score
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

        # Store metrics for later
        accuracy_list.append(epoch_accuracy)
        precision_list.append(epoch_precision)
        recall_list.append(epoch_recall)
        f1_list.append(epoch_f1)

        # Print training statistics
        if rank == 0:
            print(f"Rank {rank}, Epoch {epoch+1}/{epochs}, "
                  f"Loss: {epoch_loss / len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}, "
                  f"Epoch Time: {epoch_time:.2f}s")

    # Cleanup DDP
    cleanup()

    if rank == 0:
        timings[rank] = total_training_time

    # Save metrics after all epochs are complete
    if rank == 0:
        results = {
            'timings': timings,
            'accuracy': accuracy_list,
            'precision': precision_list,
            'recall': recall_list,
            'f1': f1_list
        }
        
        results_file = 'training_metrics_gpu_4.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {results_file}")


def main():
    data_dir = '/home/mudavadkar.g/HPC-Project'  # Adjust the data path as needed
    batch_size = 32
    epochs = 10

    timings = {}  # Dictionary to store training times
    world_size = 4  # Ensure we only run with 4 GPUs

    # Spawn processes for DDP training on 4 GPUs
    mp.spawn(
        train_model,
        args=(world_size, data_dir, epochs, batch_size, timings),
        nprocs=world_size,
        join=True
    )
    
    # Print results for 4 GPUs training time
    total_training_time = sum(timings.values())
    print(f"\nTotal training time on 4 GPUs: {total_training_time:.2f}s")

if __name__ == "__main__":
    main()
