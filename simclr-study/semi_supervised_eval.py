import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

class LinearProbingModel(nn.Module):
    def __init__(self, backbone_path, num_classes=10):
        super(LinearProbingModel, self).__init__()
        # 1. Load ResNet18 backbone and remove final FC layer
        self.encoder = resnet18(weights=None)
        in_features = self.encoder.fc.in_features  # 512
        self.encoder.fc = nn.Identity()

        # 2. Load pre-trained SimCLR backbone weights
        if not os.path.exists(backbone_path):
            raise FileNotFoundError(f"Checkpoint not found at: {backbone_path}")
        self.encoder.load_state_dict(torch.load(backbone_path, map_location='cpu'))

        # 3. Freeze backbone weights
        for param in self.encoder.parameters():
            param.requires_grad = False

        # 4. Trainable linear classifier layer
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.fc(features)

def get_stratified_subset(dataset, samples_per_class=500):
    # Sample exactly 500 items per class (10% of CIFAR-10)
    class_counts = {i: 0 for i in range(10)}
    subset_indices = []
    
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        if class_counts[label] < samples_per_class:
            subset_indices.append(idx)
            class_counts[label] += 1
            if len(subset_indices) == samples_per_class * 10:
                break
    return Subset(dataset, subset_indices)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Executing 10% Semi-Supervised Linear Probing on: {device}")

    # Standard data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    data_dir = '../vision-resnet-study/data' if os.path.exists('../vision-resnet-study/data') else './data'
    full_trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=transform_test)

    # Sample only 10% (5,000 samples)
    subset_trainset = get_stratified_subset(full_trainset, samples_per_class=500)
    print(f"[+] Labeled Training Data: {len(subset_trainset)} samples (10% of CIFAR-10)")

    trainloader = DataLoader(subset_trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Initialize model, criterion, optimizer
    model = LinearProbingModel('weights/simclr_backbone.pth', num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-2, weight_decay=1e-4)

    epochs = 5
    print("[*] Training Classifier on 10% Labels...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate on full 10,000 test set
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        print(f"[*] Epoch {epoch+1} - Loss: {train_loss/len(trainloader):.4f} | 10% Label Test Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()